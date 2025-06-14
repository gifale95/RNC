"""Generative univariate RNC iteratively generates images following two serial
objectives. Throughout the genetic optimization generations, the generated
images first drive or suppress the in-silico univariate fMRI responses of two
areas up to a threshold. Once this threshold is reached, the image complexity
– as measured by their PNG compression file size – starts to monotonically
decrease, while keeping the in-silico univariate fMRI responses over the
threshold, thus promoting the generation of images containing only the visual
properties necessary to align or disentangle the two areas.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which the in
	silico fMRI responses can be generated.
cv : int
	If '1' univariate RNC leaves the data of one subject out for
	cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
	If 'cv==0' the left-out subject during cross-validation, out of all 8 (NSD)
	subjects.
roi_pair : str
	Used pairwise ROI combination.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
control_type : str
	If 'high_1_high_2', generate images that drive both ROIs. If
	'high_1_low_2', generate images that drive the first ROI while suppressing
	the second ROI. If 'low_1_high_2', generate images that suppress the first
	ROI while driving the second ROI. If 'low_1_low_2', generate images that
	suppress both ROIs.
evolution : int
	Genetic optimization evolution. At each evolution the genetic optimization
	starts from a different random seed, resulting in different controlling
	images.
generations : int
	Number of gemetic optimization generations.
n_image_codes : int
	Number of image code, indicading how many images are generated and evaluated
	at each generation.
image_generator_name : str
	Name of the used image generator. Available options are 'DeePSiM' (a GAN).
	or 'cd_imagenet64_l2' (a class-conditioned diffusion model trained on the
	1000 ILSVRC-2012 classes).
image_generator_class : int
	Integer between 0 and 999 indicating the ILSVRC-2012 class the generated
	image belongs to (if image_generator_name=='cd_imagenet64_l2').
baseline_margin : float
	Margin to be added to the baseline univariate response score to define the
	neural control threshold. If set to None, keep on optimizing the neural
	control scores, without optimizing image complexity.
img_complexity_measure : str
	How to compute image complexity. Possbile methods are ['png', 'jpg'].
frac_kept_image_codes : float
	Fraction [0 1] of best image codes that passed onto the next generation
	without being recombined or mutated.
heritability : float
	Fraction [0 1] determining how much one image code parent (of 2) contributes
	to each image code child.
mutation_prob : float
	Probability [0 1] of each new image code genes to be mutated.
imageset : str
	Imageset from which the univariate RNC baseline scores have been computed.
	Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.
berg_dir : str
	Directory of the Brain Encoding Response Generator.
	https://github.com/gifale95/BERG

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image
from copy import copy

from utils import load_encoding_models
from utils import load_image_generator
from utils import generate_insilico_fmri
from utils import score_select
from utils import optimize_image_codes

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi_pair', type=str, default='V1-hV4')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--control_type', type=str, default='high_1_high_2')
parser.add_argument('--generations', type=int, default=500)
parser.add_argument('--evolution', type=int, default=1)
parser.add_argument('--n_image_codes', type=int, default=1000)
parser.add_argument('--image_generator_name', type=str, default='DeePSiM')
parser.add_argument('--image_generator_class', type=int, default=0)
parser.add_argument('--baseline_margin', type=float, default=0.6)
parser.add_argument('--img_complexity_measure', type=str, default='png')
parser.add_argument('--frac_kept_image_codes', type=float, default=.25)
parser.add_argument('--heritability', type=float, default=.25)
parser.add_argument('--mutation_prob', type=float, default=.25)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-reponse-generator/', type=str)
args = parser.parse_args()

print('>>> Generative univariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Random seed and device
# =============================================================================
# Set random seed for reproducible results
seed = args.evolution
np.random.seed(seed)
random.seed(seed)
random_generator = np.random.RandomState(seed=seed)
torch.manual_seed(seed)

# Compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]


# =============================================================================
# Load the univariate RNC baseline
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	'nsd_encoding_models', 'baseline', 'cv-'+format(args.cv), 'imageset-'+
	args.imageset)

if args.cv == 0:
	file_name_1 = 'baseline_roi-' + roi_1 + '.npy'
	file_name_2 = 'baseline_roi-' + roi_2 + '.npy'
	baseline_roi_1 = np.load(os.path.join(data_dir, file_name_1),
		allow_pickle=True).item()['baseline_images_score']
	baseline_roi_2 = np.load(os.path.join(data_dir, file_name_2),
		allow_pickle=True).item()['baseline_images_score']

elif args.cv == 1:
	file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02') + \
		'_roi-' + args.roi + '.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	baseline_roi_1 = np.load(os.path.join(data_dir, file_name_1),
		allow_pickle=True).item()['baseline_images_score_train']
	baseline_roi_2 = np.load(os.path.join(data_dir, file_name_2),
		allow_pickle=True).item()['baseline_images_score_train']


# =============================================================================
# Load the encoding models of all subjects
# =============================================================================
encoding_models_roi_1, metadata_roi_1 = load_encoding_models(args, roi_1,
	device)
encoding_models_roi_2, metadata_roi_2 = load_encoding_models(args, roi_2,
	device)


# =============================================================================
# Import the image generator
# =============================================================================
image_generator = load_image_generator(args, device)

if args.image_generator_name == 'DeePSiM':
	model_save_dir = 'image_generator-' + args.image_generator_name
elif args.image_generator_name == 'cd_imagenet64_l2':
	model_save_dir = 'image_generator-' + args.image_generator_name + '/' + \
		'image_generator_class-' + format(args.image_generator_class+1, '04')
	random_generator_diffusion = torch.Generator(device=device)


# =============================================================================
# Initialize the image codes
# =============================================================================
# Get the image codes dimensionality
if args.image_generator_name == 'DeePSiM':
	image_code_size = (args.n_image_codes, image_generator.fc7.in_features)
elif args.image_generator_name == 'cd_imagenet64_l2':
	image_code_size = (args.n_image_codes, 3, 64, 64)

# Randomly initialize image codes from a normal distributions
image_codes_new = random_generator.normal(loc=0, scale=1, size=image_code_size)

# Number of kept image codes at each generation
n_kept = int(len(image_codes_new) * args.frac_kept_image_codes)
images_kept = np.empty(0)
image_codes_kept = np.empty(0)
fmri_roi_1_kept = np.empty(0)
fmri_roi_2_kept = np.empty(0)


# =============================================================================
# Results variables
# =============================================================================
# Neural control scores
best_neural_control_scores_train = np.zeros(args.generations, dtype=np.float32)
best_neural_control_scores_test = np.zeros(args.generations, dtype=np.float32)
# Penalty scores
best_baseline_penalty_train = np.zeros(args.generations, dtype=np.float32)
best_baseline_penalty_test = np.zeros(args.generations, dtype=np.float32)
# Image complexity scores
best_images_complexity = np.zeros(args.generations, dtype=np.float32)
# Total scores
best_scores_train = np.zeros(args.generations, dtype=np.float32)
best_scores_test = np.zeros(args.generations, dtype=np.float32)
# Image codes
total_image_code_size = (args.generations,) + image_code_size
best_image_codes = np.zeros(total_image_code_size, dtype=np.float32)
# In silico fMRI responses
best_fmri_roi_1 = np.zeros((args.generations, len(args.all_subjects)),
	dtype=np.float32)
best_fmri_roi_2 = np.zeros((args.generations, len(args.all_subjects)),
	dtype=np.float32)


# =============================================================================
# Generate images from the image codes
# =============================================================================
# Generation loop
for g in tqdm(range(args.generations), leave=False):

	img_codes = torch.FloatTensor(copy(image_codes_new))
#	img_codes.to(device)

	# Generate the images using a GAN (DeePSiM)
	if args.image_generator_name == 'DeePSiM':
		# Generate the images
		images_new = image_generator.forward(img_codes).detach().numpy()
		# Clip and scale the images synthesized by the image generator: clamp
		# the output image pixel values to the range [0 255]
		images_new = np.clip(images_new, a_min=0, a_max=255)
		# ======
		# Version 2 (as in Ponce et al., 2019):
		# """To synthesize an image from an input image code, we forward
		# propagated the code through the generative network, clamped the
		# output image pixel values to the valid range between 0 and 1, and
		# visualized them as an 8-bit color image."""
		#images_new = np.clip(images_new, a_min=0, a_max=1) * 255
		# ======
		# Version 3:
		# Clamp the output image pixel values to the range [-255 255],
		# normalize them in the range [0 1], and scale them to the range
		# [0 255]
		#images_new = np.clip(images_new, a_min=-255, a_max=255)
		#images_new = (images_new - np.min(images_new.flatten())) / \
		#	(np.max(images_new.flatten()) - np.min(images_new.flatten())) * 255

	# Generate the images using a diffusion model (cd_imagenet64_l2)
	elif args.image_generator_name == 'cd_imagenet64_l2':
		# Generate the image codes into two batches (for GPU RAM)
		batch_n = 2
		batch_size = int(np.ceil(len(img_codes) / batch_n))
		class_labels = [args.image_generator_class] * batch_size
		for b in range(batch_n):
			idx_start = batch_size * b
			idx_end = idx_start + batch_size
			# Set a constant random seed to enforce a deterministic image
			# generation
			#torch.manual_seed(seed) # Used to determine the image class
			random_generator_diffusion.manual_seed(seed) # Used to determine the image style
			# Generate the images
			with torch.inference_mode():
				images_new_batch = image_generator(
					batch_size=batch_size,
					class_labels=class_labels,
					num_inference_steps=40,
					generator=random_generator_diffusion,
					latents=img_codes[idx_start:idx_end],
					output_type='np'
					).images
				if b == 0:
					images_new = images_new_batch
				else:
					images_new = np.append(images_new, images_new_batch, 0)
				del images_new_batch
		# Reshape to (Batch size x 3 RGB Channels x Width x Height)
		images_new = np.transpose(images_new, (0, 3, 1, 2))
		# Scale to the range [0, 255]
		images_new *= 255

	# Convert the images to uint8
	images_new = images_new.astype(np.uint8)
	del img_codes


# =============================================================================
# Generate in silico fMRI responses for the synthesized images
# =============================================================================
	fmri_roi_1_new = generate_insilico_fmri(args, encoding_models_roi_1,
		metadata_roi_1, copy(images_new), device)
	fmri_roi_2_new = generate_insilico_fmri(args, encoding_models_roi_2,
		metadata_roi_2, copy(images_new), device)


# =============================================================================
# Add kept data from the previous generation
# =============================================================================
	if g == 0:
		image_codes = image_codes_new
		images = images_new
		fmri_roi_1 = fmri_roi_1_new
		fmri_roi_2 = fmri_roi_2_new
	else:
		image_codes = np.append(image_codes_kept, image_codes_new, 0)
		images = np.append(images_kept, images_new, 0)
		fmri_roi_1 = np.append(fmri_roi_1_kept, fmri_roi_1_new, 1)
		fmri_roi_2 = np.append(fmri_roi_2_kept, fmri_roi_2_new, 1)

	del image_codes_kept, image_codes_new, images_kept, images_new, \
		fmri_roi_1_kept, fmri_roi_1_new, fmri_roi_2_kept, \
		fmri_roi_2_new


# =============================================================================
# Compute the neural control scores, and select the image codes accordingly
# =============================================================================
	# Score the generated images, rank the scores, and then select/store the
	# image codes of the best N images
	scores_train, scores_test, neural_control_scores_train, \
		neural_control_scores_test, baseline_penalty_train, \
		baseline_penalty_test, images_complexity, image_codes, fmri_roi_1, \
		fmri_roi_2, images = score_select(args, fmri_roi_1, fmri_roi_2,
		image_codes, images, baseline_roi_1, baseline_roi_2)

	# Save the best scores, image codes and fMRI responses of each generation
	best_scores_train[g] = scores_train[0]
	best_scores_test[g] = scores_test[0]
	best_neural_control_scores_train[g] = neural_control_scores_train[0]
	best_neural_control_scores_test[g] = neural_control_scores_test[0]
	best_baseline_penalty_train[g] = baseline_penalty_train[0]
	best_baseline_penalty_test[g] = baseline_penalty_test[0]
	best_images_complexity[g] = images_complexity[0]
	best_image_codes[g] = image_codes[0]
	best_fmri_roi_1[g] = fmri_roi_1[:,0]
	best_fmri_roi_2[g] = fmri_roi_2[:,0]


# =============================================================================
# Store the results from the kept image codes to reduce computation
# =============================================================================
	image_codes_kept = copy(image_codes[:n_kept])
	images_kept = copy(images[:n_kept])
	fmri_roi_1_kept = copy(fmri_roi_1[:,:n_kept])
	fmri_roi_2_kept = copy(fmri_roi_2[:,:n_kept])
	del images, fmri_roi_1, fmri_roi_2


# =============================================================================
# Optimize the image codes using a genetic algorithm
# =============================================================================
	image_codes_new = optimize_image_codes(args, image_codes,
		copy(scores_train), random_generator)
	del image_codes


# =============================================================================
# Save the best image of every Nth generation
# =============================================================================
	if args.cv == 0 and (g+1) % 1 == 0:

		if args.cv == 0:
			save_dir = os.path.join(args.project_dir,
				'generative_univariate_rnc', 'controlling_images', 'cv-'+
				format(args.cv), roi_1+'-'+roi_2, 'control_condition-'+
				args.control_type, model_save_dir, 'evolution-'+
				format(args.evolution, '02'), 'baseline_margin-'+
				str(args.baseline_margin))
		elif args.cv == 1:
			save_dir = os.path.join(args.project_dir,
				'generative_univariate_rnc', 'controlling_images', 'cv-'+
				format(args.cv), roi_1+'-'+roi_2, 'control_condition-'+
				args.control_type, 'cv_subject-'+format(args.cv_subject, '02'),
				model_save_dir, 'evolution-'+format(args.evolution, '02'),
				'baseline_margin-'+str(args.baseline_margin))

		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)

		for i in range(args.n_image_codes):
			img = Image.fromarray(np.swapaxes(np.swapaxes(
				images_kept[0], 0, 1), 1, 2))
			file_name = 'gan_img_' + args.control_type + \
				'_generation-' + format(g+1, '05') + '_null_penalty-' + \
				str(best_baseline_penalty_train[g]) + '_complexity-' + \
				format(best_images_complexity[g], '08') + '.png'
			img.save(os.path.join(save_dir, file_name))


# =============================================================================
# Save the optimization scores
# =============================================================================
data_dict = {
	'args': args,
	'best_scores_train': best_scores_train,
	'best_scores_test': best_scores_test,
	'best_neural_control_scores_train': best_neural_control_scores_train,
	'best_neural_control_scores_test': best_neural_control_scores_test,
	'best_baseline_penalty_train': best_baseline_penalty_train,
	'best_baseline_penalty_test': best_baseline_penalty_test,
	'best_images_complexity': best_images_complexity,
	'best_fmri_roi_1': best_fmri_roi_1,
	'best_fmri_roi_2': best_fmri_roi_2,
	'baseline_score_train_roi_1': baseline_roi_1,
	'baseline_score_train_roi_2': baseline_roi_2
	}

save_dir = os.path.join(args.project_dir, 'generative_univariate_rnc',
	'optimization_scores', 'cv-'+format(args.cv), roi_1+'-'+roi_2,
	'control_condition-'+args.control_type, model_save_dir, 'evolution-'+
	format(args.evolution, '02'), 'baseline_margin-'+str(args.baseline_margin))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'optimization_scores'
elif args.cv == 1:
	file_name = 'optimization_scores_cv_subject-' + \
		format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), data_dict)


# =============================================================================
# Save the image codes
# =============================================================================
save_dir = os.path.join(args.project_dir, 'generative_univariate_rnc',
	'image_codes', 'cv-'+format(args.cv), roi_1+'-'+roi_2, 'control_condition-'+
	args.control_type, model_save_dir, 'evolution-'+
	format(args.evolution, '02'), 'baseline_margin-'+str(args.baseline_margin))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'image_codes'
elif args.cv == 1:
	file_name = 'image_codes_cv_subject-' + format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), np.asarray(best_image_codes))
