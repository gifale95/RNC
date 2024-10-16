"""Generative univariate RNC iteratively generates images following two serial
objectives. Throughout the genetic optimization generations, the generated
images first drove or suppressed the in-silico univariate fMRI responses of V1
and V4 up to a threshold. Once this threshold is reached, the image complexity
– as measured by their PNG compression file size – started to monotonically
decrease, while keeping the in-silico univariate fMRI responses over the
threshold, thus promoting the generation of images containing only the visual
properties necessary to align or disentangle the two areas.

This code is available at:
https://github.com/gifale95/RNC/blob/main/03_generative_univariate_rnc/01_generative_univariate_rnc.py

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
roi_pair : int
	Integer indicating the chosen pairwise ROI combination on which to perform
	generative univariate RNC. Possible values are '0' (V1-V2), '1' (V1-V3),
	'2' (V1-hV4), '3' (V2-V3), '4' (V2-hV4), '5' (V3-hV4).
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
control_type : int
	If '0' generate images that drive both ROIs. If '1', generate images that
	drive the first ROI while suppressing the second ROI. If '2', generate
	images that suppress the first ROI while driving the second ROI. If '3',
	generate images that suppress both ROIs.
gan : str
	Name of the used GAN. The only available options is 'DeePSiM'.
gan_type : int
	Whether to use DeePSiM models trained on AlexNet 'fc6', 'fc7' or 'fc8'
	image_codes.
image_codes_initialization : str
	If 'random_normal', the image codes are randomly initialized from a standard
	normal distribution (mean=0, SD=1). If 'best_images', use the image codes of
	the 'best_images' found by the high throughput univariate control algorithm.
synt_img_clip_type : int
	If 1, clip the images in the range [0 1]. If 2 clip the images in the
	range [0 255]. If 3 clip the images in the range [-255 255] and normalize
	them in the range [0 1].
generations : int
	Number of gemetic optimization generations.
n_image_codes : int
	Number of image code, indicading how many images are generated and evaluated
	at each generation.
baseline_margin : float
	Margin to be added to the baseline univariate response score to define the
	neural control threshold.
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
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED

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
from utils import load_generator
from utils import generate_insilico_fmri
from utils import score_select
from utils import optimize_image_codes

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi_pair', type=int, default=2)
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--control_type', type=int, default=0)
parser.add_argument('--gan', type=str, default='DeePSiM')
parser.add_argument('--gan_type', type=str, default='fc7')
parser.add_argument('--synt_img_clip_type', type=int, default=2)
parser.add_argument('--generations', type=int, default=500)
parser.add_argument('--n_image_codes', type=int, default=1000)
parser.add_argument('--baseline_margin', type=float, default=0.6)
parser.add_argument('--img_complexity_measure', type=str, default='png')
parser.add_argument('--frac_kept_image_codes', type=float, default=.25)
parser.add_argument('--heritability', type=float, default=.25)
parser.add_argument('--mutation_prob', type=float, default=.25)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
args = parser.parse_args()

print('>>> Generative univariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)
random_generator = np.random.RandomState(seed=seed)


# =============================================================================
# Pairwise ROI combinations & neural control types
# =============================================================================
# 0 --> V1 - V2
# 1 --> V1 - V3
# 2 --> V1 - V4
# 3 --> V2 - V3
# 4 --> V2 - V4
# 5 --> V3 - V4
roi_comb_names = [['V1', 'V2'], ['V1', 'V3'], ['V1', 'hV4'], ['V2', 'V3'],
	['V2', 'hV4'], ['V3', 'hV4']]
roi_1 = roi_comb_names[args.roi_pair][0]
roi_2 = roi_comb_names[args.roi_pair][1]

# 0: V1
# 1: V2
# 2: V3
# 3: hV4
r1 = [0, 0, 0, 1, 1, 2]
r2 = [1, 2, 3, 2, 3, 3]

# Neural control types
control_types = ['high_1_high_2', 'high_1_low_2', 'low_1_high_2', 'low_1_low_2']


# =============================================================================
# Load the univariate RNC baseline
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'baseline', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if args.cv == 0:
	file_name = 'baseline.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	baseline_scores = data_dict['baseline_images_score']

elif args.cv == 1:
	file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02') + '.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	baseline_scores = data_dict['baseline_images_score_test']

baseline_roi_1 = baseline_scores[r1[args.roi_pair]]
baseline_roi_2 = baseline_scores[r2[args.roi_pair]]


# =============================================================================
# Load the encoding models of all subjects
# =============================================================================
encoding_models_roi_1, metadata_roi_1 = load_encoding_models(args, roi_1)
encoding_models_roi_2, metadata_roi_2 = load_encoding_models(args, roi_2)


# =============================================================================
# Import the GAN (the image generator)
# =============================================================================
generator = load_generator(args)

# Get the image codes dimensionality
if args.gan == 'DeePSiM':
	if args.gan_type == 'fc6':
		image_code_size = generator.fc6.in_features
	elif args.gan_type == 'fc7':
		image_code_size = generator.fc7.in_features
	elif args.gan_type == 'fc8':
		image_code_size = generator.fc8.in_features


# =============================================================================
# Initialize the image codes
# =============================================================================
# Randomly initialize image codes from a normal distributions
image_codes_new = random_generator.normal(loc=0, scale=1,
	size=(args.n_image_codes, image_code_size))

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
best_neural_control_scores_train = np.zeros((args.generations))
best_neural_control_scores_test = np.zeros((args.generations))
# Penalty scores
best_baseline_penalty_train = np.zeros((args.generations))
best_baseline_penalty_test = np.zeros((args.generations))
# Image complexity scores
best_images_complexity = np.zeros((args.generations))
# Total scores
best_scores_train = np.zeros((args.generations))
best_scores_test = np.zeros((args.generations))
# Image codes
best_image_codes = np.zeros((args.generations, image_code_size))
# In silico fMRI responses
best_fmri_roi_1 = np.zeros((args.generations, len(args.all_subjects)))
best_fmri_roi_2 = np.zeros((args.generations, len(args.all_subjects)))


# =============================================================================
# Generate images from the image codes
# =============================================================================
# Generation loop
for g in tqdm(range(args.generations), leave=False):

	# Generate the images
	img_codes = torch.FloatTensor(copy(image_codes_new))
	images_new = generator.forward(img_codes).detach().numpy()
	del img_codes

	# Clip and scale the images synthesized by the generator
	if args.synt_img_clip_type == 1:
		# Version 1 (as in Ponce et al., 2019):
		# """To synthesize an image from an input image code, we forward
		# propagated the code through the generative network, clamped the
		# output image pixel values to the valid range between 0 and 1, and
		# visualized them as an 8-bit color image."""
		images_new = np.clip(images_new, a_min=0, a_max=1) * 255

	elif args.synt_img_clip_type == 2:
		# Version 2:
		# Clamp the output image pixel values to the range [0 255]
		images_new = np.clip(images_new, a_min=0, a_max=255)

	elif args.synt_img_clip_type == 3:
		# Version 3:
		# Clamp the output image pixel values to the range [-255 255],
		# normalize them in the range [0 1], and scale them to the range
		# [0 255]
		images_new = np.clip(images_new, a_min=-255, a_max=255)
		images_new = (images_new - np.min(images_new.flatten())) / \
			(np.max(images_new.flatten()) - np.min(images_new.flatten())) * 255

	# Convert the images to uint8
	images_new = images_new.astype(np.uint8)


# =============================================================================
# Generate in silico fMRI responses for the synthesized images
# =============================================================================
	fmri_roi_1_new = generate_insilico_fmri(args, encoding_models_roi_1,
		metadata_roi_1, copy(images_new))
	fmri_roi_2_new = generate_insilico_fmri(args, encoding_models_roi_2,
		metadata_roi_2, copy(images_new))


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
	# image codes of the best N images.
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
				control_types[args.control_type])
		elif args.cv == 1:
			save_dir = os.path.join(args.project_dir,
				'generative_univariate_rnc', 'controlling_images', 'cv-'+
				format(args.cv), roi_1+'-'+roi_2, 'control_condition-'+
				control_types[args.control_type], 'cv_subject-'+
				format(args.cv_subject, '02'))

		if os.path.isdir(save_dir) == False:
			os.makedirs(save_dir)

		for i in range(args.n_image_codes):
			img = Image.fromarray(np.swapaxes(np.swapaxes(
				images_kept[0], 0, 1), 1, 2))
			file_name = 'gan_img_' + control_types[args.control_type] + \
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
	'control_condition-'+control_types[args.control_type])

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
	control_types[args.control_type])

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'image_codes'
elif args.cv == 1:
	file_name = 'image_codes_cv_subject-' + format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), np.asarray(best_image_codes))

