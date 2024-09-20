def load_encoding_models(args, roi):
	"""Load the encoding models, and metadata.

	Parameters
	----------
	args : Namespace
		Input arguments.
	roi : str
		Used ROI.

	Returns
	-------
	encoding_models : list
		List containing the trained fwrf encoding models.
	metadata : dict
		Synthetic neural responses metadata.

	"""

	from ned.ned import NED
	from copy import deepcopy

	### Initialize NED ###
	ned_object = NED(args.ned_dir)

	### Load the trained encoding model weights and metadata ###
	encoding_models = []
	metadata = []
	for sub in args.all_subjects:
		# Encoding model weights
		encoding_models.append(deepcopy(ned_object.get_encoding_model(
			modality='fmri',
			train_dataset='nsd',
			model='fwrf',
			subject=sub,
			roi=roi,
			device='auto'
			)))
		# Metadata
		metadata.append(deepcopy(ned_object.get_metadata(
			modality='fmri',
			train_dataset='nsd',
			model='fwrf',
			subject=sub,
			roi=roi
			)))

	### Output ###
	return encoding_models, metadata


def load_generator(args):
	"""Load the image generator.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	generator : PyTorch model
		Image generator model.

	"""

	import os
	import torch
	import torch_nets # https://github.com/willwx/XDream/tree/master/xdream/net_utils/torch_nets

	### Instantiate the generator model ###
	if args.gan == 'DeePSiM':
		# (Dosovitskiy & Bronx, 2016) https://doi.org/10.48550/arXiv.1602.02644
		# (Xiao & Kreiman, 2020) https://doi.org/10.1371/journal.pcbi.1007973
		# (Ponce et al., 2019) https://doi.org/10.1016/j.cell.2019.04.005
		# ------------------------------
		# When using the 'fc6' version of DeePSiM I sometimes get a patch with
		# stereotypical shape in the center right of the synthesized images.
		# This is a known problem, that has been pointed out in
		# (Ponce et al., 2019):
		# """Some images synthesized by the network contained a patch with a
		# stereotypical shape that occurred in the center right of the image
		# (e.g., second and third image in Figure 4B, and "late synthetic" in
		# Figure 5C and 5D). This was identified as an artifact of the network
		# commonly known as "mode collapse" and it appeared in the same position
		# in a variety of contexts, including a subset of simulated evolutions.
		# This artifact was easily identifiable and it did not affect our
		# interpretations. In the future, more modern GNN training methods
		# should avoid this problem (personal correspondence with Alexey
		# Dosovitskiy)."""
		generator = torch_nets.load_net('deepsim-'+args.gan_type)

	### Load the generator model weights ###
	if args.gan == 'DeePSiM':
		# The generator trained weights can be downloaded from:
		# https://drive.google.com/drive/folders/1sV54kv5VXvtx4om1c9kBPbdlNuurkGFi
		weight_dir = os.path.join(args.project_dir, 'generative_univariate_rnc',
			'generator_weights', 'deepsim', args.gan_type+'.pt')
	generator_weights = torch.load(weight_dir, map_location='cpu')

	### Load the trained weights into the generator ###
	generator.load_state_dict(generator_weights)

	#### Set "requires_grad" to False, and turn on evaluation mode
	for param in generator.parameters():
		param.requires_grad = False
	generator.eval()

	### Output ###
	return generator


def synthesize_fmri(args, encoding_models, metadata, images):
	"""Use the encoding models to synthesize the fMRI responses for the
	generated images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	encoding_models : list
		List containing the trained fwrf encoding models.
	metadata : dict
		Synthetic neural responses metadata.
	images : int
		Synthesized images. Must be a 4-D numpy array of shape
		(Batch size x 3 RGB Channels x Width x Height) consisting of integer
		values in the range [0, 255]. Furthermore, the images must be of square
		size (i.e., equal width and height).

	Returns
	-------
	fmri : float
		fMRI responses for the generated images.

	"""

	import numpy as np
	from ned.ned import NED
	from copy import copy

	### Initialize NED ###
	ned_object = NED(args.ned_dir)

	### Synthesize the fMRI image responses ###
	fmri = []
	for s in range(len(args.all_subjects)):
		fmri_sub = ned_object.encode(
			encoding_models[s],
			images,
			return_metadata=False,
			device='auto'
			)
		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold
		best_voxels = np.where(
			metadata[s]['fmri']['ncsnr'] > args.ncsnr_threshold)[0]
		fmri_sub = fmri_sub[:,best_voxels]
		# Get the univariate responses by averaging the fMRI responses across
		# voxels
		fmri_sub = np.nanmean(fmri_sub, 1)
		fmri.append(copy(fmri_sub))
	fmri = np.asarray(fmri)

	### Output ###
	return fmri


def score_select(args, fmri_roi_1, fmri_roi_2, image_codes, images,
	baseline_roi_1, baseline_roi_2):
	"""Score and rank the generated images based on their neural control
	magnitude on the synthetic fMRI responses and their complexity.

	Parameters
	----------
	args : Namespace
		Input arguments.
	fmri_roi_1 : float
		ROI 1 fMRI responses for the generated images.
	fmri_roi_2 : float
		ROI 2 fMRI responses for the generated images.
	image_codes : float
		Image codes for the generated images.
	images : int
		Synthesized images. Must be a 4-D numpy array of shape
		(Batch size x 3 RGB Channels x Width x Height) consisting of integer
		values in the range [0, 255]. Furthermore, the images must be of square
		size (i.e., equal width and height).
	baseline_roi_1 : float
		ROI 1 baseline score.
	baseline_roi_2 : float
		ROI 2 baseline score.

	Returns
	-------
	scores_train : float
		Generated image scores for the train subjects.
	scores_test : float
		Generated image scores for the test subject.
	neural_control_scores_train : float
		Neural control scores for the train subjects.
	neural_control_scores_test : float
		Neural control scores for the test subject.
	baseline_penalty_train : float
		Baseline penalty scores for the train subjects.
	baseline_penalty_test : float
		Baseline penalty scores for the test subject.
	images_complexity : int
		Image complexity.
	image_codes : float
		Selected image codes for the best generated images.
	fmri_roi_1 : float
		ROI 1 fMRI responses.
	fmri_roi_2 : float
		ROI 2 fMRI responses.
	images : PyTorch tensor
		Generated images.

	"""

	import numpy as np
	import io
	from PIL import Image
	from copy import copy

	### Define the train/test partitions ###
	if args.cv == 0:
		fmri_roi_1_train = np.mean(fmri_roi_1, 0)
		fmri_roi_1_test = np.mean(fmri_roi_1, 0)
		fmri_roi_2_train = np.mean(fmri_roi_2, 0)
		fmri_roi_2_test = np.mean(fmri_roi_2, 0)
	elif args.cv == 1:
		fmri_roi_1_train = np.mean(np.delete(
			fmri_roi_1, args.cv_subject-1, 0), 0)
		fmri_roi_1_test = fmri_roi_1[args.cv_subject-1]
		fmri_roi_2_train = np.mean(np.delete(
			fmri_roi_2, args.cv_subject-1, 0), 0)
		fmri_roi_2_test = fmri_roi_2[args.cv_subject-1]

	### Compute the neural control scores ###
	# [0] --> High ROI 1 - High ROI 2
	# [1] --> High ROI 1 - Low ROI 2
	# [2] --> Low ROI 1 - High ROI 2
	# [3] --> Low ROI 1 - Low ROI 2
	if args.control_type == 0:
		neural_control_scores_train = fmri_roi_1_train + fmri_roi_2_train
		neural_control_scores_test = fmri_roi_1_test + fmri_roi_2_test
	elif args.control_type == 1:
		neural_control_scores_train = fmri_roi_1_train - fmri_roi_2_train
		neural_control_scores_test = fmri_roi_1_test - fmri_roi_2_test
	elif args.control_type == 2:
		neural_control_scores_train = fmri_roi_2_train - fmri_roi_1_train
		neural_control_scores_test = fmri_roi_2_test - fmri_roi_1_test
	elif args.control_type == 3:
		neural_control_scores_train = fmri_roi_1_train + fmri_roi_2_train
		neural_control_scores_test = fmri_roi_1_test + fmri_roi_2_test

	### Compute a penalty based on the fMRI baseline ###
	baseline_penalty_train = np.zeros((args.n_image_codes), dtype=int)
	baseline_penalty_test = np.zeros((args.n_image_codes), dtype=int)
	if args.control_type == 0:
		idx_bad_train_roi_1 = fmri_roi_1_train < (baseline_roi_1 + args.baseline_margin)
		idx_bad_train_roi_2 = fmri_roi_2_train < (baseline_roi_2 + args.baseline_margin)
		idx_bad_test_roi_1 = fmri_roi_1_test < (baseline_roi_1 + args.baseline_margin)
		idx_bad_test_roi_2 = fmri_roi_2_test < (baseline_roi_2 + args.baseline_margin)
	elif args.control_type == 1:
		idx_bad_train_roi_1 = fmri_roi_1_train < (baseline_roi_1 + args.baseline_margin)
		idx_bad_train_roi_2 = fmri_roi_2_train > (baseline_roi_2 - args.baseline_margin)
		idx_bad_test_roi_1 = fmri_roi_1_test < (baseline_roi_1 + args.baseline_margin)
		idx_bad_test_roi_2 = fmri_roi_2_test > (baseline_roi_2 - args.baseline_margin)
	elif args.control_type == 2:
		idx_bad_train_roi_1 = fmri_roi_1_train > (baseline_roi_1 - args.baseline_margin)
		idx_bad_train_roi_2 = fmri_roi_2_train < (baseline_roi_2 + args.baseline_margin)
		idx_bad_test_roi_1 = fmri_roi_1_test > (baseline_roi_1 - args.baseline_margin)
		idx_bad_test_roi_2 = fmri_roi_2_test < (baseline_roi_2 + args.baseline_margin)
	elif args.control_type == 3:
		idx_bad_train_roi_1 = fmri_roi_1_train > (baseline_roi_1 - args.baseline_margin)
		idx_bad_train_roi_2 = fmri_roi_2_train > (baseline_roi_2 - args.baseline_margin)
		idx_bad_test_roi_1 = fmri_roi_1_test > (baseline_roi_1 - args.baseline_margin)
		idx_bad_test_roi_2 = fmri_roi_2_test > (baseline_roi_2 - args.baseline_margin)
	idx_bad_train = np.where(idx_bad_train_roi_1 + idx_bad_train_roi_2)[0]
	baseline_penalty_train[idx_bad_train] = 1e+10
	idx_bad_test = np.where(idx_bad_test_roi_1 + idx_bad_test_roi_2)[0]
	baseline_penalty_test[idx_bad_test] = 1e+10

	### Compute the images complexity ###
	images_complexity = np.zeros((args.n_image_codes), dtype=np.int32)
	# Image complexity is estimated through compression: given that all
	# images have equal pixel sizes, the more an image can be compressed
	# (i.e., the lower the bytes of the compressed image), the less complex
	# it is (i.e., the less information it contains).
	for i, img in enumerate(images):
		img = np.swapaxes(np.swapaxes(copy(img), 0, 1), 1, 2)
		img_pil = Image.fromarray(img, 'RGB')
		output = io.BytesIO()
		if args.img_complexity_measure == 'png':
			img_pil.save(output, format='PNG')
		elif args.img_complexity_measure == 'jpg':
			img_pil.save(output, format='JPEG')
		images_complexity[i] = output.tell()

	### Aggregate neural control and baseline penalty scores ###
	if args.control_type == 3:
		neural_scores_train = neural_control_scores_train + \
			baseline_penalty_train
		neural_scores_test = neural_control_scores_test + baseline_penalty_test
	else:
		neural_scores_train = neural_control_scores_train - \
			baseline_penalty_train
		neural_scores_test = neural_control_scores_test - baseline_penalty_test
	# Only take neural control scores into account when they are below the
	# baseline + margin threshold: in this way, once the images well control
	# neural responses, their successive optimizations are only based on images
	# complexity.
	neural_scores_train[baseline_penalty_train==0] = 0
	neural_scores_test[baseline_penalty_test==0] = 0

	### Image complexity scores ###
	image_scores = - copy(images_complexity) # we want to reduce complexity
	# Once the neural control scores pass the threshold, only use the images
	# complexity to rank the images. However, these image scores are ignored
	# prior to passing the threshold.
	image_scores[baseline_penalty_train!=0] = 0

	### Aggregate neural control and complexity scores ###
	# The loss will only care about the neural control scores until these pass
	# the threshold (baseline + margin). After this threshold is reached, the
	# loss will only care about images complexity. This is because we
	# necessarily want images which well control neural responses but, once
	# we have such images, we want them to be as simple (to isolate controlling
	# visual features) as possible.
	if args.control_type == 3:
		scores_train = neural_scores_train - image_scores
		scores_test = neural_scores_test - image_scores
	else:
		scores_train = neural_scores_train + image_scores
		scores_test = neural_scores_test + image_scores

	### Rank the scores from best to worst ###
	if args.control_type == 3:
		rank = np.argsort(scores_train)
	else:
		rank = np.argsort(scores_train)[::-1]

	### Output ###
	return scores_train[rank], scores_test[rank], \
		neural_control_scores_train[rank], neural_control_scores_test[rank], \
		baseline_penalty_train[rank], baseline_penalty_test[rank], \
		images_complexity[rank], image_codes[rank], \
		fmri_roi_1[:,rank], fmri_roi_2[:,rank], images[rank]


def optimize_image_codes(args, image_codes, scores, random_generator):
	"""Genetic algorithm optimization from (Ponce et al., 2019):

	"The algorithm began with an initial population of 40 image codes
	('individuals'), each consisting of a 4096-dimensional vector ('genes') and
	associated with a synthesized image. Images were presented to the subject,
	and the corresponding spiking response was used to calculate the 'fitness'
	of the image codes by transforming the firing rate into a Z-score within the
	generation, scaling it by a selectiveness factor of 0.5, and passing it
	through a softmax function to become a probability. The 10 highest-fitness
	individuals (25%) were passed on to the next generation without
	recombination or mutation. Another 30 children image codes (75%) were
	produced from recombinations between two parent image codes from the current
	generation, with the probability for each image code to be a parent being
	its fitness. The two parents contributed unevenly (75%:25%) to any one
	child. Individual children genes had a 0.25 probability of being mutated,
	with mutations drawn from a 0-centered Gaussian with standard deviation
	0.75."

	Parameters
	----------
	args : Namespace
		Input arguments.
	image_codes : float
		Selected image codes for the best synthesized images.
	scores : float
		Generated images scores.
	random_generator : object
		Random numbers generator.

	Returns
	-------
	image_codes_new : float
		Optimized image codes.

	"""

	import numpy as np
	from sklearn.preprocessing import StandardScaler
	from scipy.special import softmax
	from copy import copy

	### Standardize the scores, scale them by 0.5, and apply softmax ###
	# https://towardsdatascience.com/transformer-networks-a-mathematical-explanation-why-scaling-the-dot-products-leads-to-more-stable-414f87391500
	scaler = StandardScaler()
	scores = scaler.fit_transform(
		np.reshape(scores, (len(scores), -1))).squeeze() * .5
	if args.control_type == 3:
		scores = - scores
	prob_scores = softmax(scores)

	### Keep X% of the best image codes untouched ###
	kept_img_codes = int(len(image_codes) * args.frac_kept_image_codes)

	### Recombine and mutate the remaining image codes ###
	image_codes_new = []
	for i in range(len(image_codes) - kept_img_codes):
		# Select the two parent image codes based on their probability scores
		parents = random_generator.choice(len(image_codes), size=2,
			replace=False, p=prob_scores)
		# Create the child image code
		random_idx = random_generator.choice(image_codes.shape[1],
			size=int(image_codes.shape[1]*args.heritability),
			replace=False)
		child_image_code = copy(image_codes[parents[0]])
		child_image_code[random_idx] = copy(image_codes[parents[1],random_idx])
		# Mutate the child image code
		mutations_idx = np.where(random_generator.choice(np.asarray((0,1)),
			size=len(child_image_code),
			p=np.asarray((1-args.mutation_prob,args.mutation_prob))))[0]
		child_image_code[mutations_idx] += random_generator.normal(loc=0,
			scale=.75, size=len(mutations_idx))
		image_codes_new.append(copy(child_image_code))
	image_codes_new = np.asarray(image_codes_new)

	### Output ###
	return image_codes_new
