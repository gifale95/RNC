"""For each ROI, get the in silico fMRI responses for a randomly selected batch
of 25 images (out of all images from the chosen image set), average the
responses across all voxels within the ROI (obtaining the ROI's univariate
responses), and then average the univariate responses across the 25 images.
This will result in one score indicating the mean in silico univariate fMRI
response for that image batch.

Repeating this step 1 million times will create the univariate RNC null 
distribution, from which the 25 images from the batch with score closest to the
distribution's mean are selected. The in silico fMRI univariate response score
averaged across these 25 images provides the ROI's univariate response
baseline.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
cv : int
	If '1' univariate RNC leaves the data of one subject out for
	cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
	If cv==1, the left-out subject during cross-validation, out of the 8 NSD
	subjects (if encoding_models_train_dataset=='nsd'), or the 7 Visual Illusion
	Reconstruction dataset subjects
	(if encoding_models_train_dataset=='VisualIllusionRecon').
roi : str
	Used ROI.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_images : int
	Image batch size used to compute the null distribution.
null_dist_samples : int
	Amount of null distribution samples.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_images', type=int, default=25)
parser.add_argument('--null_dist_samples', type=int, default=1000000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Univariate RNC baseline <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# Compute the univariate responses
# =============================================================================
# In silico fMRI univariate responses array of shape:
# (Subjects × ROIs × Images)
if args.imageset == 'nsd':
	images = 73000
elif args.imageset == 'imagenet_val':
	images = 50000
elif args.imageset == 'things':
	images = 26107
uni_resp = np.zeros((len(all_subjects), images), dtype=np.float32)

for s, sub in enumerate(all_subjects):

	# Load the in silico fMRI responses
	data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
		args.encoding_models_train_dataset+'_encoding_models', 'insilico_fmri',
		'imageset-'+args.imageset, 'insilico_fmri_responses_sub-'+
		format(sub, '02')+'_roi-'+args.roi+'.h5')
	betas = h5py.File(data_dir).get('insilico_fmri_responses')

	# Load the ncsnr
	ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
		args.encoding_models_train_dataset+'_encoding_models', 'insilico_fmri',
		'ncsnr_sub-'+format(sub, '02')+'_roi-'+args.roi+'.npy')
	ncsnr = np.load(ncsnr_dir)

	# Only retain voxels with noise ceiling signal-to-noise ratio scores
	# above the selected threshold.
	best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
	# For subject 4 of the Visual Illusion Reconstruction dataset, lower the
	# ncsnr theshold for ROI hV4 to 0.4, since there are no voxels above a
	# threshold of 0.5
	if args.encoding_models_train_dataset == 'VisualIllusionRecon':
		if args.roi == 'hV4' and sub == 4 and args.ncsnr_threshold > 0.4:
			best_voxels = np.where(ncsnr > 0.4)[0]
	betas = betas[:,best_voxels]
	# Average the fMRI activity across voxels (there might be NaN values since
	# some subjects have missing data)
	uni_resp[s] = np.nanmean(betas, 1)

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the baseline images, and the same images are also used as
# baseline for the test subjects.
if args.cv == 0:
	uni_resp_mean = np.mean(uni_resp, 0)
elif args.cv == 1:
	uni_resp_mean_train = np.delete(uni_resp, args.cv_subject-1, 0)
	uni_resp_mean_train = np.mean(uni_resp_mean_train, 0)
	uni_resp_mean_test = uni_resp[args.cv_subject-1]


# =============================================================================
# Create the null distribution
# =============================================================================
# If cross-validating, separate null distributions are created for the in
# silico fMRI responses of the train and test subjects.

null_distrubition_images = np.zeros((args.null_dist_samples, args.n_images),
	dtype=np.int32)
if args.cv == 0:
	null_distribution = np.zeros((args.null_dist_samples))
elif args.cv == 1:
	null_distribution_train = np.zeros((args.null_dist_samples))
	null_distribution_test = np.zeros((args.null_dist_samples))
for i in tqdm(range(args.null_dist_samples), leave=False):
	sample = resample(np.arange(images), replace=False,
		n_samples=args.n_images)
	sample.sort()
	null_distrubition_images[i] = copy(sample)
	if args.cv == 0:
		null_distribution[i] = np.mean(uni_resp_mean[sample])
	elif args.cv == 1:
		null_distribution_train[i] = np.mean(uni_resp_mean_train[sample])
		null_distribution_test[i] = np.mean(uni_resp_mean_test[sample])


# =============================================================================
# Select the baseline images from the null distribution
# =============================================================================
# The baseline images are the image batch with syntehtic univariate fMRI score
# closest to the null distribution's mean.

if args.cv == 0:
	null_distribution_mean = np.mean(null_distribution)
	idx_best = np.argsort(abs(null_distribution - null_distribution_mean))[0]
	baseline_images = null_distrubition_images[idx_best]
	baseline_images_score = null_distribution[idx_best]

elif args.cv == 1:
	# The baseline images are chosen from the train subjects null
	# distribution, and then also evaluated on the test subject (to see
	# whether they generalize).
	null_distribution_mean = np.mean(null_distribution_train)
	idx_best = np.argsort(abs(
		null_distribution_train - null_distribution_mean))[0]
	baseline_images = null_distrubition_images[idx_best]
	baseline_images_score_train = null_distribution_train[idx_best]
	baseline_images_score_test = null_distribution_test[idx_best]

# Convert the baseline images to integer
baseline_images = baseline_images.astype(np.int32)


# =============================================================================
# Save the results
# =============================================================================
if args.cv == 0:
	results = {
		'baseline_images': baseline_images,
		'baseline_images_score': baseline_images_score
		}
elif args.cv == 1:
	results = {
		'baseline_images': baseline_images,
		'baseline_images_score_test': baseline_images_score_test,
		'baseline_images_score_train': baseline_images_score_train,
		}

save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'baseline', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'baseline_roi-' + args.roi
elif args.cv == 1:
	file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02') + \
		'_roi-' + args.roi

np.save(os.path.join(save_dir, file_name), results)
