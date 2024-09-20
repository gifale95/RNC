"""For each ROI, get the synthetic fMRI responses for a randomly selected batch
of 25 images (out of all images from the chosen image set), average the
responses across all voxels within the ROI (obtaining the ROI's univariate
responses), and then average the univariate responses across the 25 images. This
will result in one score indicating the mean synthetic univariate fMRI response
for that image batch.

Repeating this step 1 million times will create the univariate RNC null 
distribution, from which the 25 images from the batch with score closest to the
distribution's mean are selected. The synthetic fMRI univariate response score
averaged across these 25 images provides the ROI's univariate response baseline.

This code is available at:
https://github.com/gifale95/RNC/blob/main/02_univariate_rnc/01_baseline.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
cv : int
	If '1' the synthetic univariate fMRI responses of one subject are left out
	for cross-validation, if '0' the univariate responses of all subjects are
	used.
cv_subject : int
	If cv=0 the left-out subject during cross-validation, out of all 8 (NSD)
	subjects.
rois : list of str
	List of used ROIs.
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
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import resample
from copy import copy
import h5py
from ned.ned import NED

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_images', type=int, default=25)
parser.add_argument('--null_dist_samples', type=int, default=1000000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
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
# Initialize the Neural Encoding Dataset (NED) object
# =============================================================================
# https://github.com/gifale95/NED
ned_object = NED(args.ned_dir)


# =============================================================================
# Sum the ROI data across voxels
# =============================================================================
# Synthetic fMRI univariate responses array of shape:
# (Subjects × ROIs × Images)
if args.imageset == 'nsd':
	images = 73000
elif args.imageset == 'imagenet_val':
	images = 50000
elif args.imageset == 'things':
	images = 26107
uni_resp = np.zeros((len(args.all_subjects), len(args.rois), images),
	dtype=np.float32)

for s, sub in enumerate(args.all_subjects):
	for r, roi in enumerate(args.rois):

		# Load the synthetic fMRI responses
		data_dir = os.path.join(args.project_dir, 'synthetic_fmri_responses',
			'imageset-'+args.imageset, 'synthetic_fmri_responses_sub-'+
			format(sub, '02')+'_roi-'+roi+'.h5')
		betas = h5py.File(data_dir).get('synthetic_fmri_responses')

		# Load the synthetic fMRI responses metadata
		metadata = ned_object.get_metadata(
			modality='fmri',
			train_dataset='nsd',
			model='fwrf',
			subject=sub,
			roi=roi
			)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold
		best_voxels = np.where(
			metadata['fmri']['ncsnr'] > args.ncsnr_threshold)[0]
		betas = betas[:,best_voxels]

		# Score the fMRI activity across voxels (there might be NaN values since
		# some subjects have missing data)
		uni_resp[s,r] = np.nanmean(betas, 1)

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are used
# to select the baseline images, and the same images are also used as baseline
# for the test subjects.
if args.cv == 0:
	uni_resp_mean = np.mean(uni_resp, 0)
elif args.cv == 1:
	uni_resp_mean_train = np.delete(uni_resp, args.cv_subject-1, 0)
	uni_resp_mean_train = np.mean(uni_resp_mean_train, 0)
	uni_resp_mean_test = uni_resp[args.cv_subject-1]


# =============================================================================
# Create the null distribution
# =============================================================================
# If cross-validating, separate null distributions are created for the synthetic
# fMRI responses of the train and test subjects.

null_distrubition_images = np.zeros((args.null_dist_samples, args.n_images),
	dtype=np.int32)
if args.cv == 0:
	null_distribution = np.zeros((args.null_dist_samples, len(args.rois)))
elif args.cv == 1:
	null_distribution_train = np.zeros((args.null_dist_samples, len(args.rois)))
	null_distribution_test = np.zeros((args.null_dist_samples, len(args.rois)))
for i in tqdm(range(args.null_dist_samples), leave=False):
	sample = resample(np.arange(images), replace=False, n_samples=args.n_images)
	sample.sort()
	null_distrubition_images[i] = copy(sample)
	for r in range(len(args.rois)):
		if args.cv == 0:
			null_distribution[i,r] = np.mean(uni_resp_mean[r,sample])
		elif args.cv == 1:
			null_distribution_train[i,r] = np.mean(uni_resp_mean_train[r,sample])
			null_distribution_test[i,r] = np.mean(uni_resp_mean_test[r,sample])


# =============================================================================
# Select the baseline images from the null distribution
# =============================================================================
# The baseline images are the image batch with syntehtic univariate fMRI score
# closest to the null distribution's mean.

if args.cv == 0:
	baseline_images_score = np.zeros(len(args.rois))
elif args.cv == 1:
	baseline_images_score_train = np.zeros(len(args.rois))
	baseline_images_score_test = np.zeros(len(args.rois))
baseline_images = np.zeros((len(args.rois), args.n_images), dtype=np.int32)

for r in range(len(args.rois)):
	if args.cv == 0:
		null_distribution_mean = np.mean(null_distribution[:,r])
		idx_best = np.argsort(abs(
			null_distribution[:,r] - null_distribution_mean))[0]
		baseline_images[r] = null_distrubition_images[idx_best]
		baseline_images_score[r] = null_distribution[idx_best,r]
	elif args.cv == 1:
		# The baseline images are chosen from the train subjects null
		# distribution, and then also evaluated on the test subject (to see
		# whether they generalize).
		null_distribution_mean = np.mean(null_distribution_train[:,r])
		idx_best = np.argsort(abs(
			null_distribution_train[:,r] - null_distribution_mean))[0]
		baseline_images[r] = null_distrubition_images[idx_best]
		baseline_images_score_train[r] = null_distribution_train[idx_best,r]
		baseline_images_score_test[r] = null_distribution_test[idx_best,r]


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

save_dir = os.path.join(args.project_dir, 'univariate_rnc', 'baseline', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'baseline'
elif args.cv == 1:
	file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), results)

