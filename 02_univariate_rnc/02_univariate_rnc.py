"""For each pairwise area combination (V1 vs. V2, V1 vs. V3, V1 vs. V4,
V2 vs. V3, V2 vs. V4, V3 vs. V4) univariate RNC searches, across stimuli from
the chosen imageset, for images that align or disentangle the in silico
univariate fMRI responses for the two ROIs being compared, thus highlighting
shared and unique representational content, respectively.

The in silico fMRI responses for all images from the chosen image set are
averaged across voxels, thus obtaining univariate responses. The in silico
univariate fMRI responses of the two ROIs are then either summed (alignment) or
subtracted (disentanglement), and the resulting sum/difference scores ranked.

Finally, the 25 controlling images leading to highest and lowest scores, while
at the same time resulting in in silico univariate fMRI responses higher (or
lower, depending on the control condition) than the ROIs' univariate response
baselines by a margin of at least 0.05, are kept.

This results in four sets of 25 controlling images, each set corresponding to a
different neural control condition. The controlling images from the sum vector
lead to two control conditions in which both areas have aligned univariate
responses (i.e., images that either drove or suppressed the responses of both
areas), whereas the controlling images from the difference vector lead to two
control conditions in which both areas have disentangled univariate responses
(i.e. images that drove the responses of one area while suppressing the
responses of the other area, and vice versa).

The in silico fMRI responses come from the Neural Encoding Dataset (NED):
https://github.com/gifale95/NED

This code is available at:
https://github.com/gifale95/RNC/blob/main/02_univariate_rnc/02_univariate_rnc.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
cv : int
	If '1' univariate RNC leaves the data of one subject out for
	cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
	If 'cv==0' the left-out subject during cross-validation, out of all 8 (NSD)
	subjects.
rois : list of str
	List of used ROIs.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
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
import h5py
from ned.ned import NED

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
args = parser.parse_args()

print('>>> Univariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize the Neural Encoding Dataset (NED) object
# =============================================================================
# https://github.com/gifale95/NED
ned_object = NED(args.ned_dir)


# =============================================================================
# Sum the ROI data across voxels
# =============================================================================
# In silico fMRI univariate responses array of shape:
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

		# Load the in silico fMRI responses
		data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'imageset-'+args.imageset, 'insilico_fmri_responses_sub-'+
			format(sub, '02')+'_roi-'+roi+'.h5')
		betas = h5py.File(data_dir).get('insilico_fmri_responses')

		# Load the in silico fMRI responses metadata
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
# to select the controlling images, and the controlling images will then be
# validated on the fMRI responses for the test subjects.
if args.cv == 0:
	uni_resp_mean = np.mean(uni_resp, 0)
elif args.cv == 1:
	uni_resp_mean = np.delete(uni_resp, args.cv_subject-1, 0)
	uni_resp_mean = np.mean(uni_resp_mean, 0)


# =============================================================================
# Load the univariate RNC baseline scores
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


# =============================================================================
# Pairwise ROI comparisons
# =============================================================================
# 0: V1
# 1: V2
# 2: V3
# 3: hV4
r1 = [0, 0, 0, 1, 1, 2]
r2 = [1, 2, 3, 2, 3, 3]


# =============================================================================
# Rank the images based on their univariate responses
# =============================================================================
# Empty ranking matrices of shape (ROI comparisons × Image conditions)
roi_sum = np.zeros((len(r1), images), dtype=np.float32)
roi_diff = np.zeros((roi_sum.shape), dtype=np.float32)
low_1_low_2 = np.zeros((roi_sum.shape), dtype=np.float32)
high_1_high_2 = np.zeros((roi_sum.shape), dtype=np.float32)
high_1_low_2 = np.zeros((roi_sum.shape), dtype=np.float32)
low_1_high_2 = np.zeros((roi_sum.shape), dtype=np.float32)

# Univariate response score margin used to constrain the selection of the
# control images
margin = 0.04

for r in range(len(r1)):

	# Select the top N images that align the in silico univariate fMRI
	# responses of the two ROIs (i.e., that lead both ROIs having either high
	# or low univariate responses).
	# 1st ranking: images with high univariate responses for both ROIs
	resp_roi_1 = uni_resp_mean[r1[r]]
	resp_roi_2 = uni_resp_mean[r2[r]]
	roi_sum[r] = resp_roi_1 + resp_roi_2
	high_1_high_2[r] = np.argsort(roi_sum[r])[::-1]
	# Ignore images conditions with univariate responses below the baseline
	# scores (plus a margin)
	idx_bad_roi_1 = np.where(resp_roi_1[high_1_high_2[r].astype(np.int32)] < \
		baseline_scores[r1[r]]+margin)[0]
	idx_bad_roi_2 = np.where(resp_roi_2[high_1_high_2[r].astype(np.int32)] < \
		baseline_scores[r2[r]]+margin)[0]
	high_1_high_2[r,idx_bad_roi_1] = np.nan
	high_1_high_2[r,idx_bad_roi_2] = np.nan
	# 2nd ranking: images with low univariate responses for both ROIs
	low_1_low_2[r] = np.argsort(roi_sum[r])
	# Ignore images conditions with univariate responses above the baseline
	# scores (plus a margin)
	idx_bad_roi_1 = np.where(resp_roi_1[low_1_low_2[r].astype(np.int32)] > \
		baseline_scores[r1[r]]-margin)[0]
	idx_bad_roi_2 = np.where(resp_roi_2[low_1_low_2[r].astype(np.int32)] > \
		baseline_scores[r2[r]]-margin)[0]
	low_1_low_2[r,idx_bad_roi_1] = np.nan
	low_1_low_2[r,idx_bad_roi_2] = np.nan

	# Select the top N images that differentiate the in silico univariate fMRI
	# responses of the two ROIs (i.e., that lead one ROI having high
	# responses and the other ROI low responses, or vice versa).
	# 3rd ranking: images with high univariate responses for ROI 1 and low
	# univariate responses for ROI 2
	roi_diff[r] = resp_roi_1 - resp_roi_2
	high_1_low_2[r] = np.argsort(roi_diff[r])[::-1]
	# Ignore images conditions with univariate responses below (ROI 1) or above
	# (ROI 2) the baseline scores (plus/minus a margin)
	idx_bad_roi_1 = np.where(resp_roi_1[high_1_low_2[r].astype(np.int32)] < \
		baseline_scores[r1[r]]+margin)[0]
	idx_bad_roi_2 = np.where(resp_roi_2[high_1_low_2[r].astype(np.int32)] > \
		baseline_scores[r2[r]]-margin)[0]
	high_1_low_2[r,idx_bad_roi_1] = np.nan
	high_1_low_2[r,idx_bad_roi_2] = np.nan
	# 4th ranking: images with low univariate responses for ROI 1 and high
	# univariate responses for ROI 2
	low_1_high_2[r] = np.argsort(roi_diff[r])
	# Ignore images conditions with univariate responses above (ROI 1) or below
	# (ROI 2) the baseline scores (minus/plus a margin)
	idx_bad_roi_1 = np.where(resp_roi_1[low_1_high_2[r].astype(np.int32)] > \
		baseline_scores[r1[r]]-margin)[0]
	idx_bad_roi_2 = np.where(resp_roi_2[low_1_high_2[r].astype(np.int32)] < \
		baseline_scores[r2[r]]+margin)[0]
	low_1_high_2[r,idx_bad_roi_1] = np.nan
	low_1_high_2[r,idx_bad_roi_2] = np.nan


# =============================================================================
# Save the results
# =============================================================================
data_dict = { 
	'uni_resp': uni_resp,
	'high_1_high_2': high_1_high_2,
	'low_1_low_2': low_1_low_2,
	'high_1_low_2': high_1_low_2,
	'low_1_high_2': low_1_high_2
	}

save_dir = os.path.join(args.project_dir, 'univariate_rnc', 'image_ranking',
	'cv-'+format(args.cv), 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'image_ranking'
elif args.cv == 1:
	file_name = 'image_ranking_cv_subject-' + format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), data_dict)

