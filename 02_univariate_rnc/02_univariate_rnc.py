"""For each pairwise area combination univariate RNC searches, across stimuli
from the chosen imageset, for images that align or disentangle the in silico
univariate fMRI responses for the two ROIs being compared, thus highlighting
shared and unique representational content, respectively.

The in silico fMRI responses for all images from the chosen image set are
averaged across voxels, thus obtaining univariate responses. The in silico
univariate fMRI responses of the two ROIs are then either summed (alignment) or
subtracted (disentanglement), and the resulting sum/difference scores ranked.

Finally, the 25 controlling images leading to highest and lowest scores, while
at the same time resulting in in silico univariate fMRI responses higher (or
lower, depending on the control condition) than the ROIs' univariate response
baselines by a margin of at least 0.04, are kept.

This results in four sets of 25 controlling images, each set corresponding to a
different neural control condition. The controlling images from the sum vector
lead to two control conditions in which both areas have aligned univariate
responses (i.e., images that either drove or suppressed the responses of both
areas), whereas the controlling images from the difference vector lead to two
control conditions in which both areas have disentangled univariate responses
(i.e. images that drove the responses of one area while suppressing the
responses of the other area, and vice versa).

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
roi_pair : str
	Used pairwise ROI combination.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi_pair', type=str, default='V1-V2')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Univariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]


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
uni_resp = np.zeros((len(all_subjects), len(rois), images),
	dtype=np.float32)

for s, sub in enumerate(all_subjects):
	for r, roi in enumerate(rois):

		# Load the in silico fMRI responses
		data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri', 'imageset-'+args.imageset,
			'insilico_fmri_responses_sub-'+format(sub, '02')+'_roi-'+roi+'.h5')
		betas = h5py.File(data_dir).get('insilico_fmri_responses')

		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri', 'ncsnr_sub-'+format(sub, '02')+'_roi-'+roi+'.npy')
		ncsnr = np.load(ncsnr_dir)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold.
		best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
		# For subject 4 of the Visual Illusion Reconstruction dataset, lower the
		# ncsnr theshold for ROI hV4 to 0.4, since there are no voxels above a
		# threshold of 0.5
		if args.encoding_models_train_dataset == 'VisualIllusionRecon':
			if roi == 'hV4' and sub == 4 and args.ncsnr_threshold > 0.4:
				best_voxels = np.where(ncsnr > 0.4)[0]
		betas = betas[:,best_voxels]

		# Average the fMRI activity across voxels (there might be NaN values
		# since some subjects have missing data)
		uni_resp[s,r] = np.nanmean(betas, 1)

# If cross-validating, remove the CV (test) subject, and average over the
# remaining (train) subjects. The fMRI responses for the train subjects are
# used to select the controlling images, and the controlling images will then
# be validated on the fMRI responses for the test subjects.
if args.cv == 0:
	uni_resp_mean = np.mean(uni_resp, 0)
elif args.cv == 1:
	uni_resp_mean = np.delete(uni_resp, args.cv_subject-1, 0)
	uni_resp_mean = np.mean(uni_resp_mean, 0)


# =============================================================================
# Load the univariate RNC baseline scores
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	'nsd_encoding_models', 'baseline', 'cv-'+format(args.cv), 'imageset-'+
	args.imageset)

baseline_scores = np.zeros(len(rois))

for r, roi in enumerate(rois):

	if args.cv == 0:
		file_name = 'baseline_roi-' + roi + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_scores[r] = data_dict['baseline_images_score']

	elif args.cv == 1:
		file_name = 'baseline_cv_subject-' + format(args.cv_subject, '02') + \
			'_roi-' + roi + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_scores[r] = data_dict['baseline_images_score_train']


# =============================================================================
# Rank the images based on their univariate responses
# =============================================================================
# Univariate response score margin used to constrain the selection of the
# control images
margin = 0.04

# Select the top N images that align the in silico univariate fMRI
# responses of the two ROIs (i.e., that lead both ROIs having either high
# or low univariate responses).
# 1st ranking: images with high univariate responses for both ROIs
resp_roi_1 = uni_resp_mean[0]
resp_roi_2 = uni_resp_mean[1]
roi_sum = resp_roi_1 + resp_roi_2
high_1_high_2 = np.argsort(roi_sum)[::-1].astype(np.float32)
# Ignore images conditions with univariate responses below the baseline
# scores (plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[high_1_high_2.astype(np.int32)] < \
	baseline_scores[0]+margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[high_1_high_2.astype(np.int32)] < \
	baseline_scores[1]+margin)[0]
high_1_high_2[idx_bad_roi_1] = np.nan
high_1_high_2[idx_bad_roi_2] = np.nan
# 2nd ranking: images with low univariate responses for both ROIs
low_1_low_2 = np.argsort(roi_sum).astype(np.float32)
# Ignore images conditions with univariate responses above the baseline
# scores (plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_low_2.astype(np.int32)] > \
	baseline_scores[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_low_2.astype(np.int32)] > \
	baseline_scores[1]-margin)[0]
low_1_low_2[idx_bad_roi_1] = np.nan
low_1_low_2[idx_bad_roi_2] = np.nan

# Select the top N images that disentangle the in silico univariate fMRI
# responses of the two ROIs (i.e., that lead one ROI having high
# responses and the other ROI low responses, or vice versa).
# 3rd ranking: images with high univariate responses for ROI 1 and low
# univariate responses for ROI 2
roi_diff = resp_roi_1 - resp_roi_2
high_1_low_2 = np.argsort(roi_diff)[::-1].astype(np.float32)
# Ignore images conditions with univariate responses below (ROI 1) or above
# (ROI 2) the baseline scores (plus/minus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[high_1_low_2.astype(np.int32)] < \
	baseline_scores[0]+margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[high_1_low_2.astype(np.int32)] > \
	baseline_scores[1]-margin)[0]
high_1_low_2[idx_bad_roi_1] = np.nan
high_1_low_2[idx_bad_roi_2] = np.nan
# 4th ranking: images with low univariate responses for ROI 1 and high
# univariate responses for ROI 2
low_1_high_2 = np.argsort(roi_diff).astype(np.float32)
# Ignore images conditions with univariate responses above (ROI 1) or below
# (ROI 2) the baseline scores (minus/plus a margin)
idx_bad_roi_1 = np.where(resp_roi_1[low_1_high_2.astype(np.int32)] > \
	baseline_scores[0]-margin)[0]
idx_bad_roi_2 = np.where(resp_roi_2[low_1_high_2.astype(np.int32)] < \
	baseline_scores[1]+margin)[0]
low_1_high_2[idx_bad_roi_1] = np.nan
low_1_high_2[idx_bad_roi_2] = np.nan


# =============================================================================
# Save the results
# =============================================================================
data_dict = {
	'roi_1': roi_1,
	'roi_2': roi_2,
	'uni_resp': uni_resp,
	'high_1_high_2': high_1_high_2,
	'low_1_low_2': low_1_low_2,
	'high_1_low_2': high_1_low_2,
	'low_1_high_2': low_1_high_2
	}

save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'image_ranking',
	'cv-'+format(args.cv), 'imageset-'+args.imageset, args.roi_pair)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'image_ranking.npy'
elif args.cv == 1:
	file_name = 'image_ranking_cv_subject-' + format(args.cv_subject, '02') + \
		'.npy'

np.save(os.path.join(save_dir, file_name), data_dict)
