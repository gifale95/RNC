"""Create RSMs using the synthetic fMRI responses for the aligning and
disentangling RNC images.

For the aligning images, additionally compute the average RSM correlation
scores for images with and without skies on their upper half.

To analyze the same controlling images as in the RNC paper, use the stats file:
https://github.com/gifale95/RNC/05_multivariate_rnc_retinotopy/stats.py

This code is available at:
https://github.com/gifale95/RNC/05_multivariate_rnc_retinotopy/01_controlling_images_rsms.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
control_condition : str
	Whether to build RSMs for images that 'align' or 'disentangle' the
	multivariate fMRI responses for the two ROIs being compared.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import h5py
from ned.ned import NED
from scipy.stats import pearsonr
from copy import copy
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--control_condition', type=str, default='align')
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
8args = parser.parse_args()

print('>>> Create controlling images RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Load the multivariate RNC cv-0 aligning images numbers
# =============================================================================
data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats', 'cv-0',
	'imageset-nsd', 'V1-hV4', 'stats.npy') # https://github.com/gifale95/RNC/05_multivariate_rnc_retinotopy/stats.py

stats = np.load(data_dir, allow_pickle=True).item()

best_generation_image_batches = \
	stats['best_generation_image_batches'][args.control_condition][-1]


# =============================================================================
# Select the (NSD) images with skies on their upper half
# =============================================================================
# These sky and non-sky image indices are relative to the alignment controlling
# images from the RNC paper results. If by running the analyses you obtained
# different controlling images, you will need to modify these sky and non-sky
# image indices accordingly.

sky_img_idx = [1, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18, 19, 20, 21, 22,
	23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 43, 44, 45,
	46, 47, 49]
sky_img_num = best_generation_image_batches[sky_img_idx]

non_sky_img_idx = [0, 4, 10, 11, 13, 17, 35, 38, 41, 42, 48]
non_sky_img_num = best_generation_image_batches[non_sky_img_idx]


# =============================================================================
# Subjects loop
# =============================================================================
rsm_v1_all = []
rsm_v4_all = []
v1_corr_sky_sky = []
v1_corr_non_sky_non_sky = []
v1_corr_sky_non_sky = []
v4_corr_sky_sky = []
v4_corr_non_sky_non_sky = []
v4_corr_sky_non_sky = []

for s, sub in tqdm(enumerate(args.all_subjects)):


# =============================================================================
# Load pre-generated NED NSD responses
# =============================================================================
	ned_object = NED(args.ned_dir)

	# V1
	# Load the synthetic fMRI responses
	data_dir = os.path.join(args.project_dir, 'synthetic_fmri_responses',
		'imageset-nsd', 'synthetic_fmri_responses_sub-'+format(sub, '02')+
		'_roi-V1.h5')
	fmri_v1 = h5py.File(data_dir).get('synthetic_fmri_responses')
	# Load the synthetic fMRI responses metadata
	metadata_v1 = ned_object.get_metadata(
		modality='fmri',
		train_dataset='nsd',
		model='fwrf',
		subject=sub,
		roi='V1'
		)

	# V4
	# Load the synthetic fMRI responses
	data_dir = os.path.join(args.project_dir, 'synthetic_fmri_responses',
		'imageset-nsd', 'synthetic_fmri_responses_sub-'+format(sub, '02')+
		'_roi-hV4.h5')
	fmri_v4 = h5py.File(data_dir).get('synthetic_fmri_responses')
	# Load the synthetic fMRI responses metadata
	metadata_v4 = ned_object.get_metadata(
		modality='fmri',
		train_dataset='nsd',
		model='fwrf',
		subject=sub,
		roi='hV4'
		)


# =============================================================================
# Select only the 50 multivariate RNC results images
# =============================================================================
	fmri_v1 = fmri_v1[best_generation_image_batches]
	fmri_v4 = fmri_v4[best_generation_image_batches]


# =============================================================================
# Voxels selection
# =============================================================================
	idx_voxels_v1 = metadata_v1['fmri']['ncsnr'] > args.ncsnr_threshold
	idx_voxels_v4 = metadata_v4['fmri']['ncsnr'] > args.ncsnr_threshold

	fmri_v1 = fmri_v1[:,idx_voxels_v1]
	fmri_v4 = fmri_v4[:,idx_voxels_v4]


# =============================================================================
# Create the RSMs
# =============================================================================
	rms_v1 = np.ones((len(fmri_v1), len(fmri_v1)))
	rms_v4 = np.ones((len(fmri_v4), len(fmri_v4)))

	for c1 in range(len(fmri_v1)):
		for c2 in range(c1):
			rms_v1[c1,c2] = pearsonr(fmri_v1[c1], fmri_v1[c2])[0]
			rms_v1[c2,c1] = rms_v1[c1,c2]
			rms_v4[c1,c2] = pearsonr(fmri_v4[c1], fmri_v4[c2])[0]
			rms_v4[c2,c1] = rms_v4[c1,c2]

	rsm_v1_all.append(copy(rms_v1))
	rsm_v4_all.append(copy(rms_v4))


# =============================================================================
# Compute the average correlation for different image types (i.e., sky and
# non-sky images)
# =============================================================================
	if args.control_condition == 'align':

		# Sky vs. sky
		v1 = rms_v1[sky_img_idx]
		v1 = v1[:,sky_img_idx]
		v4 = rms_v4[sky_img_idx]
		v4 = v4[:,sky_img_idx]
		idx = np.tril_indices(len(v1), -1)
		v1_corr_sky_sky.append(np.mean(v1[idx]))
		v4_corr_sky_sky.append(np.mean(v4[idx]))

		# Non sky vs. non sky
		v1 = rms_v1[non_sky_img_idx]
		v1 = v1[:,non_sky_img_idx]
		v4 = rms_v4[non_sky_img_idx]
		v4 = v4[:,non_sky_img_idx]
		idx = np.tril_indices(len(v1), -1)
		v1_corr_non_sky_non_sky.append(np.mean(v1[idx]))
		v4_corr_non_sky_non_sky.append(np.mean(v4[idx]))

		# Sky vs. non sky
		v1 = rms_v1[sky_img_idx]
		v1 = v1[:,non_sky_img_idx]
		v4 = rms_v4[sky_img_idx]
		v4 = v4[:,non_sky_img_idx]
		v1_corr_sky_non_sky.append(np.mean(v1))
		v4_corr_sky_non_sky.append(np.mean(v4))

	del rms_v1, rms_v4

v1_corr_sky_sky = np.asarray(v1_corr_sky_sky)
v1_corr_non_sky_non_sky = np.asarray(v1_corr_non_sky_non_sky)
v1_corr_sky_non_sky = np.asarray(v1_corr_sky_non_sky)
v4_corr_sky_sky = np.asarray(v4_corr_sky_sky)
v4_corr_non_sky_non_sky = np.asarray(v4_corr_non_sky_non_sky)
v4_corr_sky_non_sky = np.asarray(v4_corr_sky_non_sky)


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
# CI arrays of shape: (CI percentiles)
ci_v1_corr_sky_sky = np.zeros((2))
ci_v1_corr_non_sky_non_sky = np.zeros((2))
ci_v1_corr_sky_non_sky = np.zeros((2))
ci_v4_corr_sky_sky = np.zeros((2))
ci_v4_corr_non_sky_non_sky = np.zeros((2))
ci_v4_corr_sky_non_sky = np.zeros((2))

if args.control_condition == 'align':

	# Empty CI distribution arrays
	v1_sky_sky_dist = np.zeros((args.n_iter))
	v1_non_sky_non_sky_dist = np.zeros((args.n_iter))
	v1_sky_non_sky_dist = np.zeros((args.n_iter))
	v4_sky_sky_dist = np.zeros((args.n_iter))
	v4_non_sky_non_sky_dist = np.zeros((args.n_iter))
	v4_sky_non_sky_dist = np.zeros((args.n_iter))

	# Compute the CI distributions
	for i in tqdm(range(args.n_iter), leave=False):
		idx_resample = resample(np.arange(len(args.all_subjects)))
		v1_sky_sky_dist[i] = np.mean(v1_corr_sky_sky[idx_resample])
		v1_non_sky_non_sky_dist[i] = np.mean(
			v1_corr_non_sky_non_sky[idx_resample])
		v1_sky_non_sky_dist[i] = np.mean(v1_corr_sky_non_sky[idx_resample])
		v4_sky_sky_dist[i] = np.mean(v4_corr_sky_sky[idx_resample])
		v4_non_sky_non_sky_dist[i] = np.mean(
			v4_corr_non_sky_non_sky[idx_resample])
		v4_sky_non_sky_dist[i] = np.mean(v4_corr_sky_non_sky[idx_resample])

	# Get the 5th and 95th CI distributions percentiles
	ci_v1_corr_sky_sky[0] = np.percentile(v1_sky_sky_dist, 2.5)
	ci_v1_corr_sky_sky[1] = np.percentile(v1_sky_sky_dist, 97.5)
	ci_v1_corr_non_sky_non_sky[0] = np.percentile(v1_non_sky_non_sky_dist, 2.5)
	ci_v1_corr_non_sky_non_sky[1] = np.percentile(v1_non_sky_non_sky_dist, 97.5)
	ci_v1_corr_sky_non_sky[0] = np.percentile(v1_sky_non_sky_dist, 2.5)
	ci_v1_corr_sky_non_sky[1] = np.percentile(v1_sky_non_sky_dist, 97.5)
	ci_v4_corr_sky_sky[0] = np.percentile(v4_sky_sky_dist, 2.5)
	ci_v4_corr_sky_sky[1] = np.percentile(v4_sky_sky_dist, 97.5)
	ci_v4_corr_non_sky_non_sky[0] = np.percentile(v4_non_sky_non_sky_dist, 2.5)
	ci_v4_corr_non_sky_non_sky[1] = np.percentile(v4_non_sky_non_sky_dist, 97.5)
	ci_v4_corr_sky_non_sky[0] = np.percentile(v4_sky_non_sky_dist, 2.5)
	ci_v4_corr_sky_non_sky[1] = np.percentile(v4_sky_non_sky_dist, 97.5)


# =============================================================================
# Calculate the significance
# =============================================================================
# (sky vs. sky) vs. (sky vs. non sky)
v1_p_value_1 = ttest_rel(v1_corr_sky_sky, v1_corr_sky_non_sky,
	alternative='greater')[1]
v4_p_value_1 = ttest_rel(v4_corr_sky_sky, v4_corr_sky_non_sky,
	alternative='greater')[1]

# (sky vs. sky) vs. (non sky vs. non sky)
v1_p_value_2 = ttest_rel(v1_corr_sky_sky, v1_corr_non_sky_non_sky,
	alternative='greater')[1]
v4_p_value_2 = ttest_rel(v4_corr_sky_sky, v4_corr_non_sky_non_sky,
	alternative='greater')[1]

# (sky vs. non sky) vs. (non sky vs. non sky)
v1_p_value_3 = ttest_rel(v1_corr_sky_non_sky, v1_corr_non_sky_non_sky,
	alternative='less')[1]
v4_p_value_3 = ttest_rel(v4_corr_sky_non_sky, v4_corr_non_sky_non_sky,
	alternative='less')[1]

# Correct for multiple comparisons
V1_p_values_all = np.asarray((v1_p_value_1, v1_p_value_2, v1_p_value_3))
V4_p_values_all = np.asarray((v4_p_value_1, v4_p_value_2, v4_p_value_3))
v1_significance, v1_p_values_corrected, _, _ = multipletests(V1_p_values_all,
	0.05, 'fdr_bh')
v4_significance, v4_p_values_corrected, _, _ = multipletests(V4_p_values_all,
	0.05, 'fdr_bh')

# Store the significance and corrected p-values
v1_significance_1 = v1_significance[0]
v1_significance_2 = v1_significance[1]
v1_significance_3 = v1_significance[2]
v1_p_value_corrected_1 = v1_p_values_corrected[0]
v1_p_value_corrected_2 = v1_p_values_corrected[1]
v1_p_value_corrected_3 = v1_p_values_corrected[2]
v4_significance_1 = v4_significance[0]
v4_significance_2 = v4_significance[1]
v4_significance_3 = v4_significance[2]
v4_p_value_corrected_1 = v4_p_values_corrected[0]
v4_p_value_corrected_2 = v4_p_values_corrected[1]
v4_p_value_corrected_3 = v4_p_values_corrected[2]


# =============================================================================
# Save the results
# =============================================================================
results = {
	'rsm_v1': rsm_v1_all,
	'rsm_v4': rsm_v4_all,
	'v1_corr_sky_sky': v1_corr_sky_sky,
	'v1_corr_non_sky_non_sky': v1_corr_non_sky_non_sky,
	'v1_corr_sky_non_sky': v1_corr_sky_non_sky,
	'v4_corr_sky_sky': v4_corr_sky_sky,
	'v4_corr_non_sky_non_sky': v4_corr_non_sky_non_sky,
	'v4_corr_sky_non_sky': v4_corr_sky_non_sky,

	'ci_v1_corr_sky_sky': ci_v1_corr_sky_sky,
	'ci_v1_corr_non_sky_non_sky': ci_v1_corr_non_sky_non_sky,
	'ci_v1_corr_sky_non_sky': ci_v1_corr_sky_non_sky,
	'ci_v4_corr_sky_sky': ci_v4_corr_sky_sky,
	'ci_v4_corr_non_sky_non_sky': ci_v4_corr_non_sky_non_sky,
	'ci_v4_corr_sky_non_sky': ci_v4_corr_sky_non_sky,

	'v1_p_value_1': v1_p_value_1,
	'v1_p_value_2': v1_p_value_2,
	'v1_p_value_3': v1_p_value_3,
	'v4_p_value_1': v4_p_value_1,
	'v4_p_value_2': v4_p_value_2,
	'v4_p_value_3': v4_p_value_3,

	'v1_p_value_corrected_1': v1_p_value_corrected_1,
	'v1_p_value_corrected_2': v1_p_value_corrected_2,
	'v1_p_value_corrected_3': v1_p_value_corrected_3,
	'v4_p_value_corrected_1': v4_p_value_corrected_1,
	'v4_p_value_corrected_2': v4_p_value_corrected_2,
	'v4_p_value_corrected_3': v4_p_value_corrected_3,

	'v1_significance_1': v1_significance_1,
	'v1_significance_2': v1_significance_2,
	'v1_significance_3': v1_significance_3,
	'v4_significance_1': v4_significance_1,
	'v4_significance_2': v4_significance_2,
	'v4_significance_3': v4_significance_3
	}

save_dir = os.path.join(args.project_dir, 'retinotopy_effect', 'imageset-nsd')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'rsms_control_condition-' + args.control_condition

np.save(os.path.join(save_dir, file_name), results)

