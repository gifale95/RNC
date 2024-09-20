"""Compare univariate responses (i.e., voxel-average response) of V1 and V4
voxels tuned to the upper and lower portion of the visual field. Perform this
comparison for aligning images that both include and not include uniform empty
regions (i.e., the sky) their upper half.

To analyze the same controlling images as in the RNC paper, use the stats file:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/stats.py

The V1 ventral and dorsal delineations are provided in NSD. The V4 ventral and
dorsal delineations are avalable at:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/delineate_nsd_v4_dorsal_ventral/nsd_v4_delineations

This code is available at:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/02_retinotopy_effect.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
roi : str
	Whether to test 'V1' or 'hV4'.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED
nsd_dir : str
	Directory of the Natural Scenes Dataset.
	https://naturalscenesdataset.org/

"""

import argparse
import os
import numpy as np
from copy import copy
import random
from tqdm import tqdm
import nibabel as nib
import h5py
from ned.ned import NED
import pandas as pd
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='/home/ale/scratch/projects/relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='/home/ale/scratch/projects/neural_encoding_dataset/', type=str)
parser.add_argument('--nsd_dir', default='/media/ale/Elements/PhD/datasets/natural-scenes-dataset', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/relational_neural_control/', type=str)
#parser.add_argument('--ned_dir', default='/scratch/giffordale95/projects/neural_encoding_dataset/', type=str)
#parser.add_argument('--nsd_dir', default='/scratch/giffordale95/datasets/natural-scenes-dataset/', type=str)
args = parser.parse_args()

print('>>> Test retinotopy effect <<<')
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
	'imageset-nsd', 'V1-hV4', 'stats.npy') # https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/stats.py

stats = np.load(data_dir, allow_pickle=True).item()

best_generation_image_batches = \
	stats['best_generation_image_batches']['align'][-1]


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
mean_ventral_sky = np.zeros(len(args.all_subjects))
mean_ventral_non_sky = np.zeros(len(args.all_subjects))
mean_dorsal_sky = np.zeros(len(args.all_subjects))
mean_dorsal_non_sky = np.zeros(len(args.all_subjects))

for s, sub in tqdm(enumerate(args.all_subjects)):


# =============================================================================
# Load ventral/dorsal ROI masks
# =============================================================================
	# Mask indices of the prf-visualrois
	roi_family = 'prf-visualrois'
	roi_masks_dir = os.path.join(args.nsd_dir, 'nsddata', 'ppdata', 'subj'+
		format(sub, '02'), 'func1pt8mm', 'roi', roi_family+'.nii.gz')
	all_roi_mask_volume = nib.load(roi_masks_dir).get_fdata()
	volume_shape = all_roi_mask_volume.shape
	# Mapping dictionaries of the prf-visualrois
	roi_maps_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer',
		'subj'+format(sub, '02'), 'label', roi_family+'.mgz.ctab')
	roi_map = pd.read_csv(roi_maps_dir, delimiter=' ', header=None,
		index_col=0).to_dict()[1]

	if args.roi == 'V1':

		# Get the voxels of V1 ventral
		roi_mask_volume_ventral = np.zeros(volume_shape, dtype=bool)
		roi_value = [k for k, v in roi_map.items() if v in 'V1v'][0]
		roi_mask_volume_ventral[all_roi_mask_volume==roi_value] = True

		# Get the voxels of V1 dorsal
		roi_mask_volume_dorsal = np.zeros(volume_shape, dtype=bool)
		roi_value = [k for k, v in roi_map.items() if v in 'V1d'][0]
		roi_mask_volume_dorsal[all_roi_mask_volume==roi_value] = True

	elif args.roi == 'hV4':

		# Get the voxels of V4
		roi_mask_volume = np.zeros(volume_shape, dtype=bool)
		roi_value = [k for k, v in roi_map.items() if v in 'hV4'][0]
		roi_mask_volume[all_roi_mask_volume==roi_value] = True

		# Get the voxels of V4 ventral
		roi_mask_ventral = nib.load(os.path.join(args.nsd_dir, 'nsddata',
			'ppdata', 'subj'+format(sub, '02'), 'func1pt8mm', 'roi',
			'V4v.nii.gz')).get_fdata() == 1 # https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/delineate_nsd_v4_dorsal_ventral/nsd_v4_delineations
		roi_mask_volume_ventral = np.logical_and(roi_mask_volume,
			roi_mask_ventral)

		# Get the voxels of V4 dorsal
		roi_mask_dorsal = nib.load(os.path.join(args.nsd_dir, 'nsddata',
			'ppdata', 'subj'+format(sub, '02'), 'func1pt8mm', 'roi',
			'V4d.nii.gz')).get_fdata() == 1 # # https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/delineate_nsd_v4_dorsal_ventral/nsd_v4_delineations
		roi_mask_volume_dorsal = np.logical_and(roi_mask_volume,
			roi_mask_dorsal)


# =============================================================================
# Load pre-generated NED NSD responses
# =============================================================================
	ned_object = NED(args.ned_dir)

	# Load the synthetic fMRI responses
	data_dir = os.path.join(args.project_dir, 'synthetic_fmri_responses',
		'imageset-nsd', 'synthetic_fmri_responses_sub-'+format(sub, '02')+
		'_roi-'+args.roi+'.h5')
	fmri = h5py.File(data_dir).get('synthetic_fmri_responses')

	# Load the synthetic fMRI responses metadata
	metadata = ned_object.get_metadata(
		modality='fmri',
		train_dataset='nsd',
		model='fwrf',
		subject=sub,
		roi=args.roi
		)


# =============================================================================
# Select only the 50 alignment images
# =============================================================================
	fmri = fmri[best_generation_image_batches]


# =============================================================================
# Map fMRI to subject-native volume space, and index ventral/dorsal voxels
# =============================================================================
	# Create the fMRI betas volume
	fmri_vol = np.zeros((len(fmri), volume_shape[0], volume_shape[1],
		volume_shape[2]))
	for i in range(len(fmri)):
		vol = np.zeros((metadata['fmri']['roi_mask_volume'].shape))
		vol[metadata['fmri']['roi_mask_volume']] = fmri[i]
		fmri_vol[i] = copy(vol)
		del vol

	fmri_ventral = fmri_vol[:,roi_mask_volume_ventral]
	fmri_dorsal = fmri_vol[:,roi_mask_volume_dorsal]

	fmri_ventral_sky = fmri_ventral[sky_img_idx]
	fmri_ventral_non_sky = fmri_ventral[non_sky_img_idx]
	fmri_dorsal_sky = fmri_dorsal[sky_img_idx]
	fmri_dorsal_non_sky = fmri_dorsal[non_sky_img_idx]

	# Create the ncsnr volume
	ncsnr_vol = np.zeros((volume_shape[0], volume_shape[1], volume_shape[2]))
	ncsnr_vol[metadata['fmri']['roi_mask_volume']] = metadata['fmri']['ncsnr']

	ncsnr_ventral = ncsnr_vol[roi_mask_volume_ventral]
	ncsnr_dorsal = ncsnr_vol[roi_mask_volume_dorsal]


# =============================================================================
# Get the average fMRI response
# =============================================================================
	# Select voxels above the ncsnr threshold
	ventral_voxel_idx = ncsnr_ventral > args.ncsnr_threshold
	dorsal_voxel_idx = ncsnr_dorsal > args.ncsnr_threshold

	mean_ventral_sky[s] = np.mean(fmri_ventral_sky[:,ventral_voxel_idx])
	mean_ventral_non_sky[s] = np.mean(fmri_ventral_non_sky[:,ventral_voxel_idx])
	mean_dorsal_sky[s] = np.mean(fmri_dorsal_sky[:,dorsal_voxel_idx])
	mean_dorsal_non_sky[s] = np.mean(fmri_dorsal_non_sky[:,dorsal_voxel_idx])


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
# CI arrays of shape: (CI percentiles)
ci_ventral_sky = np.zeros((2))
ci_ventral_non_sky = np.zeros((2))
ci_dorsal_sky = np.zeros((2))
ci_dorsal_non_sky = np.zeros((2))

# Empty CI distribution arrays
ventral_sky_dist = np.zeros((args.n_iter))
ventral_non_sky_dist = np.zeros((args.n_iter))
dorsal_sky_dist = np.zeros((args.n_iter))
dorsal_non_sky_dist = np.zeros((args.n_iter))

# Compute the CI distributions
for i in tqdm(range(args.n_iter), leave=False):
	idx_resample = resample(np.arange(len(args.all_subjects)))
	ventral_sky_dist[i] = np.mean(mean_ventral_sky[idx_resample])
	ventral_non_sky_dist[i] = np.mean(mean_ventral_non_sky[idx_resample])
	dorsal_sky_dist[i] = np.mean(mean_dorsal_sky[idx_resample])
	dorsal_non_sky_dist[i] = np.mean(mean_dorsal_non_sky[idx_resample])

# Get the 5th and 95th CI distributions percentiles
ci_ventral_sky[0] = np.percentile(ventral_sky_dist, 2.5)
ci_ventral_sky[1] = np.percentile(ventral_sky_dist, 97.5)
ci_ventral_non_sky[0] = np.percentile(ventral_non_sky_dist, 2.5)
ci_ventral_non_sky[1] = np.percentile(ventral_non_sky_dist, 97.5)
ci_dorsal_sky[0] = np.percentile(dorsal_sky_dist, 2.5)
ci_dorsal_sky[1] = np.percentile(dorsal_sky_dist, 97.5)
ci_dorsal_non_sky[0] = np.percentile(dorsal_non_sky_dist, 2.5)
ci_dorsal_non_sky[1] = np.percentile(dorsal_non_sky_dist, 97.5)


# =============================================================================
# Calculate the significance
# =============================================================================
# The ventral/dorsal parts of V1 are tuned to the upper/lower portions of the
# visual field, whereas the ventral/dorsal parts of V4 are tuned to the
# lower/upper portions of the visual field.

# 1. For images including skies on their upper half, the mean response magnitude
# of voxels tuned to the upper portion of the visual field is lower than the
# mean response magnitude of voxels tuned to the lower portion of the visual
# field.
if args.roi == 'V1':
	p_value_1 = ttest_rel(mean_ventral_sky, mean_dorsal_sky,
		alternative='less')[1]
elif args.roi == 'hV4':
	p_value_1 = ttest_rel(mean_dorsal_sky, mean_ventral_sky,
		alternative='less')[1]

# 2. For images not including skies on their upper half, the mean response
# magnitude of voxels tuned to the upper portion of the visual field is higher
# than the mean response magnitude of voxels tuned to the lower portion of the
# visual field.
if args.roi == 'V1':
	p_value_3 = ttest_rel(mean_ventral_non_sky, mean_dorsal_non_sky,
		alternative='greater')[1]
elif args.roi == 'hV4':
	p_value_3 = ttest_rel(mean_dorsal_non_sky, mean_ventral_non_sky,
		alternative='greater')[1]

# Correct for multiple comparisons
p_values_all = np.append(p_value_1, p_value_3)
significance, p_values_corrected, _, _ = multipletests(p_values_all, 0.05,
	'fdr_bh')

# Store the significance and corrected p-values
significance_1 = significance[0]
significance_3 = significance[1]
p_value_corrected_1 = p_values_corrected[0]
p_value_corrected_3 = p_values_corrected[1]


# =============================================================================
# Save the results
# =============================================================================
results = {
	'mean_ventral_sky': mean_ventral_sky,
	'mean_ventral_non_sky': mean_ventral_non_sky,
	'mean_dorsal_sky': mean_dorsal_sky,
	'mean_dorsal_non_sky': mean_dorsal_non_sky,
	'ci_ventral_sky': ci_ventral_sky,
	'ci_ventral_non_sky': ci_ventral_non_sky,
	'ci_dorsal_sky': ci_dorsal_sky,
	'ci_dorsal_non_sky': ci_dorsal_non_sky,
	'p_value_1': p_value_1,
	'p_value_3': p_value_3,
	'significance_1': significance_1,
	'significance_3': significance_3,
	'p_value_corrected_1': p_value_corrected_1,
	'p_value_corrected_3': p_value_corrected_3
	}

save_dir = os.path.join(args.project_dir, 'retinotopy_effect', 'imageset-nsd')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'retinotopy_effect-' + args.roi

np.save(os.path.join(save_dir, file_name), results)
