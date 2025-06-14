"""Compute the Brain Encoding Response Generator (BERG) fMRI encoding models
out-of-distribution (OOD) encoding accuracies, using NSD-synthetic.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
rois : list of str
	List of used ROIs.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.
berg_dir : str
	Directory of the Brain Encoding Response Generator.
	https://github.com/gifale95/BERG
nsd_dir : str
	Directory of the Natural Scenes Dataset.
	https://naturalscenesdataset.org/

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from berg import BERG
from scipy.stats import pearsonr
import nibabel as nib
import h5py
from PIL import Image
from scipy.stats import zscore
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4', 'FFA', 'PPA', 'RSC', 'EBA'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-reponse-generator/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
args = parser.parse_args()

print('>>> OOD encoding accuracy and SNR analysis - NSD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize the BERG object
# =============================================================================
# https://github.com/gifale95/BERG
berg_object = BERG(args.berg_dir)


# =============================================================================
# Loop over subjects and ROIs
# =============================================================================
explained_variance = {}

for s in tqdm(args.all_subjects, leave=False):
	for r in args.rois:


# =============================================================================
# Get ROI voxel indices in volume space
# =============================================================================
		# Load the in silico fMRI responses metadata
		# ROI 'FFA' is divided into two parts (anterior and posterior):
		# https://cvnlab.slite.page/p/X_7BBMgghj/ROIs
		if r in ['FFA']:
			# Metadata model for ROI split 1
			metadata_1 = berg_object.get_model_metadata(
				model_id='fmri-nsd-fwrf',
				subject=s,
				roi=r+'-1'
				)
			# Metadata model for ROI split 2
			metadata_2 = berg_object.get_model_metadata(
				model_id='fmri-nsd-fwrf',
				subject=s,
				roi=r+'-2'
				)
		else:
			metadata = berg_object.get_model_metadata(
				model_id='fmri-nsd-fwrf',
				subject=s,
				roi=r
				)

		# Get the ROI voxel indices in volume space
		if r in ['FFA']:
			roi_mask_1 = metadata_1['fmri']['roi_mask_volume']
			roi_mask_2 = metadata_2['fmri']['roi_mask_volume']
			roi_mask = roi_mask_1 + roi_mask_2
		else:
			roi_mask = metadata['fmri']['roi_mask_volume']


# =============================================================================
# Load NSD-synthetic's in vivo fMRI responses
# =============================================================================
		# Get order and ID of the presented images
		# Load the experimental design info
		expdesign = loadmat(os.path.join(args.nsd_dir, 'nsddata',
			'experiments', 'nsdsynthetic', 'nsdsynthetic_expdesign.mat'))
		# Subtract 1 since the indices start with 1 (and not 0)
		masterordering = np.squeeze(expdesign['masterordering'] - 1)

		# Prepare the fMRI betas
		# Load the fMRI betas
		betas_dir = os.path.join(args.nsd_dir, 'nsddata_betas', 'ppdata',
			'subj'+format(s, '02'), 'func1pt8mm',
			'nsdsyntheticbetas_fithrf_GLMdenoise_RR',
			'betas_nsdsynthetic.nii.gz')
		betas = nib.load(betas_dir).get_fdata()
		# Mask the ROI voxels
		betas = np.transpose(betas[roi_mask])
		# Convert back to decimal format and divide by 300
		betas = betas.astype(np.float32) / 300
		# z-score the betas of each voxel within the scan session
		betas = zscore(betas, nan_policy='omit')

		# Get the NSD synthetic image condition repeats
		nsdsynthetic_img_num = np.unique(masterordering)
		nsdsynthetic_img_repeats = np.zeros(len(nsdsynthetic_img_num))
		for i, img in enumerate(nsdsynthetic_img_num):
			idx = np.where(masterordering == img)[0]
			nsdsynthetic_img_repeats[i] = len(idx)

		# Compute the ncsnr
		# When computing the ncsnr on image conditions with different amounts of trials
		# (i.e., different sample sizes), you need to correct for this:
		# https://stats.stackexchange.com/questions/488911/combined-variance-estimate-for-samples-of-varying-sizes
		num_var = np.zeros((betas.shape[1]))
		den_var = np.zeros((betas.shape[1]))
		for i, img in enumerate(nsdsynthetic_img_num):
			idx = np.where(masterordering == img)[0]
			num_var += np.var(betas[idx], axis=0, ddof=1) * (len(idx) - 1)
			den_var += len(idx) - 1
		sigma_noise = np.sqrt(num_var/den_var)
		var_data = np.var(betas, axis=0, ddof=1)
		sigma_signal = var_data - (sigma_noise ** 2)
		sigma_signal[sigma_signal<0] = 0
		sigma_signal = np.sqrt(sigma_signal)
		ncsnr = sigma_signal / sigma_noise

		# Convert the ncsnr to noise ceiling
		img_reps_2 = 236
		img_reps_4 = 32
		img_reps_8 = 8
		img_reps_10 = 8
		norm_term = (img_reps_2/2 + img_reps_4/4 + img_reps_8/8 + img_reps_10/10) / \
			(img_reps_2 + img_reps_4 + img_reps_8 + img_reps_10)
		noise_ceil = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)

		# Average the fMRI across repeats
		invivo_fmri = np.zeros((len(nsdsynthetic_img_num), betas.shape[1]))
		for i, img in enumerate(nsdsynthetic_img_num):
			idx = np.where(masterordering == img)[0]
			invivo_fmri[i] = np.nanmean(betas[idx], 0)


# =============================================================================
# Generate the in silico fMRI responses for NSD-synthetic
# =============================================================================
		# Load the NSD-synthetic stimulus images
		images = []
		# Load the 220 NSD-synthetic stimuli
		stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
			'nsdsynthetic', 'nsdsynthetic_stimuli.hdf5')
		sf = h5py.File(stimuli_dir, 'r')
		sdataset = sf.get('imgBrick')
		for img in sdataset:
			img = (np.sqrt(img/255)*255).astype(np.uint8)
			img = Image.fromarray(img).convert('RGB')
			# Center crop the image to square size
			size = min(img.size)  # Get the size of the smallest dimension
			left = (img.width - size) // 2
			top = (img.height - size) // 2
			right = left + size
			bottom = top + size
			img = img.crop((left, top, right, bottom))
			img = np.transpose(np.asarray(img), (2, 0, 1))
			images.append(img)
			del img
		# Load the 64 NSD-synthetic colorstimuli
		stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
			'nsdsynthetic', 'nsdsynthetic_colorstimuli_subj0'+str(s)+
			'.hdf5')
		sf = h5py.File(stimuli_dir, 'r')
		sdataset = sf.get('imgBrick')
		for img in sdataset:
			img = (np.sqrt(img/255)*255).astype(np.uint8)
			img = Image.fromarray(img).convert('RGB')
			# Center crop the image to square size
			size = min(img.size)  # Get the size of the smallest dimension
			left = (img.width - size) // 2
			top = (img.height - size) // 2
			right = left + size
			bottom = top + size
			img = img.crop((left, top, right, bottom))
			img = np.transpose(np.asarray(img), (2, 0, 1))
			images.append(img)
			del img
		images = np.asarray(images)

		# Load the trained encoding model weights
		# ROI 'FFA' is divided into two parts (anterior and posterior):
		# https://cvnlab.slite.page/p/X_7BBMgghj/ROIs
		if r in ['FFA']:
			# Encoding model for ROI split 1
			encoding_model_1 = berg_object.get_encoding_model(
				model_id='fmri-nsd-fwrf',
				subject=s,
				selection={'roi': r+'-1'},
				device='auto'
				)
			# Encoding model for ROI split 2
			encoding_model_2 = berg_object.get_encoding_model(
				model_id='fmri-nsd-fwrf',
				subject=s,
				selection={'roi': r+'-2'},
				device='auto'
				)
		else:
			encoding_model = berg_object.get_encoding_model(
				model_id='fmri-nsd-fwrf',
				subject=s,
				selection={'roi': r},
				device='auto'
				)

		# Generate the in silico fMRI responses to images
		if r in ['FFA']:
			insilico_fmri_1 = np.squeeze(berg_object.encode(
				encoding_model_1,
				images,
				return_metadata=False
				))
			insilico_fmri_2 = np.squeeze(berg_object.encode(
				encoding_model_2,
				images,
				return_metadata=False
				))
			insilico_fmri = np.append(insilico_fmri_1, insilico_fmri_2, 1)
			del insilico_fmri_1, insilico_fmri_2
		else:
			insilico_fmri = np.squeeze(berg_object.encode(
				encoding_model,
				images,
				return_metadata=False
				))
		insilico_fmri = insilico_fmri.astype(np.float32)


# =============================================================================
# Voxel selection
# =============================================================================
		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'nsd_encoding_models', 'insilico_fmri', 'ncsnr_sub-'+
			format(s, '02')+'_roi-'+r+'.npy')
		ncsnr = np.load(ncsnr_dir)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold.
		best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
		invivo_fmri = invivo_fmri[:,best_voxels]
		insilico_fmri = insilico_fmri[:,best_voxels]
		noise_ceil = noise_ceil[best_voxels]


# =============================================================================
# Compute the in silico fMRI responses encoding accuracy
# =============================================================================
		# Correlate the insilico and ground truth test fMRI responses
		correlation = np.zeros(insilico_fmri.shape[1])
		for v in range(len(correlation)):
			correlation[v] = pearsonr(invivo_fmri[:,v], insilico_fmri[:,v])[0]

		# Set negative correlation values to 0, so to keep the
		# noise-ceiling-normalized encoding accuracy positive
		correlation[correlation<0] = 0

		# Square the correlation values
		r2 = correlation ** 2

		# Add a very small number to noise ceiling values of 0, otherwise
		# the noise-ceiling-normalized encoding accuracy cannot be calculated
		# (division by 0 is not possible)
		noise_ceil[noise_ceil==0] = 1e-14

		# Compute the noise-ceiling-normalized encoding accuracy
		expl_val = np.divide(r2, noise_ceil)

		# Set the noise-ceiling-normalized encoding accuracy to 1 for those
		# vertices in which the correlation is higher than the noise
		# ceiling, to prevent encoding accuracy values higher than 100%
		expl_val[expl_val>1] = 1

		# Store the explained variance scores
		explained_variance['s'+str(s)+'_'+r] = np.nanmean(expl_val)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Random seeds
seed = 20200220
random.seed(seed)
np.random.seed(seed)

ci_lower = {}
ci_upper = {}
for r in tqdm(args.rois):
	sample_dist = np.zeros(args.n_iter)
	mean_encoding_acc = []
	for s in args.all_subjects:
		mean_encoding_acc.append(np.mean(explained_variance['s'+str(s)+'_'+r]))
	mean_encoding_acc = np.asarray(mean_encoding_acc)
	for i in range(args.n_iter):
		sample_dist[i] = np.mean(resample(mean_encoding_acc))
	ci_lower[r] = np.percentile(sample_dist, 2.5)
	ci_upper[r] = np.percentile(sample_dist, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
	'explained_variance': explained_variance,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper
}

save_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'nsd_encoding_models')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'ood_encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
