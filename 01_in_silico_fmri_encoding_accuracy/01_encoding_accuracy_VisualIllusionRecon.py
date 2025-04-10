"""Compute the in-distribution (ID, natural images) and out-of-distribution
(OOD, illusory images) encoding accuracy of the Visual Illusion Reconstruction
dataset fMRI encoding models.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 7 subjects from the Visual Illusion
	Reconstruction dataset.
rois : list of str
	List of used ROIs.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
n_iter : int
	Amount of iterations for the permutation stats.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import pearsonr
import torch

from src_new.load_nsd import image_feature_fn
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Encoding accuracy - VisualIllusionRecon <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda:1") #cuda


# =============================================================================
# Empty results dictionaries
# =============================================================================
explained_variance = {}
explained_variance['id'] = {}
explained_variance['ood'] = {}
ci_lower = {}
ci_upper = {}
ci_lower['id'] = {}
ci_lower['ood'] = {}
ci_upper['id'] = {}
ci_upper['ood'] = {}


# =============================================================================
# Define the testing splits (ID)
# =============================================================================
# The ID testing split consists of the 8th (out of 8) image for each of the
# 150 'NaturalImageTraining' categories (for a total of 150 images); plus
# images 90 to 100 (out of 100) from each of the 10 'FMD' categories (for a
# total of 100 images); plus images 900 to 1,000 (out of 1,000) from 'MSCOCO'.
# Thus, the ID testing split includes a total of 350 images.

# Select the validation image conditions
test_img = np.empty(0, dtype=np.int32)
# 'NaturalImageTraining'
categories = 150
exemplars = 8
test_exemplars = np.arange(7, 8)
for c in range(categories):
	test_img = np.append(test_img, test_exemplars+(c*exemplars))
# 'FMD'
categories = 10
exemplars = 100
test_exemplars = np.arange(90, 100)
offset = 1200
for c in range(categories):
	test_img = np.append(test_img, test_exemplars+(c*exemplars)+offset)
# 'MSCOCO'
test_exemplars = np.arange(900, 1000)
offset = 2200
test_img = np.append(test_img, test_exemplars+offset)


# =============================================================================
# Loop over subjects and ROIs
# =============================================================================
for s in tqdm(args.all_subjects, leave=False):
	for r in args.rois:


# =============================================================================
# Load the in vivo fMRI data (ID)
# =============================================================================
		# Load the data
		data_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
			'prepared_data', 'fmri', 'fmri_sub-0'+str(s)+'_roi-'+r+'.npy')
		data = np.load(data_dir, allow_pickle=True).item()

		# Append the fMRI responses
		fmri = []
		imagesets = ['ImageNetTraining', 'FMD', 'MSCOCO']
		for imageset in imagesets:
			fmri.append(data['fmri'][imageset])
		fmri = np.concatenate(fmri, 0)

		# Append the image indices of all images
		image_index = data['image_index']
		image_index_all = np.append(image_index['ImageNetTraining'],
			image_index['FMD']+len(np.unique(image_index['ImageNetTraining'])))
		image_index_all = np.append(image_index_all, image_index['MSCOCO']+
			len(np.unique(image_index['ImageNetTraining']))+
			len(np.unique(image_index['FMD'])))
		unique_images = np.unique(image_index_all)

		# Average the fMRI responses for the ID test images across repeats
		invivo_fmri = []
		repeats = []
		for img in test_img:
			idx = np.where(image_index_all == img)[0]
			repeats.append(len(idx))
			invivo_fmri.append(np.mean(fmri[idx], 0))
		invivo_fmri = np.asarray(invivo_fmri)
		repeats = np.asarray(repeats)

		# Convert the ncsnr to noise ceiling
		ncsnr = data['ncsnr_all']
		norm_term = np.zeros(1)
		unique, unique_count = np.unique(repeats, return_counts=True)
		for i in range(len(unique)):
			norm_term += unique_count[i] / unique[i]
		norm_term = norm_term / len(repeats)
		noise_ceiling = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)


# =============================================================================
# Load the in vivo fMRI data (OOD)
# =============================================================================
		# Average the fMRI across repeats
		invivo_fmri_ood = []
		repeats = []
		image_index = data['image_index']['Illusion']
		for img in np.unique(image_index):
			idx = np.where(image_index == img)[0]
			repeats.append(len(idx))
			invivo_fmri_ood.append(np.mean(data['fmri']['Illusion'][idx], 0))
		invivo_fmri_ood = np.asarray(invivo_fmri_ood)
		repeats = np.asarray(repeats)

		# Convert the ncsnr to noise ceiling (OOD)
		ncsnr = data['ncsnr']['Illusion']
		norm_term = np.zeros(1)
		unique, unique_count = np.unique(repeats, return_counts=True)
		for i in range(len(unique)):
			norm_term += unique_count[i] / unique[i]
		norm_term = norm_term / len(repeats)
		noise_ceiling_ood = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)
		del data


# =============================================================================
# Load the stimulus images (ID)
# =============================================================================
		# Load the images
		data_dir = os.path.join(args.project_dir,
			'VisualIllusionRecon_dataset', 'prepared_data', 'images')
		images_all = []
		if s == 6:
			imagesets = ['ImageNetTraining_S6', 'FMD', 'MSCOCO']
		else:
			imagesets = ['ImageNetTraining', 'FMD', 'MSCOCO']
		for imageset in imagesets:
			file_name = 'images_dataset-' + imageset + '_encoding-fwrf.npy'
			images_all.append(np.load(os.path.join(data_dir, file_name)))
		images_all = np.concatenate(images_all, 0)

		# Select the images from the testing split
		images = []
		for i in test_img:
			images.append(images_all[i])
		images = np.asarray(images)
		del images_all


# =============================================================================
# Load the stimulus images (OOD)
# =============================================================================
		if s == 4:
			imageset = 'Illusion_S4'
		else:
			imageset = 'Illusion'
		file_name = 'images_dataset-' + imageset + '_encoding-fwrf.npy'
		images_ood = np.load(os.path.join(data_dir, file_name))


# =============================================================================
# Load the trained encoding models
# =============================================================================
		# Load the trained encoding model weights
		model_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'VisualIllusionRecon_encoding_models', 'trained_encoding_models',
			'weights_sub-0'+str(s)+'_roi-'+r+'.pt')
		trained_model = torch.load(model_dir, map_location=torch.device('cpu'))
		stim_mean = trained_model['stim_mean']

		# Model instantiation
		# Model functions
		_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)
		def _model_fn(_ext, _con, _x):
			'''model consists of an extractor (_ext) and a connection model (_con)'''
			_y, _fm, _h = _ext(_x)
			return _con(_fm)
		def _pred_fn(_ext, _con, xb):
			return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))
		# Shared encoder model
		stim_data = image_feature_fn(images[:20])
		backbone_model = Encoder(mu=stim_mean, trunk_width=64,
			use_prefilter=1).to(device)
		rec, fmaps, h = backbone_model(torch.from_numpy(stim_data).to(device))
		# Subject specific FWRF models
		fwrf_model = Torch_LayerwiseFWRF(fmaps, nv=invivo_fmri.shape[1],
			pre_nl=_log_act_fn, post_nl=_log_act_fn,
			dtype=np.float32).to(device)

		# Load the pretrained weights into the model
		backbone_model.load_state_dict(trained_model['best_params']['enc'])
		fwrf_model.load_state_dict(trained_model['best_params']['fwrf'])
		backbone_model.eval()
		fwrf_model.eval()


# =============================================================================
# Generate the in silico fMRI responses
# =============================================================================
		# ID testing images
		insilico_fmri = subject_pred_pass(_pred_fn, backbone_model,
			fwrf_model, image_feature_fn(images), batch_size=len(images))

		# OOD testing images
		insilico_fmri_ood = subject_pred_pass(_pred_fn, backbone_model,
			fwrf_model, image_feature_fn(images_ood), batch_size=len(images))


# =============================================================================
# Voxel selection
# =============================================================================
		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'VisualIllusionRecon_encoding_models', 'insilico_fmri',
			'ncsnr_sub-'+format(s, '02')+'_roi-'+r+'.npy')
		ncsnr = np.load(ncsnr_dir)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold.
		best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
		# For subject 4 of the Visual Illusion Reconstruction dataset, lower the
		# ncsnr theshold for ROI hV4 to 0.4, since there are no voxels above a
		# threshold of 0.5
		if r == 'hV4' and s == 4 and args.ncsnr_threshold > 0.4:
			best_voxels = np.where(ncsnr > 0.4)[0]
		insilico_fmri = insilico_fmri[:,best_voxels]
		insilico_fmri_ood = insilico_fmri_ood[:,best_voxels]
		invivo_fmri = invivo_fmri[:,best_voxels]
		invivo_fmri_ood = invivo_fmri_ood[:,best_voxels]
		noise_ceiling = noise_ceiling[best_voxels]
		noise_ceiling_ood = noise_ceiling_ood[best_voxels]


# =============================================================================
# Compute the in silico fMRI responses encoding accuracy
# =============================================================================
		# Correlate the insilico and ground truth test fMRI responses
		correlation = np.zeros(insilico_fmri.shape[1])
		correlation_ood = np.zeros(insilico_fmri.shape[1])
		for v in range(len(correlation)):
			correlation[v] = pearsonr(invivo_fmri[:,v], insilico_fmri[:,v])[0]
			correlation_ood[v] = pearsonr(invivo_fmri_ood[:,v],
				insilico_fmri_ood[:,v])[0]

		# Set negative correlation values to 0, so to keep the
		# noise-ceiling-normalized encoding accuracy positive
		correlation[correlation<0] = 0
		correlation_ood[correlation_ood<0] = 0

		# Square the correlation values
		r2 = correlation ** 2
		r2_ood = correlation_ood ** 2

		# Add a very small number to noise ceiling values of 0, otherwise
		# the noise-ceiling-normalized encoding accuracy cannot be calculated
		# (division by 0 is not possible)
		noise_ceiling[noise_ceiling==0] = 1e-14
		noise_ceiling_ood[noise_ceiling_ood==0] = 1e-14

		# Compute the noise-ceiling-normalized encoding accuracy
		expl_val = np.divide(r2, noise_ceiling)
		expl_val_ood = np.divide(r2_ood, noise_ceiling_ood)

		# Set the noise-ceiling-normalized encoding accuracy to 1 for those
		# vertices in which the correlation is higher than the noise
		# ceiling, to prevent encoding accuracy values higher than 100%
		expl_val[expl_val>1] = 1
		expl_val_ood[expl_val_ood>1] = 1

		# Store the explained variance scores
		explained_variance['id']['s'+str(s)+'_'+r] = np.nanmean(expl_val)
		explained_variance['ood']['s'+str(s)+'_'+r] = np.nanmean(expl_val_ood)


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Random seeds
seed = 20200220
random.seed(seed)
np.random.seed(seed)

for r in tqdm(args.rois):
	sample_dist = np.zeros(args.n_iter)
	sample_dist_ood = np.zeros(args.n_iter)
	mean_encoding_acc = []
	mean_encoding_acc_ood = []
	for s in args.all_subjects:
		mean_encoding_acc.append(np.mean(
			explained_variance['id']['s'+str(s)+'_'+r]))
		mean_encoding_acc_ood.append(np.mean(
			explained_variance['ood']['s'+str(s)+'_'+r]))
	mean_encoding_acc = np.asarray(mean_encoding_acc)
	mean_encoding_acc_ood = np.asarray(mean_encoding_acc_ood)
	for i in range(args.n_iter):
		sample_dist[i] = np.mean(resample(mean_encoding_acc))
		sample_dist_ood[i] = np.mean(resample(mean_encoding_acc_ood))
	ci_lower['id'][r] = np.percentile(sample_dist, 2.5)
	ci_upper['id'][r] = np.percentile(sample_dist, 97.5)
	ci_lower['ood'][r] = np.percentile(sample_dist_ood, 2.5)
	ci_upper['ood'][r] = np.percentile(sample_dist_ood, 97.5)


# =============================================================================
# Save the results
# =============================================================================
results = {
	'explained_variance': explained_variance,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
}

save_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'VisualIllusionRecon_encoding_models')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
