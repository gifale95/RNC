"""Prepare the Visual Illusion Reconstruction dataset fMRI responses, for
encoding model training and testing.

The Visual Illusion Reconstruction dataset fMRI responses are found at:
https://figshare.com/articles/dataset/Reconstructing_visual_illusory_experiences_from_human_brain_activity/23590302

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
subject : int
	Used subject, out of the 7 Visual Illusion Reconstruction dataset subjects.
roi : str
	Used ROI.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy.stats import zscore
import bdpy # https://github.com/KamitaniLab/bdpy

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Prepare the Visual Illusion Reconstruction dataset fMRI responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Dataset types
# =============================================================================
datasets = [
	'ImageNetTraining', # 1,200 images from 150 object categories (ImageNet)
	'FMD', # 1,000 images from 9 material categories (Flickr Material Database)
	'MSCOCO', # 1,000 images (MS COCO)
	'Illusion' # 38 images (32 for subject 4)
	]


# =============================================================================
# Image repeats
# =============================================================================
if args.subject in [1, 2, 3, 4]:
	image_repeats = {
		'ImageNetTraining': 5,
		'FMD': 5,
		'MSCOCO': 5
		}

elif args.subject in [5, 6, 7]:
	image_repeats = {
		'ImageNetTraining': 2,
		'FMD': 2,
		'MSCOCO': 2
		}


# =============================================================================
# Load the fMRI responses
# =============================================================================
fmri = {}
image_index = {}

for dataset in datasets:

	data_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
		'fmri', 'S'+str(args.subject)+'_'+dataset+'.h5')
	bdata = bdpy.BData(data_dir)
	fmri[dataset] = zscore(bdata.select('ROI_'+args.roi), nan_policy='omit')
	image_index[dataset] = np.squeeze(
		bdata.select('image_index')-1).astype(np.int32) # subtract 1, since Python numbering is zero-based
	# Shift the Illusion images indices to start from zero
	if dataset == 'Illusion':
		if args.subject == 4:
			image_index[dataset] = image_index[dataset] - 32
		else:
			image_index[dataset] = image_index[dataset] - 38


# =============================================================================
# Compute the noise ceiling signal-to-noise ratio
# =============================================================================
# The ncsnr is computed following the method proposed in the Natural Scenes
# Dataset (NSD) paper: https://www.nature.com/articles/s41593-021-00962-x

ncsnr = {}

for dataset in fmri.keys():

	# Estimate the noise standard deviation
	# Calculate the variance of the betas across the presentations of each
	# image condition
	sigma_noise = []
	for img in np.unique(image_index[dataset]):
		idx = np.where(image_index[dataset] == img)[0]
		sigma_noise.append(np.var(fmri[dataset][idx], axis=0, ddof=1))
	# Average the variance across images and compute the square root of the
	# result
	sigma_noise = np.sqrt(np.mean(sigma_noise, 0))

	# Estimate the signal standard deviation
	var_data = np.var(fmri[dataset], axis=0, ddof=1)
	sigma_signal = var_data - (sigma_noise ** 2)
	sigma_signal[sigma_signal<0] = 0
	sigma_signal = np.sqrt(sigma_signal)

	# Compute the ncsnr
	ncsnr[dataset] = sigma_signal / sigma_noise


# =============================================================================
# Compute the noise ceiling signal-to-noise ratio using all ID stimulus images
# =============================================================================
# When computing the ncsnr on image conditions with different amounts of trials
# (i.e., different sample sizes), I need to correct for this:
# https://stats.stackexchange.com/questions/488911/combined-variance-estimate-for-samples-of-varying-sizes

# Append the image indices
image_index_all = np.append(image_index['ImageNetTraining'],
	image_index['FMD']+len(np.unique(image_index['ImageNetTraining'])))
image_index_all = np.append(image_index_all, image_index['MSCOCO']+
	len(np.unique(image_index['ImageNetTraining']))+
	len(np.unique(image_index['FMD'])))

# Append the fMRI responses
fmri_all = np.append(fmri['ImageNetTraining'], fmri['FMD'], 0)
fmri_all = np.append(fmri_all, fmri['MSCOCO'], 0)

# Compute the ncsnr
num_var = np.zeros((fmri_all.shape[1]))
den_var = np.zeros((fmri_all.shape[1]))
for img in np.unique(image_index_all):
	idx = np.where(image_index_all == img)[0]
	num_var += np.var(fmri_all[idx], axis=0, ddof=1) * (len(idx) - 1)
	den_var += len(idx) - 1
sigma_noise = np.sqrt(num_var/den_var)
var_data = np.var(fmri_all, axis=0, ddof=1)
sigma_signal = var_data - (sigma_noise ** 2)
sigma_signal[sigma_signal<0] = 0
sigma_signal = np.sqrt(sigma_signal)
ncsnr_all = sigma_signal / sigma_noise


# =============================================================================
# Save the prepared fMRI responses
# =============================================================================
data = {
	'fmri': fmri,
	'image_index': image_index,
	'image_repeats': image_repeats,
	'ncsnr': ncsnr,
	'ncsnr_all': ncsnr_all
	}

save_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'prepared_data', 'fmri')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

save_file = 'fmri_sub-0' + str(args.subject) + '_roi-' + args.roi + '.npy'

np.save(os.path.join(save_dir, save_file), data)
