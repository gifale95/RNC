"""Create the in silico fMRI RSMs that will be later used by the multivariate
RNC algorithm. Each RSM consists in the pairwise comparisons for all images
from the chosen imageset.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
subject : int
	Used subject, out of the 8 NSD subjects
	(if encoding_models_train_dataset=='nsd'), or the 7 Visual Illusion
	Reconstruction dataset subjects
	(if encoding_models_train_dataset=='VisualIllusionRecon').
roi : str
	Used ROI.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
total_rsm_splits : int
	Number of total RSM splits.
rsm_split : int
	Integer indicating the RSM partition to create. To reduce compute time the
	RSM creation is split into multiple partitions that can run in parallel.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--total_rsm_splits', type=int, default=50)
parser.add_argument('--rsm_split', type=int, default=1)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Create RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
# Load the in silico fMRI responses
data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	args.encoding_models_train_dataset+'_encoding_models', 'insilico_fmri',
	'imageset-'+args.imageset, 'insilico_fmri_responses_sub-'+
	format(args.subject, '02')+'_roi-'+args.roi+'.h5')
betas = h5py.File(data_dir).get('insilico_fmri_responses')

# Load the ncsnr
ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	args.encoding_models_train_dataset+'_encoding_models', 'insilico_fmri',
	'ncsnr_sub-'+format(args.subject, '02')+'_roi-'+args.roi+'.npy')
ncsnr = np.load(ncsnr_dir)

# Only retain voxels with noise ceiling signal-to-noise ratio scores
# above the selected threshold.
best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
# For subject 4 of the Visual Illusion Reconstruction dataset, lower the
# ncsnr theshold for ROI hV4 to 0.4, since there are no voxels above a
# threshold of 0.5
if args.encoding_models_train_dataset == 'VisualIllusionRecon':
	if args.roi == 'hV4' and args.subject == 4 and args.ncsnr_threshold > 0.4:
		best_voxels = np.where(ncsnr > 0.4)[0]
betas = betas[:,best_voxels]


# =============================================================================
# Create the in silico fMRI responses RSM
# =============================================================================
# Establish which RSM partition to create
all_img_cond = np.arange(len(betas))
idx = int(np.ceil(len(betas) / args.total_rsm_splits))
idx_start = int(idx * (args.rsm_split-1))
idx_end = int(idx * (args.rsm_split))
used_img_cond = all_img_cond[idx_start:idx_end]

# Create the RSM
betas_rsm = np.zeros((len(used_img_cond), len(betas)), dtype=np.float32)
for c1, cond_1 in enumerate(tqdm(used_img_cond)):
	for c2 in range(cond_1):
		betas_rsm[c1,c2] = pearsonr(betas[cond_1], betas[c2])[0]


# =============================================================================
# Save the RSMs
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'rsms', 'imageset-'+
	args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'rsm_sub-' + format(args.subject, '02') + '_' + args.roi + \
	'_split-' + format(args.rsm_split, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), betas_rsm)
