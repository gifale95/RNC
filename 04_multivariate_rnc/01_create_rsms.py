"""Create the in silico fMRI RSMs that will be later used by the multivariate
RNC algorithm. Each RSM consists in the pairwise comparisons for all images
from the chosen imageset.

This code is available at:
https://github.com/gifale95/RNC/blob/main/04_multivariate_rnc/01_create_rsms.py

Parameters
----------
sub : int
	Used subject.
roi : str
	Used ROI. Possible choices are 'V1', 'V3', 'V2', 'hV4'.
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
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED

"""

import argparse
import os
import numpy as np
import h5py
from ned.ned import NED
from tqdm import tqdm
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--total_rsm_splits', type=int, default=50)
parser.add_argument('--rsm_split', type=int, default=1)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
args = parser.parse_args()

print('>>> Create RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize the Neural Encoding Dataset (NED) object
# =============================================================================
# https://github.com/gifale95/NED
ned_object = NED(args.ned_dir)


# =============================================================================
# Load the in silico fMRI responses
# =============================================================================
# Load the in silico fMRI responses
data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'imageset-'+args.imageset, 'insilico_fmri_responses_sub-'+
	format(args.sub, '02')+'_roi-'+args.roi+'.h5')
betas = h5py.File(data_dir).get('insilico_fmri_responses')

# Load the in silico fMRI responses metadata
metadata = ned_object.get_metadata(
	modality='fmri',
	train_dataset='nsd',
	model='fwrf',
	subject=args.sub,
	roi=args.roi
	)

# Only retain voxels with noise ceiling signal-to-noise ratio scores above the
# selected threshold
best_voxels = np.where(metadata['fmri']['ncsnr'] > args.ncsnr_threshold)[0]
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
betas_rsm = np.ones((len(used_img_cond), len(betas)), dtype=np.float32)
for c1, cond_1 in enumerate(tqdm(used_img_cond)):
	for c2 in range(cond_1):
		betas_rsm[c1,c2] = pearsonr(betas[cond_1], betas[c2])[0]


# =============================================================================
# Save the RSMs
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'rsms',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'rsm_sub-' + format(args.sub, '02') + '_' + args.roi + '_split-' + \
	format(args.rsm_split, '02')

np.save(os.path.join(save_dir, file_name), betas_rsm)

