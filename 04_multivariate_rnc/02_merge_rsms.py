"""Combine the in silico fMRI RSMs across partitions, and merge them across all
subjects.

This code is available at:
https://github.com/gifale95/RNC/blob/main/04_multivariate_rnc/02_merge_rsms.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
roi : str
	Used ROI. Possible choices are 'V1', 'V3', 'V2', 'hV4'.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
total_rsm_splits : int
	Number of total RSM splits.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from copy import copy
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--total_rsm_splits', type=int, default=50)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Merge RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Combine the in silico fMRI RSMs across partitions, and merge them across all
# subjects
# =============================================================================
# Get the image conditions number
if args.imageset == 'nsd':
	img_cond = np.arange(73000)
elif args.imageset == 'imagenet_val':
	img_cond = np.arange(50000)
elif args.imageset == 'things':
	img_cond = np.arange(26107)

# Create the empty merged RSMs
rsms_all_subj = np.zeros((int(np.ceil(len(args.all_subjects)/2)),
	len(img_cond), len(img_cond)), dtype=np.float32)
idx_upper_tr = np.triu_indices(len(img_cond), 1)
idx_lower_tr = np.tril_indices(len(img_cond), -1)

for s, sub in enumerate(args.all_subjects):

	# Combine the in silico fMRI RSMs across partitions
	data_dir = os.path.join(args.project_dir, args.project_dir,
		'multivariate_rnc', 'rsms', 'imageset-'+args.imageset)
	for split in range(args.total_rsm_splits):
		rsm_file = 'rsm_sub-' + format(sub, '02') + '_' + args.roi + \
			'_split-' + format(split+1, '02') + '.npy'
		rsm_part = np.load(os.path.join(data_dir, rsm_file))
		if split == 0:
			rsm = copy(rsm_part)
		else:
			rsm = np.append(rsm, copy(rsm_part), 0)
	idx_upper_tr = np.triu_indices(len(rsm), 1)
	rsm[idx_upper_tr] = np.transpose(rsm)[idx_upper_tr]

	# Merge the RSMs across subjects. Fill the RSM's lower-triangular matrices
	# with even subjects, and the upper-triangular matrices with odd subjects.
	idx = int(np.floor(s / 2))
	if s % 2 == 0:
		rsms_all_subj[idx,idx_lower_tr[0],idx_lower_tr[1]] = \
			copy(rsm[idx_lower_tr])
	else:
		rsms_all_subj[idx,idx_upper_tr[0],idx_upper_tr[1]] = \
			copy(rsm[idx_upper_tr])
	del rsm


# =============================================================================
# Save the merged RSMs
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'rsms',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

save_file = os.path.join(save_dir, 'merged_rsms_'+args.roi)

with h5py.File(save_file+'.h5py', 'w') as hf:
	for k,v in {'rsms': rsms_all_subj}.items():
		hf.create_dataset(k,data=v)

