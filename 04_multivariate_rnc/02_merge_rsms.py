"""Combine the in silico fMRI RSMs across partitions, and merge them across all
subjects.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
roi : str
	Used ROI.
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
from tqdm import tqdm
import h5py
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
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
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# Combine the in silico fMRI RSMs across partitions, and merge them across all
# subjects
# =============================================================================
# Get the image conditions number
if args.imageset == 'nsd':
	img_cond = 73000
elif args.imageset == 'imagenet_val':
	img_cond = 50000
elif args.imageset == 'things':
	img_cond = 26107

# Create the empty merged RSMs
rsms_all_subj = np.zeros((int(np.ceil(len(all_subjects)/2)), img_cond,
	img_cond), dtype=np.float32)

for s, sub in enumerate(tqdm(all_subjects)):

	# Combine the in silico fMRI RSMs across splits
	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models',
		'rsms', 'imageset-'+args.imageset)
	for split in range(args.total_rsm_splits):
		rsm_file = 'rsm_sub-' + format(sub, '02') + '_' + args.roi + \
			'_split-' + format(split+1, '02') + '.npy'
		if split == 0:
			rsm = np.load(os.path.join(data_dir, rsm_file))
		else:
			rsm = np.append(rsm, np.load(os.path.join(data_dir, rsm_file)), 0)
		# Force garbage collection to free memory
		gc.collect()

	# Merge the RSMs across subjects. Fill the RSM's lower-triangular matrices
	# with even subjects, and the upper-triangular matrices with odd subjects.
	idx = int(np.floor(s / 2))
	if s % 2 == 0:
		rsms_all_subj[idx] += rsm
	else:
		rsms_all_subj[idx] += np.transpose(rsm)
	del rsm
	# Force garbage collection to free memory
	gc.collect()


# =============================================================================
# Save the merged RSMs
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'rsms', 'imageset-'+
	args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

save_file = os.path.join(save_dir, 'merged_rsms_'+args.roi)

with h5py.File(save_file+'.h5py', 'w') as hf:
	for k,v in {'rsms': rsms_all_subj}.items():
		hf.create_dataset(k,data=v)
