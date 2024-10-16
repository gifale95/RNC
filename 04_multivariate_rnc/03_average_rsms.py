"""Average the in silico fMRI RSMs across subjects.

If cross-validation is used, the RSMs are averaged across N-1 (train) subjects.
Multivariate RNC will later be applied on these averaged RSMs, and the resulting
controlling images validated on the RSMs of the left-out (test) subject.

If cross-validation is not used, the RSMs are averaged across all subjects, and
multivariate RNC will later be applied on the these RSMs to select the
controlling images.

This code is available at:
https://github.com/gifale95/RNC/blob/main/04_multivariate_rnc/03_average_rsms.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
cv : int
	If '1' the RSM of one subject is left out for cross-validation, and the RSMs
	are averaged across the remaining N-1 subjects. If '0' the RSMs are averaged
	across all subjects.
cv_subject : int
	If 'cv==0' the left-out subject during cross-validation, out of all 8 (NSD)
	subjects.
roi : str
	Used ROI. Possible choices are 'V1', 'V3', 'V2', 'hV4'.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Average RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the merged RSMs
# =============================================================================
dir_rsm = os.path.join(args.project_dir, 'multivariate_rnc', 'rsms',
	'imageset-'+args.imageset)
rsms = h5py.File(os.path.join(dir_rsm, 'merged_rsms_'+args.roi+'.h5py'),
	'r')['rsms']

idx_upper_tr = np.triu_indices(rsms.shape[1], 1)
idx_lower_tr = np.tril_indices(rsms.shape[1], -1)


# =============================================================================
# Average the RSMs across subjects (CV == 0)
# =============================================================================
if args.cv == 0:

	rsm_all = np.zeros((len(args.all_subjects), rsms.shape[1], rsms.shape[1]),
		dtype=np.float32)

	for s, sub in enumerate(args.all_subjects):
		idx_rsm = int(np.floor(s / 2))
		rsm = copy(rsms[idx_rsm])
		if s % 2 == 0:
			rsm[idx_upper_tr] = np.transpose(rsm)[idx_upper_tr]
		else:
			rsm[idx_lower_tr] = np.transpose(rsm)[idx_lower_tr]
		rsm_all[s] = rsm

	rsm_all = np.nanmean(rsm_all, 0)


# =============================================================================
# Average the RSMs across subjects (CV == 1)
# =============================================================================
elif args.cv == 1:

	rsm_train = np.zeros((len(args.all_subjects)-1, rsms.shape[1], rsms.shape[1]),
		dtype=np.float32)

	idx_sub = 0
	for s, sub in enumerate(args.all_subjects):
		idx_rsm = int(np.floor(s / 2))
		rsm = copy(rsms[idx_rsm])
		if s % 2 == 0:
			rsm[idx_upper_tr] = np.transpose(rsm)[idx_upper_tr]
		else:
			rsm[idx_lower_tr] = np.transpose(rsm)[idx_lower_tr]
		if sub == args.cv_subject:
			rsm_test = rsm
		else:
			rsm_train[idx_sub] = rsm
			idx_sub += 1

	rsm_train = np.nanmean(rsm_train, 0)


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'rsms',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'averaged_rsm_' + args.roi + '_all_subjects'
	np.save(os.path.join(save_dir, file_name), rsm_all)

elif args.cv == 1:
	file_name_train = 'averaged_rsm_' + args.roi + '_cv_subject-' + \
		format(args.cv_subject, '02') + '_train'
	file_name_test = 'averaged_rsm_' + args.roi + '_cv_subject-' + \
		format(args.cv_subject, '02') + '_test'
	np.save(os.path.join(save_dir, file_name_train), rsm_train)
	np.save(os.path.join(save_dir, file_name_test), rsm_test)

