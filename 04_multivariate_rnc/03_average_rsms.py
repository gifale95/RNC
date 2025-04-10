"""Average the in silico fMRI RSMs across subjects.

If cross-validation is used, the RSMs are averaged across N-1 (train) subjects.
Multivariate RNC will later be applied on these averaged RSMs, and the
resulting controlling images validated on the RSMs of the left-out (test)
subject.

If cross-validation is not used, the RSMs are averaged across all subjects, and
multivariate RNC will later be applied on the these RSMs to select the
controlling images.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
cv : int
	If '1' multivariate RNC leaves the data of one subject out for
	cross-validation, if '0' multivariate RNC uses the data of all subjects.
cv_subject : int
	If cv==1, the left-out subject during cross-validation, out of the 8 NSD
	subjects (if encoding_models_train_dataset=='nsd'), or the 7 Visual Illusion
	Reconstruction dataset subjects
	(if encoding_models_train_dataset=='VisualIllusionRecon').
roi : str
	Used ROI.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
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
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# Load the merged RSMs
# =============================================================================
dir_rsm = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'rsms', 'imageset-'+args.imageset)
rsms = h5py.File(os.path.join(dir_rsm, 'merged_rsms_'+args.roi+'.h5py'),
	'r')['rsms']

idx_upper_tr = np.triu_indices(rsms.shape[1], 1)
idx_lower_tr = np.tril_indices(rsms.shape[1], -1)


# =============================================================================
# Average the RSMs across subjects (CV == 0)
# =============================================================================
if args.cv == 0:

	rsm_all = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)

	for s, sub in enumerate(all_subjects):
		idx_rsm = int(np.floor(s / 2))
		if s % 2 == 0:
			rsm_all[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
			rsm_all[idx_upper_tr] += np.transpose(rsms[idx_rsm])[idx_upper_tr]
		else:
			rsm_all[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
			rsm_all[idx_lower_tr] += np.transpose(rsms[idx_rsm])[idx_lower_tr]

	rsm_all /= len(all_subjects)


# =============================================================================
# Average the RSMs across subjects (CV == 1)
# =============================================================================
elif args.cv == 1:

	rsm_train = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)
	rsm_test = np.zeros((rsms.shape[1], rsms.shape[1]), dtype=np.float32)

	for s, sub in enumerate(all_subjects):
		idx_rsm = int(np.floor(s / 2))
		if s % 2 == 0:
			if sub == args.cv_subject:
				rsm_test[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
				rsm_test[idx_upper_tr] += np.transpose(
					rsms[idx_rsm])[idx_upper_tr]
			else:
				rsm_train[idx_lower_tr] += rsms[idx_rsm][idx_lower_tr]
				rsm_train[idx_upper_tr] += np.transpose(
					rsms[idx_rsm])[idx_upper_tr]
		else:
			if sub == args.cv_subject:
				rsm_test[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
				rsm_test[idx_lower_tr] += np.transpose(
					rsms[idx_rsm])[idx_lower_tr]
			else:
				rsm_train[idx_upper_tr] += rsms[idx_rsm][idx_upper_tr]
				rsm_train[idx_lower_tr] += np.transpose(
					rsms[idx_rsm])[idx_lower_tr]

	rsm_train /= len(all_subjects) - 1


# =============================================================================
# Save the averaged RSMs
# =============================================================================
save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'rsms', 'imageset-'+
	args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'averaged_rsm_' + args.roi + '_all_subjects.npy'
	np.save(os.path.join(save_dir, file_name), rsm_all)

elif args.cv == 1:
	file_name_train = 'averaged_rsm_' + args.roi + '_cv_subject-' + \
		format(args.cv_subject, '02') + '_train.npy'
	file_name_test = 'averaged_rsm_' + args.roi + '_cv_subject-' + \
		format(args.cv_subject, '02') + '_test.npy'
	np.save(os.path.join(save_dir, file_name_train), rsm_train)
	np.save(os.path.join(save_dir, file_name_test), rsm_test)
