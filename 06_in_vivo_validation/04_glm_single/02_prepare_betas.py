"""Prepare the fMRI betas for further analyses, and compute the ncsnr.

This code is available at:
https://github.com/gifale95/RNC/06_in_vivo_validation/04_glm_single/02_prepare_betas.py

Parameters
----------
subject : int
	Used subject, out of all 6 collected.
rois : list
	List of used ROIs. Here we care about V1 and V4.
exp_name : str
	Experiment name. Available choices are 'univariate_rnc' and
	'multivariate_rnc'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--rois', default=['V1', 'V4'], type=list)
parser.add_argument('--exp_name', default='univariate_rnc', type=str)
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()

print('>>> Prepare betas <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get experimental design info
# =============================================================================
if args.exp_name == 'univariate_rnc':
	condition_repeats = 6
	runs = 10
elif args.exp_name == 'multivariate_rnc':
	condition_repeats = 8
	runs = 12


# =============================================================================
# Get the stimulus presentation order
# =============================================================================
stim_presentation_order = []

for r in range(runs):

	# Load the functional data events
	if args.exp_name == 'univariate_rnc':
		events_file = os.path.join(args.project_dir, 'in_vivo_validation',
			'in_vivo_fmri_dataset', 'raw', 'sub-0'+str(args.subject), 'ses-01',
			'func', 'sub-0'+str(args.subject)+'_ses-01_task-univariate_run-'+
			format(r+1, '02')+'_events.tsv') # Download the fMRI events from: https://openneuro.org/datasets/ds005503
	elif args.exp_name == 'multivariate_rnc':
		events_file = os.path.join(args.project_dir, 'in_vivo_validation',
			'in_vivo_fmri_dataset', 'raw', 'sub-0'+str(args.subject), 'ses-02',
			'func', 'sub-0'+str(args.subject)+'_ses-02_task-multivariate_run-'+
			format(r+1, '02')+'_events.tsv') # Download the fMRI events from: https://openneuro.org/datasets/ds005503
	events = np.asarray(pd.read_csv(events_file, sep='\t')['stim_id'])
	stim_presentation_order.append(events)

stim_presentation_order = np.concatenate(stim_presentation_order, 0)

# Remove the blank trials
idx_blank = np.where(np.isnan(stim_presentation_order))[0]
stim_presentation_order = np.delete(stim_presentation_order, idx_blank)
stim_presentation_order = stim_presentation_order.astype(np.int32)

# Remove the catch trials (with ID 151)
idx_catch = np.where(stim_presentation_order == 151)[0]
stim_presentation_order = np.delete(stim_presentation_order, idx_catch)


# =============================================================================
# Load and z-score the betas of all ROIs
# =============================================================================
betas = {}
betas_zscored = {}

for roi in args.rois:

	data_dir = os.path.join(args.project_dir, 'in_vivo_validation',
		'in_vivo_fmri_dataset', 'GLMsingle', args.exp_name+'_experiment',
		'sub-'+format(args.subject, '02'), 'roi-'+roi,
		'TYPED_FITHRF_GLMDENOISE_RR.npy')
	data = np.transpose(np.squeeze(np.load(
		data_dir, allow_pickle=True).item()['betasmd']))

	# Remove the catch trials (with ID 151)
	betas[roi] = np.delete(data, idx_catch, 0)
	del data

	# z-score the responses of each voxel within each scan session to eliminate
	# potential (across sessions) non-stationarities and to equalize units
	# across voxels.
	scaler = StandardScaler()
	betas_zscored[roi] = scaler.fit_transform(betas[roi])


# =============================================================================
# Compute the ncsnr on the z-scored betas
# =============================================================================
ncsnr = {}
noise_ceiling = {}

for roi, roi_betas in betas_zscored.items():

	# Reshape the data to (Conditions x Repeats x Voxels)
	conditions = np.unique(stim_presentation_order)
	betas_newshape = np.zeros((len(conditions), condition_repeats,
		roi_betas.shape[1]))
	for c, cond in enumerate(conditions):
		idx_cond = np.where(stim_presentation_order == cond)[0]
		if len(idx_cond) != condition_repeats:
			raise Exception('Wrong image condition repetition amount!')
		betas_newshape[c] = roi_betas[idx_cond]

	# Compute the ncsnr
	std_noise = np.sqrt(np.mean(np.var(betas_newshape, axis=1, ddof=1), 0))
	std_signal = 1 - (std_noise ** 2)
	std_signal[std_signal<0] = 0
	std_signal = np.sqrt(std_signal)
	ncsnr[roi] = std_signal / std_noise


# =============================================================================
# Save the formatted data
# =============================================================================
results = {
	'betas': betas,
	'betas_zscored': betas_zscored,
	'stim_presentation_order': stim_presentation_order,
	'ncsnr': ncsnr
	}

save_dir = os.path.join(args.project_dir, 'in_vivo_validation',
	'in_vivo_fmri_dataset', 'prepared_betas', args.exp_name+'_experiment',
	'sub-'+format(args.subject, '02'))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'prepared_betas'

np.save(os.path.join(save_dir, file_name), results)

