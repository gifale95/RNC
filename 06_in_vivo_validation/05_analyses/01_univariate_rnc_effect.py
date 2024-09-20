"""Evaluate the univariate RNC controlling images effect on the in vivo fMRI
data.

This code is available at:
https://github.com/gifale95/RNC/06_in_vivo_validation/05_analyses/01_univariate_rnc_effect.py

Parameters
----------
subjects : list
	List of used subjects.
rois : list
	List of used ROIs.
zscored_data : int
	Whether evaluate the controlling images effec on zscored data [1] or
	non-zscored data [0].
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
from sklearn.utils import resample
from tqdm import tqdm
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--rois', default=['V1', 'V4'], type=list)
parser.add_argument('--zscored_data', type=int, default=1)
parser.add_argument('--ncsnr_threshold', type=float, default=0.4)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()

print('>>> Univariate RNC effect <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Set random seeds to make results reproducible
# =============================================================================
# Random seeds
seed = 202020
seed = int(seed)
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the fMRI betas
# =============================================================================
stim_presentation_order = []
betas = {}
vox_num_all = {}
vox_num_kept = {}

for r in args.rois:

	bet = []

	for s in args.subjects:

		data_dir = os.path.join(args.project_dir, 'in_vivo_validation',
			'in_vivo_fmri_dataset', 'prepared_betas',
			'univariate_rnc_experiment', 'sub-'+format(s, '02'),
			'prepared_betas.npy')
		data = np.load(data_dir, allow_pickle=True).item()
		if r == args.rois[0]:
			stim_presentation_order.append(data['stim_presentation_order'])

		# Data and voxel selection
		vox_num_all['s'+str(s)+'_'+r] = data['betas'][r].shape[1]
		best_voxels = data['ncsnr'][r] > args.ncsnr_threshold
		vox_num_kept['s'+str(s)+'_'+r] = sum(best_voxels)
		if args.zscored_data == 0:
			bet.append(data['betas'][r][:,best_voxels])
		elif args.zscored_data == 1:
			bet.append(data['betas_zscored'][r][:,best_voxels])

	betas[r] = bet
	del bet


# =============================================================================
# Compute the univariate control scores
# =============================================================================
h1h2_scores = {}
l1l2_scores = {}
h1l2_scores = {}
l1h2_scores = {}
baseline_v1 = []
baseline_v4 = []

for r in args.rois:

	h1h2 = []
	l1l2 = []
	h1l2 = []
	l1h2 = []

	for s, sub in enumerate(args.subjects):

		# Images with high univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(1, 26))
		h1h2.append(np.mean(betas[r][s][img_cond]))
		# Images with low univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(76, 101))
		l1l2.append(np.mean(betas[r][s][img_cond]))
		# Images with high univariate responses for V1 and low univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(26, 51))
		h1l2.append(np.mean(betas[r][s][img_cond]))
		# Images with low univariate responses for V1 and high univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(51, 76))
		l1h2.append(np.mean(betas[r][s][img_cond]))
		# Baseline images
		if r == 'V1':
			img_cond = np.isin(stim_presentation_order[s], np.arange(101, 126))
			baseline_v1.append(np.mean(betas[r][s][img_cond]))
		elif r == 'V4':
			img_cond = np.isin(stim_presentation_order[s], np.arange(126, 151))
			baseline_v4.append(np.mean(betas[r][s][img_cond]))

	h1h2_scores[r] = np.asarray(h1h2)
	l1l2_scores[r] = np.asarray(l1l2)
	h1l2_scores[r] = np.asarray(h1l2)
	l1h2_scores[r] = np.asarray(l1h2)
	del h1h2, l1l2, h1l2, l1h2

baseline_v1 = np.asarray(baseline_v1)
baseline_v4 = np.asarray(baseline_v4)


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
h1h2_ci = {}
l1l2_ci = {}
h1l2_ci = {}
l1h2_ci = {}
baseline_v1_ci = []
baseline_v4_ci = []

for r in args.rois:

	# CI arrays of shape:(CI percentiles)
	h1h2_ci_part = np.zeros((2))
	l1l2_ci_part = np.zeros((2))
	h1l2_ci_part = np.zeros((2))
	l1h2_ci_part = np.zeros((2))
	baseline_ci_part = np.zeros((2))

	# Empty CI distribution arrays
	h1h2_dist = np.zeros((args.n_iter))
	l1l2_dist = np.zeros((args.n_iter))
	h1l2_dist = np.zeros((args.n_iter))
	l1h2_dist = np.zeros((args.n_iter))
	baseline_dist = np.zeros((args.n_iter))

	# Compute the CI distributions
	for i in tqdm(range(args.n_iter)):
		idx_resample = resample(np.arange(len(args.subjects)))
		h1h2_dist[i] = np.mean(h1h2_scores[r][idx_resample])
		l1l2_dist[i] = np.mean(l1l2_scores[r][idx_resample])
		h1l2_dist[i] = np.mean(h1l2_scores[r][idx_resample])
		l1h2_dist[i] = np.mean(l1h2_scores[r][idx_resample])
		if r == 'V1':
			baseline_dist[i] = np.mean(
				baseline_v1[idx_resample])
		elif r == 'V4':
			baseline_dist[i] = np.mean(
				baseline_v4[idx_resample])

	# Get the 5th and 95th CI distributions percentiles
	h1h2_ci_part[0] = np.percentile(h1h2_dist, 2.5)
	h1h2_ci_part[1] = np.percentile(h1h2_dist, 97.5)
	l1l2_ci_part[0] = np.percentile(l1l2_dist, 2.5)
	l1l2_ci_part[1] = np.percentile(l1l2_dist, 97.5)
	h1l2_ci_part[0] = np.percentile(h1l2_dist, 2.5)
	h1l2_ci_part[1] = np.percentile(h1l2_dist, 97.5)
	l1h2_ci_part[0] = np.percentile(l1h2_dist, 2.5)
	l1h2_ci_part[1] = np.percentile(l1h2_dist, 97.5)
	baseline_ci_part[0] = np.percentile(baseline_dist, 2.5)
	baseline_ci_part[1] = np.percentile(baseline_dist, 97.5)

	# Store the CIs
	h1h2_ci[r] = h1h2_ci_part
	l1l2_ci[r] = l1l2_ci_part
	h1l2_ci[r] = h1l2_ci_part
	l1h2_ci[r] = l1h2_ci_part
	if r == 'V1':
		baseline_v1_ci.append(baseline_ci_part)
	elif r == 'V4':
		baseline_v4_ci.append(baseline_ci_part)
	del h1h2_ci_part, l1l2_ci_part, h1l2_ci_part, l1h2_ci_part, \
		baseline_ci_part


# =============================================================================
# Compute the significance
# =============================================================================
h1h2_pval = {}
l1l2_pval = {}
h1l2_pval = {}
l1h2_pval = {}
h1h2_pval_corrected = {}
l1l2_pval_corrected = {}
h1l2_pval_corrected = {}
l1h2_pval_corrected = {}
h1h2_sig = {}
l1l2_sig = {}
h1l2_sig = {}
l1h2_sig = {}

for r in args.rois:

	# Compute significance using a paired samples t-test
	if r == 'V1':
		baseline_dist_scores = baseline_v1
		h1h2_pval_part = ttest_rel(h1h2_scores[r], baseline_dist_scores,
			alternative='greater')[1]
		l1l2_pval_part = ttest_rel(l1l2_scores[r], baseline_dist_scores,
			alternative='less')[1]
		h1l2_pval_part = ttest_rel(h1l2_scores[r], baseline_dist_scores,
			alternative='greater')[1]
		l1h2_pval_part = ttest_rel(l1h2_scores[r], baseline_dist_scores,
			alternative='less')[1]
	elif r == 'V4':
		baseline_dist_scores = baseline_v4
		h1h2_pval_part = ttest_rel(h1h2_scores[r], baseline_dist_scores,
			alternative='greater')[1]
		l1l2_pval_part = ttest_rel(l1l2_scores[r], baseline_dist_scores,
			alternative='less')[1]
		h1l2_pval_part = ttest_rel(h1l2_scores[r], baseline_dist_scores,
			alternative='less')[1]
		l1h2_pval_part = ttest_rel(l1h2_scores[r], baseline_dist_scores,
			alternative='greater')[1]

	# Append all p-values together
	pval_all = []
	pval_all.append(h1h2_pval_part)
	pval_all.append(l1l2_pval_part)
	pval_all.append(h1l2_pval_part)
	pval_all.append(l1h2_pval_part)
	pval_all = np.asarray(pval_all)
	pval_all_shape = pval_all.shape
	pval_all = np.reshape(pval_all, -1)

	# Correct for multiple comparisons
	sig, pval_corrected, _, _ = multipletests(pval_all, 0.05, 'fdr_bh')
	sig = np.reshape(sig, pval_all_shape)
	pval_corrected = np.reshape(pval_corrected, pval_all_shape)

	# Store the results
	h1h2_pval[r] = h1h2_pval_part
	l1l2_pval[r] = l1l2_pval_part
	h1l2_pval[r] = h1l2_pval_part
	l1h2_pval[r] = l1h2_pval_part
	h1h2_pval_corrected[r] = pval_corrected[0]
	l1l2_pval_corrected[r] = pval_corrected[1]
	h1l2_pval_corrected[r] = pval_corrected[2]
	l1h2_pval_corrected[r] = pval_corrected[3]
	h1h2_sig[r] = sig[0]
	l1l2_sig[r] = sig[1]
	h1l2_sig[r] = sig[2]
	l1h2_sig[r] = sig[3]


# =============================================================================
# Save the results
# =============================================================================
# Store the neural control scores
control_scores = {}
control_scores['h1h2_scores'] = h1h2_scores
control_scores['l1l2_scores'] = l1l2_scores
control_scores['h1l2_scores'] = h1l2_scores
control_scores['l1h2_scores'] = l1h2_scores
control_scores['baseline_v1'] = baseline_v1
control_scores['baseline_v4'] = baseline_v4

# Store the confidence intervals
confidence_intervals = {}
confidence_intervals['h1h2_ci'] = h1h2_ci
confidence_intervals['l1l2_ci'] = l1l2_ci
confidence_intervals['h1l2_ci'] = h1l2_ci
confidence_intervals['l1h2_ci'] = l1h2_ci
confidence_intervals['baseline_v1_ci'] = baseline_v1_ci
confidence_intervals['baseline_v4_ci'] = baseline_v4_ci

# Store the significance
significance = {}
significance['h1h2_pval'] = h1h2_pval
significance['l1l2_pval'] = l1l2_pval
significance['h1l2_pval'] = h1l2_pval
significance['l1h2_pval'] = l1h2_pval
significance['h1h2_pval_corrected'] = h1h2_pval_corrected
significance['l1l2_pval_corrected'] = l1l2_pval_corrected
significance['h1l2_pval_corrected'] = h1l2_pval_corrected
significance['l1h2_pval_corrected'] = l1h2_pval_corrected
significance['h1h2_sig'] = h1h2_sig
significance['l1l2_sig'] = l1l2_sig
significance['h1l2_sig'] = h1l2_sig
significance['l1h2_sig'] = l1h2_sig

results = {
	'betas': betas,
	'stim_presentation_order': stim_presentation_order,
	'control_scores': control_scores,
	'confidence_intervals': confidence_intervals,
	'significance': significance,
	'vox_num_all': vox_num_all,
	'vox_num_kept': vox_num_kept
	}

save_dir = os.path.join(args.project_dir, 'in_vivo_validation',
	'univariate_rnc_experiment')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'univariate_rnc_experiment_results.npy'

np.save(os.path.join(save_dir, file_name), results)
