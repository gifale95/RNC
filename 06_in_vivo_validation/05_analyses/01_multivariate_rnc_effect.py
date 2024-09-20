"""Evaluate the multivariate RNC controlling images effect on the in vivo fMRI
data.

This code is available at:
https://github.com/gifale95/RNC/06_in_vivo_validation/05_analyses/01_multivariate_rnc_effect.py

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
from scipy.stats import pearsonr as corr
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

print('>>> Multivariate RNC effect <<<')
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
			'multivariate_rnc_experiment', 'sub-'+format(s, '02'),
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
# Create the RSMs
# =============================================================================
# Alignment images
img_cond = np.arange(1, 51) # Alignment image indices
rsms_align = {}
for key, val in betas.items():
	rsm = np.ones((len(val), len(img_cond), len(img_cond)))
	for s in range(len(val)):
		for i1 in range(len(img_cond)):
			idx_1 = np.isin(stim_presentation_order[s], img_cond[i1])
			for i2 in range(i1):
				idx_2 = np.isin(stim_presentation_order[s], img_cond[i2])
				rsm[s,i1,i2] = corr(
					np.mean(val[s][idx_1], 0), np.mean(val[s][idx_2], 0))[0]
				rsm[s,i2,i1] = rsm[s,i1,i2]
	rsms_align[key] = rsm
	del rsm

# Disentanglement images
img_cond = np.arange(51, 101) # Disentanglement image indices
rsms_disentangle = {}
for key, val in betas.items():
	rsm = np.ones((len(val), len(img_cond), len(img_cond)))
	for s in range(len(val)):
		for i1 in range(len(img_cond)):
			idx_1 = np.isin(stim_presentation_order[s], img_cond[i1])
			for i2 in range(i1):
				idx_2 = np.isin(stim_presentation_order[s], img_cond[i2])
				rsm[s,i1,i2] = corr(
					np.mean(val[s][idx_1], 0), np.mean(val[s][idx_2], 0))[0]
				rsm[s,i2,i1] = rsm[s,i1,i2]
	rsms_disentangle[key] = rsm
	del rsm

# Baseline images
img_cond = np.arange(101, 151) # Baseline image indices
rsms_baseline = {}
for key, val in betas.items():
	rsm = np.ones((len(val), len(img_cond), len(img_cond)))
	for s in range(len(val)):
		for i1 in range(len(img_cond)):
			idx_1 = np.isin(stim_presentation_order[s], img_cond[i1])
			for i2 in range(i1):
				idx_2 = np.isin(stim_presentation_order[s], img_cond[i2])
				rsm[s,i1,i2] = corr(
					np.mean(val[s][idx_1], 0), np.mean(val[s][idx_2], 0))[0]
				rsm[s,i2,i1] = rsm[s,i1,i2]
	rsms_baseline[key] = rsm
	del rsm


# =============================================================================
# Compute the multivariate control scores
# =============================================================================
# Alignment images
rsa_alignemnt_scores = []
idx = np.tril_indices(len(img_cond), -1)
for s in range(len(args.subjects)):
	rsm_1 = rsms_align[args.rois[0]][s]
	rsm_2 = rsms_align[args.rois[1]][s]
	rsm_1 = rsm_1[idx]
	rsm_2 = rsm_2[idx]
	rsa_alignemnt_scores.append(corr(rsm_1, rsm_2)[0])
rsa_alignemnt_scores = np.asarray(rsa_alignemnt_scores)

# Disentanglement images
rsa_disentanglement_scores = []
for s in range(len(args.subjects)):
	rsm_1 = rsms_disentangle[args.rois[0]][s]
	rsm_2 = rsms_disentangle[args.rois[1]][s]
	rsm_1 = rsm_1[idx]
	rsm_2 = rsm_2[idx]
	rsa_disentanglement_scores.append(corr(rsm_1, rsm_2)[0])
rsa_disentanglement_scores = np.asarray(rsa_disentanglement_scores)

# Baseline images
rsa_baseline_scores = []
for s in range(len(args.subjects)):
	rsm_1 = rsms_baseline[args.rois[0]][s]
	rsm_2 = rsms_baseline[args.rois[1]][s]
	rsm_1 = rsm_1[idx]
	rsm_2 = rsm_2[idx]
	rsa_baseline_scores.append(corr(rsm_1, rsm_2)[0])
rsa_baseline_scores = np.asarray(rsa_baseline_scores)


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
# CI arrays of shape:(CI percentiles)
rsa_alignment_ci = np.zeros((2))
rsa_disentanglement_ci = np.zeros((2))
rsa_baseline_ci = np.zeros((2))

# Empty CI distribution arrays
rsa_alignment_dist = np.zeros((args.n_iter))
rsa_disentanglement_dist = np.zeros((args.n_iter))
rsa_baseline_dist = np.zeros((args.n_iter))

# Compute the CI distributions
for i in tqdm(range(args.n_iter)):
	idx_resample = resample(np.arange(len(args.subjects)))
	rsa_alignment_dist[i] = np.mean(rsa_alignemnt_scores[idx_resample])
	rsa_disentanglement_dist[i] = np.mean(
		rsa_disentanglement_scores[idx_resample])
	rsa_baseline_dist[i] = np.mean(rsa_baseline_scores[idx_resample])

# Get the 5th and 95th CI distributions percentiles
rsa_alignment_ci[0] = np.percentile(rsa_alignment_dist, 2.5)
rsa_alignment_ci[1] = np.percentile(rsa_alignment_dist, 97.5)
rsa_disentanglement_ci[0] = np.percentile(rsa_disentanglement_dist, 2.5)
rsa_disentanglement_ci[1] = np.percentile(rsa_disentanglement_dist, 97.5)
rsa_baseline_ci[0] = np.percentile(rsa_baseline_dist, 2.5)
rsa_baseline_ci[1] = np.percentile(rsa_baseline_dist, 97.5)


# =============================================================================
# Compute the significance
# =============================================================================
# Compute significance using a paired samples t-test
rsa_alignment_pval = ttest_rel(rsa_alignemnt_scores, rsa_baseline_scores,
	alternative='greater')[1]
rsa_disentanglement_pval = ttest_rel(rsa_disentanglement_scores,
	rsa_baseline_scores, alternative='less')[1]

# Correct for multiple comparisons
p_values_all = np.append(rsa_alignment_pval, rsa_disentanglement_pval)
significance, p_values_corrected, _, _ = multipletests(p_values_all, 0.05,
	'fdr_bh')

# Store the significance and corrected p-values
rsa_alignment_pval_corrected = p_values_corrected[0]
rsa_disentanglement_pval_corrected = p_values_corrected[1]
rsa_alignment_sig = significance[0]
rsa_disentanglement_sig = significance[1]


# =============================================================================
# Save the results
# =============================================================================
# Store the neural control scores
control_scores = {}
control_scores['rsa_alignemnt_scores'] = rsa_alignemnt_scores
control_scores['rsa_disentanglement_scores'] = rsa_disentanglement_scores
control_scores['rsa_baseline_scores'] = rsa_baseline_scores

# Store the confidence intervals
confidence_intervals = {}
confidence_intervals['rsa_alignment_ci'] = rsa_alignment_ci
confidence_intervals['rsa_disentanglement_ci'] = rsa_disentanglement_ci
confidence_intervals['rsa_baseline_ci'] = rsa_baseline_ci

# Store the significance
significance = {}
significance['rsa_alignment_pval'] = rsa_alignment_pval
significance['rsa_disentanglement_pval'] = rsa_disentanglement_pval
significance['rsa_alignment_pval_corrected'] = rsa_alignment_pval_corrected
significance['rsa_disentanglement_pval_corrected'] = \
	rsa_disentanglement_pval_corrected
significance['rsa_alignment_sig'] = rsa_alignment_sig
significance['rsa_disentanglement_sig'] = rsa_disentanglement_sig

results = {
	'control_scores': control_scores,
	'confidence_intervals': confidence_intervals,
	'significance': significance,
	'vox_num_all': vox_num_all,
	'vox_num_kept': vox_num_kept
	}

save_dir = os.path.join(args.project_dir, 'in_vivo_validation',
	'multivariate_rnc_experiment')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'multivariate_rnc_experiment_results.npy'

np.save(os.path.join(save_dir, file_name), results)
