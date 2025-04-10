"""Evaluate the univariate RNC controlling images effect on the in vivo fMRI
data.

This code is available at:
https://github.com/gifale95/RNC

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
	Amount of iterations for the permutation stats.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
from sklearn.utils import resample
from tqdm import tqdm
from scipy.stats import binom
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
parser.add_argument('--project_dir', default='../relational_neural_control_old/', type=str)
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
h1h2_minus_baseline = {}
l1l2_minus_baseline = {}
h1l2_minus_baseline = {}
l1h2_minus_baseline = {}
baseline_v1 = np.zeros((len(args.subjects)))
baseline_v4 = np.zeros((len(args.subjects)))

for r in args.rois:

	h1h2 = np.zeros((len(args.subjects)))
	l1l2 = np.zeros((len(args.subjects)))
	h1l2 = np.zeros((len(args.subjects)))
	l1h2 = np.zeros((len(args.subjects)))

	for s, sub in enumerate(args.subjects):

		# Images with high univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(1, 26))
		h1h2[s] = np.mean(betas[r][s][img_cond])
		# Images with low univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(76, 101))
		l1l2[s] = np.mean(betas[r][s][img_cond])
		# Images with high univariate responses for V1 and low univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(26, 51))
		h1l2[s] = np.mean(betas[r][s][img_cond])
		# Images with low univariate responses for V1 and high univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(51, 76))
		l1h2[s] = np.mean(betas[r][s][img_cond])
		# Baseline images
		if r == 'V1':
			img_cond = np.isin(stim_presentation_order[s], np.arange(101, 126))
			baseline_v1[s] = np.mean(betas[r][s][img_cond])
		elif r == 'V4':
			img_cond = np.isin(stim_presentation_order[s], np.arange(126, 151))
			baseline_v4[s] = np.mean(betas[r][s][img_cond])

	# Store the univariate responses
	h1h2_scores[r] = h1h2
	l1l2_scores[r] = l1l2
	h1l2_scores[r] = h1l2
	l1h2_scores[r] = l1h2

	# Store the difference between controlling and and baseline image univariate
	# responses
	if r == 'V1':
		h1h2_minus_baseline[r] = h1h2 - baseline_v1
		l1l2_minus_baseline[r] = l1l2 - baseline_v1
		h1l2_minus_baseline[r] = h1l2 - baseline_v1
		l1h2_minus_baseline[r] = l1h2 - baseline_v1
	elif r == 'V4':
		h1h2_minus_baseline[r] = h1h2 - baseline_v4
		l1l2_minus_baseline[r] = l1l2 - baseline_v4
		h1l2_minus_baseline[r] = h1l2 - baseline_v4
		l1h2_minus_baseline[r] = l1h2 - baseline_v4
	del h1h2, l1l2, h1l2, l1h2


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

	# CI arrays of shape: (CI percentiles)
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
# Compute the within-subject significance
# =============================================================================
# Create the permutation-based null distributions
h1h2_scores_null_dist = {}
l1l2_scores_null_dist = {}
h1l2_scores_null_dist = {}
l1h2_scores_null_dist = {}
h1h2_minus_baseline_null_dist = {}
l1l2_minus_baseline_null_dist = {}
h1l2_minus_baseline_null_dist = {}
l1h2_minus_baseline_null_dist = {}
baseline_v1_null_dist = np.zeros((len(args.subjects), args.n_iter),
	dtype=np.float32)
baseline_v4_null_dist = np.zeros((len(args.subjects), args.n_iter),
	dtype=np.float32)
h1h2_r1 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
l1l2_r1 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
h1l2_r1 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
l1h2_r1 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
h1h2_r2 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
l1l2_r2 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
h1l2_r2 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
l1h2_r2 = np.zeros((len(args.subjects), args.n_iter), dtype=np.float32)
# Loop across iterations and subjects
for i in tqdm(range(args.n_iter)):
	for s, sub in enumerate(args.subjects):
		# Shuffle the betas across samples
		idx = np.arange(len(betas[r][s]))
		np.random.shuffle(idx)
		# Images with high univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(1, 26))
		h1h2_r1[s,i] = np.mean(betas[args.rois[0]][s][idx][img_cond])
		h1h2_r2[s,i] = np.mean(betas[args.rois[1]][s][idx][img_cond])
		# Images with low univariate responses for both ROIs
		img_cond = np.isin(stim_presentation_order[s], np.arange(76, 101))
		l1l2_r1[s,i] = np.mean(betas[args.rois[0]][s][idx][img_cond])
		l1l2_r2[s,i] = np.mean(betas[args.rois[1]][s][idx][img_cond])
		# Images with high univariate responses for V1 and low univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(26, 51))
		h1l2_r1[s,i] = np.mean(betas[args.rois[0]][s][idx][img_cond])
		h1l2_r2[s,i] = np.mean(betas[args.rois[1]][s][idx][img_cond])
		# Images with low univariate responses for V1 and high univariate
		# responses for V4
		img_cond = np.isin(stim_presentation_order[s], np.arange(51, 76))
		l1h2_r1[s,i] = np.mean(betas[args.rois[0]][s][idx][img_cond])
		l1h2_r2[s,i] = np.mean(betas[args.rois[1]][s][idx][img_cond])
		# Baseline images
		img_cond = np.isin(stim_presentation_order[s],
			np.arange(101, 126))
		baseline_v1_null_dist[s,i] = np.mean(
			betas[args.rois[0]][s][idx][img_cond])
		img_cond = np.isin(stim_presentation_order[s],
			np.arange(126, 151))
		baseline_v4_null_dist[s,i] = np.mean(
			betas[args.rois[1]][s][idx][img_cond])
# Store the univariate responses
h1h2_scores_null_dist[args.rois[0]] = h1h2_r1
l1l2_scores_null_dist[args.rois[0]] = l1l2_r1
h1l2_scores_null_dist[args.rois[0]] = h1l2_r1
l1h2_scores_null_dist[args.rois[0]] = l1h2_r1
h1h2_scores_null_dist[args.rois[1]] = h1h2_r2
l1l2_scores_null_dist[args.rois[1]] = l1l2_r2
h1l2_scores_null_dist[args.rois[1]] = h1l2_r2
l1h2_scores_null_dist[args.rois[1]] = l1h2_r2
# Store the difference between controlling and and baseline image univariate
# responses
h1h2_minus_baseline_null_dist[args.rois[0]] = h1h2_r1 - baseline_v1_null_dist
l1l2_minus_baseline_null_dist[args.rois[0]] = l1l2_r1 - baseline_v1_null_dist
h1l2_minus_baseline_null_dist[args.rois[0]] = h1l2_r1 - baseline_v1_null_dist
l1h2_minus_baseline_null_dist[args.rois[0]] = l1h2_r1 - baseline_v1_null_dist
h1h2_minus_baseline_null_dist[args.rois[1]] = h1h2_r2 - baseline_v4_null_dist
l1l2_minus_baseline_null_dist[args.rois[1]] = l1l2_r2 - baseline_v4_null_dist
h1l2_minus_baseline_null_dist[args.rois[1]] = h1l2_r2 - baseline_v4_null_dist
l1h2_minus_baseline_null_dist[args.rois[1]] = l1h2_r2 - baseline_v4_null_dist

# Compute the within-subject p-values
h1h2_within_subject_pval = {}
l1l2_within_subject_pval = {}
h1l2_within_subject_pval = {}
l1h2_within_subject_pval = {}
for r in args.rois:
	h1h2 = np.zeros((len(args.subjects)), dtype=np.float32)
	l1l2 = np.zeros((len(args.subjects)), dtype=np.float32)
	h1l2 = np.zeros((len(args.subjects)), dtype=np.float32)
	l1h2 = np.zeros((len(args.subjects)), dtype=np.float32)
	# Compute the p-values
	for s, sub in enumerate(args.subjects):
		# h1h2
		idx = sum(h1h2_minus_baseline_null_dist[r][s] > \
			h1h2_minus_baseline[r][s])
		h1h2[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
		# l1l2
		idx = sum(l1l2_minus_baseline_null_dist[r][s] < \
			l1l2_minus_baseline[r][s])
		l1l2[s] = (idx + 1) / (args.n_iter + 1)
		if r == 'V1':
			# h1l2
			idx = sum(h1l2_minus_baseline_null_dist[r][s] > \
				h1l2_minus_baseline[r][s])
			h1l2[s] = (idx + 1) / (args.n_iter + 1)
			# l1h2
			idx = sum(l1h2_minus_baseline_null_dist[r][s] < \
				l1h2_minus_baseline[r][s])
			l1h2[s] = (idx + 1) / (args.n_iter + 1)
		if r == 'V4':
			# h1l2
			idx = sum(h1l2_minus_baseline_null_dist[r][s] < \
				h1l2_minus_baseline[r][s])
			h1l2[s] = (idx + 1) / (args.n_iter + 1)
			# l1h2
			idx = sum(l1h2_minus_baseline_null_dist[r][s] > \
				l1h2_minus_baseline[r][s])
			l1h2[s] = (idx + 1) / (args.n_iter + 1)
	# Store the p-values
	h1h2_within_subject_pval[r] = h1h2
	l1l2_within_subject_pval[r] = l1l2
	h1l2_within_subject_pval[r] = h1l2
	l1h2_within_subject_pval[r] = l1h2

# Benjamini/Hochberg correct the within-subject alphas over:
# 4 control conditions × 2 ROIs per condition = 8 comparisons
n_control_conditions = 4
n_rois = 2
# Empty significance variables
h1h2_within_subject_sig = {}
l1l2_within_subject_sig = {}
h1l2_within_subject_sig = {}
l1h2_within_subject_sig = {}
for roi in args.rois:
	h1h2_within_subject_sig[roi] = np.zeros((len(args.subjects)))
	l1l2_within_subject_sig[roi] = np.zeros((len(args.subjects)))
	h1l2_within_subject_sig[roi] = np.zeros((len(args.subjects)))
	l1h2_within_subject_sig[roi] = np.zeros((len(args.subjects)))
# Loop across subjects
for s in range(len(args.subjects)):
	# Append the within-subject p-values across the 8 comparisons
	pvals = np.zeros((n_control_conditions, n_rois))
	for r, roi in enumerate(args.rois):
		pvals[0,r] = h1h2_within_subject_pval[roi][s]
		pvals[1,r] = l1l2_within_subject_pval[roi][s]
		pvals[2,r] = h1l2_within_subject_pval[roi][s]
		pvals[3,r] = l1h2_within_subject_pval[roi][s]
	pvals = pvals.flatten()
	# Correct for multiple comparisons
	sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
	sig = np.reshape(sig, (n_control_conditions, n_rois))
	# Store the significance scores
	for r, roi in enumerate(args.rois):
		h1h2_within_subject_sig[roi][s] = sig[0,r]
		l1l2_within_subject_sig[roi][s] = sig[1,r]
		h1l2_within_subject_sig[roi][s] = sig[2,r]
		l1h2_within_subject_sig[roi][s] = sig[3,r]


# =============================================================================
# Compute the between-subject significance
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

n = len(args.subjects) # Total number of subjects
p = 0.05 # probability of success in each trial

h1h2_between_subject_pval = {}
l1l2_between_subject_pval = {}
h1l2_between_subject_pval = {}
l1h2_between_subject_pval = {}

for r in args.rois:

	# h1h2
	k = sum(h1h2_within_subject_sig[r]) # Number of significant subjects
	# We use "k-1" because otherwise we would get the probability of observing
	# k+1 or more significant results by chance
	h1h2_between_subject_pval[r] = 1 - binom.cdf(k-1, n, p)

	# l1l2
	k = sum(l1l2_within_subject_sig[r])
	l1l2_between_subject_pval[r] = 1 - binom.cdf(k-1, n, p)

	# h1l2
	k = sum(h1l2_within_subject_sig[r])
	h1l2_between_subject_pval[r] = 1 - binom.cdf(k-1, n, p)

	# l1h2
	k = sum(l1h2_within_subject_sig[r])
	l1h2_between_subject_pval[r] = 1 - binom.cdf(k-1, n, p)


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
significance['h1h2_within_subject_pval'] = h1h2_within_subject_pval
significance['l1l2_within_subject_pval'] = l1l2_within_subject_pval
significance['h1l2_within_subject_pval'] = h1l2_within_subject_pval
significance['l1h2_within_subject_pval'] = l1h2_within_subject_pval
significance['h1h2_within_subject_sig'] = h1h2_within_subject_sig
significance['l1l2_within_subject_sig'] = l1l2_within_subject_sig
significance['h1l2_within_subject_sig'] = h1l2_within_subject_sig
significance['l1h2_within_subject_sig'] = l1h2_within_subject_sig
significance['h1h2_between_subject_pval'] = h1h2_between_subject_pval
significance['l1l2_between_subject_pval'] = l1l2_between_subject_pval
significance['h1l2_between_subject_pval'] = h1l2_between_subject_pval
significance['l1h2_between_subject_pval'] = l1h2_between_subject_pval

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
