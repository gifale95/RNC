"""Evaluate the multivariate RNC controlling images effect on the in vivo fMRI
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
from scipy.stats import pearsonr as corr
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
img_cond = np.unique(stim_presentation_order[0])

rsms = {}
rsms[args.rois[0]] = np.zeros((len(args.subjects), len(img_cond), len(img_cond)))
rsms[args.rois[1]] = np.zeros((len(args.subjects), len(img_cond), len(img_cond)))

for s in tqdm(range(len(args.subjects))):

	for i1 in range(len(img_cond)):

		idx_1 = np.isin(stim_presentation_order[s], img_cond[i1])

		for i2 in range(i1):

			idx_2 = np.isin(stim_presentation_order[s], img_cond[i2])

			# V1 RSM
			rsms[args.rois[0]][s,i1,i2] = corr(
				np.mean(betas[args.rois[0]][s][idx_1], 0),
				np.mean(betas[args.rois[0]][s][idx_2], 0))[0]
			rsms[args.rois[0]][s,i2,i1] = rsms[args.rois[0]][s,i1,i2]

			# V4 RSMs
			rsms[args.rois[1]][s,i1,i2] = corr(
				np.mean(betas[args.rois[1]][s][idx_1], 0),
				np.mean(betas[args.rois[1]][s][idx_2], 0))[0]
			rsms[args.rois[1]][s,i2,i1] = rsms[args.rois[1]][s,i1,i2]


# =============================================================================
# Compute the multivariate control scores
# =============================================================================
# Alignment images
rsa_alignemnt_scores = []
tril_idx = np.tril_indices(50, -1)
for s in range(len(args.subjects)):
	rsm_1 = rsms[args.rois[0]][s,:50,:50]
	rsm_2 = rsms[args.rois[1]][s,:50,:50]
	rsm_1 = rsm_1[tril_idx]
	rsm_2 = rsm_2[tril_idx]
	rsa_alignemnt_scores.append(corr(rsm_1, rsm_2)[0])
rsa_alignemnt_scores = np.asarray(rsa_alignemnt_scores)

# Disentanglement images
rsa_disentanglement_scores = []
for s in range(len(args.subjects)):
	rsm_1 = rsms[args.rois[0]][s,50:100,50:100]
	rsm_2 = rsms[args.rois[1]][s,50:100,50:100]
	rsm_1 = rsm_1[tril_idx]
	rsm_2 = rsm_2[tril_idx]
	rsa_disentanglement_scores.append(corr(rsm_1, rsm_2)[0])
rsa_disentanglement_scores = np.asarray(rsa_disentanglement_scores)

# Baseline images
rsa_baseline_scores = []
for s in range(len(args.subjects)):
	rsm_1 = rsms[args.rois[0]][s,100:,100:]
	rsm_2 = rsms[args.rois[1]][s,100:,100:]
	rsm_1 = rsm_1[tril_idx]
	rsm_2 = rsm_2[tril_idx]
	rsa_baseline_scores.append(corr(rsm_1, rsm_2)[0])
rsa_baseline_scores = np.asarray(rsa_baseline_scores)

# Alignment minus baseline
rsa_alignment_minus_baseline = rsa_alignemnt_scores - rsa_baseline_scores

# Disentanglement minus baseline
rsa_disentanglement_minus_baseline = rsa_disentanglement_scores - \
	rsa_baseline_scores


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
# CI arrays of shape: (CI percentiles)
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
# Compute the within-subject significance
# =============================================================================
# Create the permutation-based null distributions
rsa_alignemnt_null_dist = np.zeros((len(args.subjects), args.n_iter),
	dtype=np.float32)
rsa_disentanglement_null_dist = np.zeros((len(args.subjects), args.n_iter),
	dtype=np.float32)
rsa_baseline_null_dist = np.zeros((len(args.subjects), args.n_iter),
	dtype=np.float32)
# Loop across iterations and subjects
for i in tqdm(range(args.n_iter)):
	for s, sub in enumerate(args.subjects):
		# Shuffle the RSMs conditions
		idx_1 = np.arange(len(rsms[args.rois[0]][s]))
		np.random.shuffle(idx_1)
		idx_2 = np.arange(len(rsms[args.rois[0]][s]))
		np.random.shuffle(idx_2)
		# Alignment images
		rsm_1 = rsms[args.rois[0]][s]
		rsm_1 = rsm_1[idx_1]
		rsm_1 = rsm_1[:,idx_1]
		rsm_1 = rsm_1[:50,:50]
		rsm_1 = rsm_1[tril_idx]
		rsm_2 = rsms[args.rois[1]][s]
		rsm_2 = rsm_2[idx_2]
		rsm_2 = rsm_2[:,idx_2]
		rsm_2 = rsm_2[:50,:50]
		rsm_2 = rsm_2[tril_idx]
		rsa_alignemnt_null_dist[s,i] = corr(rsm_1, rsm_2)[0]
		# Disentanglement images
		rsm_1 = rsms[args.rois[0]][s]
		rsm_1 = rsm_1[idx_1]
		rsm_1 = rsm_1[:,idx_1]
		rsm_1 = rsm_1[50:100,50:100]
		rsm_1 = rsm_1[tril_idx]
		rsm_2 = rsms[args.rois[1]][s]
		rsm_2 = rsm_2[idx_2]
		rsm_2 = rsm_2[:,idx_2]
		rsm_2 = rsm_2[50:100,50:100]
		rsm_2 = rsm_2[tril_idx]
		rsa_disentanglement_null_dist[s,i] = corr(rsm_1, rsm_2)[0]
		# Baseline images
		rsm_1 = rsms[args.rois[0]][s]
		rsm_1 = rsm_1[idx_1]
		rsm_1 = rsm_1[:,idx_1]
		rsm_1 = rsm_1[100:,100:]
		rsm_1 = rsm_1[tril_idx]
		rsm_2 = rsms[args.rois[1]][s]
		rsm_2 = rsm_2[idx_2]
		rsm_2 = rsm_2[:,idx_2]
		rsm_2 = rsm_2[100:,100:]
		rsm_2 = rsm_2[tril_idx]
		rsa_baseline_null_dist[s,i] = corr(rsm_1, rsm_2)[0]
# Store the difference between controlling and and baseline RSA scores
rsa_alignment_minus_baseline_null_dist = rsa_alignemnt_null_dist - \
	rsa_baseline_null_dist
rsa_disentanglement_minus_baseline_null_dist = rsa_disentanglement_null_dist - \
	rsa_baseline_null_dist

# Compute the within-subject p-values
rsa_alignment_within_subject_pval = np.zeros((len(args.subjects)),
	dtype=np.float32)
rsa_disentanglement_within_subject_pval = np.zeros((len(args.subjects)),
	dtype=np.float32)
# Compute the p-values
for s, sub in enumerate(args.subjects):
	# Alignment
	idx = sum(rsa_alignment_minus_baseline_null_dist[s] > \
		rsa_alignment_minus_baseline[s])
	rsa_alignment_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
	# Disentanglement
	idx = sum(rsa_disentanglement_minus_baseline_null_dist[s] < \
		rsa_disentanglement_minus_baseline[s])
	rsa_disentanglement_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1)

# Benjamini/Hochberg correct the within-subject alphas over 2 comparisons
n_control_conditions = 2
# Empty significance variables
rsa_alignment_within_subject_sig = np.zeros((len(args.subjects)))
rsa_disentanglement_within_subject_sig = np.zeros((len(args.subjects)))
# Loop across subjects
for s in range(len(args.subjects)):
	# Append the within-subject p-values across the 2 comparisons
	pvals = np.zeros((n_control_conditions))
	pvals[0] = rsa_alignment_within_subject_pval[s]
	pvals[1] = rsa_disentanglement_within_subject_pval[s]
	# Correct for multiple comparisons
	sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
	# Store the significance scores
	rsa_alignment_within_subject_sig[s] = sig[0]
	rsa_disentanglement_within_subject_sig[s] = sig[1]


# =============================================================================
# Compute the between-subject significance
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

n = len(args.subjects) # Total number of subjects
p = 0.05 # probability of success in each trial

# Alignment
k = sum(rsa_alignment_within_subject_sig) # Number of significant subjects
# We use "k-1" because otherwise we would get the probability of observing
# k+1 or more significant results by chance
rsa_alignment_between_subject_pval = 1 - binom.cdf(k-1, n, p)

# l1l2
k = sum(rsa_disentanglement_within_subject_sig)
rsa_disentanglement_between_subject_pval = 1 - binom.cdf(k-1, n, p)


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
significance['rsa_alignment_within_subject_pval'] = \
	rsa_alignment_within_subject_pval
significance['rsa_disentanglement_within_subject_pval'] = \
	rsa_disentanglement_within_subject_pval
significance['rsa_alignment_within_subject_sig'] = \
	rsa_alignment_within_subject_sig
significance['rsa_disentanglement_within_subject_sig'] = \
	rsa_disentanglement_within_subject_sig
significance['rsa_alignment_between_subject_pval'] = \
	rsa_alignment_between_subject_pval
significance['rsa_disentanglement_between_subject_pval'] = \
	rsa_disentanglement_between_subject_pval

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
