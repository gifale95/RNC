"""This code tests whether the multivariate RNC alignment and disentanglement
scores change as a function of cortical distance between pairs of areas.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
rois : list of str
	List of used ROIs.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_iter : int
	Amount of iterations for the permutation stats.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from itertools import combinations
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()

print('>>> Multivariate RNC cortical distance analysis <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# Pairwise ROI comparisons
# =============================================================================
roi_comb = list(combinations(np.arange(len(args.rois)), 2))

r1 = []
r2 = []
roi_comb_names = []

for c, comb in enumerate(roi_comb):
	r1.append(comb[0])
	r2.append(comb[1])
	roi_comb_names.append(args.rois[comb[0]]+'-'+args.rois[comb[1]])


# =============================================================================
# Load the mutivariate RNC stats (cross-validated)
# =============================================================================
scores_align = np.zeros((len(all_subjects), len(roi_comb_names)))
scores_disentangle = np.zeros((scores_align.shape))
baseline_scores = np.zeros((scores_align.shape))

for r, roi in enumerate(roi_comb_names):

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-1',
		'imageset-'+args.imageset, roi, 'stats.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()

	scores_align[:,r] = data_dict['best_generation_scores_test']['align'][:,-1]
	scores_disentangle[:,r] = data_dict['best_generation_scores_test']\
		['disentangle'][:,-1]
	baseline_scores[:,r] = data_dict['baseline_images_score_test']


# =============================================================================
# Load the multivariate RNC baseline null distributions (cross-validated)
# =============================================================================
null_distribution_test = []

for roi in roi_comb_names:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'baseline',
		'cv-1', 'imageset-'+args.imageset, roi)

	null_distribution_test_all_sub = []

	for s, sub in enumerate(all_subjects):

		file_name = 'baseline_cv_subject-' + format(sub, '02') + '.npy'
		results = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		null_distribution_test_all_sub.append(results['null_distribution_test'])

	null_distribution_test.append(np.asarray(null_distribution_test_all_sub))

null_distribution_test = np.swapaxes(null_distribution_test, 0, 1)


# =============================================================================
# Aggregate the multivariate RNC RSA scores as a function of cortical distance
# =============================================================================
# There are three cortical distances:
# Cortical distance 1: [V1 vs. V2; V2 vs. V3; V3 vs. V4]
# Cortical distance 2: [V1 vs. V3; V2 vs. V4]
# Cortical distance 3: [V1 vs. V4]
cortical_distances = [(0, 3, 5), (1, 4), (2)]

# Sorted scores arrays of shape:
# (Subjects × ROI cortical distances)
sorted_align = np.zeros((len(all_subjects), len(cortical_distances)))
sorted_disentangle = np.zeros((len(all_subjects), len(cortical_distances)))
sorted_baseline = np.zeros((len(all_subjects), len(cortical_distances)))

# Group the multivariate RNC RSA scores as a function of cortical distance
for d, dist in enumerate(cortical_distances):
	if type(dist) == tuple:
		sorted_align[:,d] = np.mean(scores_align[:,dist], 1)
		sorted_disentangle[:,d] = np.mean(scores_disentangle[:,dist], 1)
		sorted_baseline[:,d] = np.mean(baseline_scores[:,dist], 1)
	else:
		sorted_align[:,d] = scores_align[:,dist]
		sorted_disentangle[:,d] = scores_disentangle[:,dist]
		sorted_baseline[:,d] = baseline_scores[:,dist]

# Compute the difference in RSA scores between cortical distances
sorted_align_diff = sorted_align[:,0] - sorted_align[:,1]
sorted_align_diff += sorted_align[:,1] - sorted_align[:,2]
sorted_disentangle_diff = sorted_disentangle[:,0] - sorted_disentangle[:,1]
sorted_disentangle_diff += sorted_disentangle[:,1] - sorted_disentangle[:,2]
sorted_baseline_diff = sorted_baseline[:,0] - sorted_baseline[:,1]
sorted_baseline_diff += sorted_baseline[:,1] - sorted_baseline[:,2]


# =============================================================================
# Compute the 95% confidence intervals
# =============================================================================
# CI arrays of shape:
# (CI percentiles × Cortical distances)
ci_sorted_align = np.zeros((2, len(cortical_distances)))
ci_sorted_disentangle = np.zeros((2, len(cortical_distances)))
ci_sorted_baseline = np.zeros((2, len(cortical_distances)))

# Loop across cortical distances
for d in tqdm(range(len(cortical_distances)), leave=False):

	# Empty CI distribution arrays
	sorted_align_dist = np.zeros((args.n_iter))
	sorted_disentangle_dist = np.zeros((args.n_iter))
	sorted_baseline_dist = np.zeros((args.n_iter))

	# Compute the CI distributions
	for i in range(args.n_iter):
		idx_resample = resample(np.arange(len(all_subjects)))
		sorted_align_dist[i] = np.mean(sorted_align[idx_resample,d])
		sorted_disentangle_dist[i] = np.mean(sorted_disentangle[idx_resample,d])
		sorted_baseline_dist[i] = np.mean(sorted_baseline[idx_resample,d])

	# Get the 5th and 95th CI distributions percentiles
	ci_sorted_align[0,d] = np.percentile(sorted_align_dist, 2.5)
	ci_sorted_align[1,d] = np.percentile(sorted_align_dist, 97.5)
	ci_sorted_disentangle[0,d] = np.percentile(sorted_disentangle_dist, 2.5)
	ci_sorted_disentangle[1,d] = np.percentile(sorted_disentangle_dist, 97.5)
	ci_sorted_baseline[0,d] = np.percentile(sorted_baseline_dist, 2.5)
	ci_sorted_baseline[1,d] = np.percentile(sorted_baseline_dist, 97.5)


# =============================================================================
# Compute the within-subject significance
# =============================================================================
# To perform the within-subject significance we will use the RSA null
# distributions created while computing the multivariate RNC baseline.
# These null distributions contain 1 million scores, each consisting of the
# RSA scores obtained when correlating the RSMs of two ROIs based on a
# different set of 50 image conditions.

# Sorted scores arrays of shape:
# (Subjects × ROI cortical distances × Iterations)
sorted_align_null_dist = np.zeros((len(all_subjects), len(cortical_distances),
	args.n_iter))
sorted_disentangle_null_dist = np.zeros((sorted_align_null_dist.shape))
sorted_baseline_null_dist = np.zeros((sorted_align_null_dist.shape))
# The first N null distribution scores will be used as the alignment
# scores (where N is equal to args.n_iter).
idx_start_align = args.n_iter * 0
idx_end_align = idx_start_align + args.n_iter
# The second N null distribution scores will be used as the disentanglement
# scores (where N is equal to args.n_iter).
idx_start_disentangle = args.n_iter * 1
idx_end_disentangle = idx_start_disentangle + args.n_iter
# The third N null distribution scores will be used as the baseline
# scores (where N is equal to args.n_iter).
idx_start_baseline = args.n_iter * 2
idx_end_baseline = idx_start_baseline + args.n_iter
# Group the multivariate RNC null distribution RSA scores as a function of
# cortical distance, after shuffling the ROIs to which the null distribution
# scores belong to
null_distribution_test = np.array(
	[np.random.permutation(a) for a in null_distribution_test])
for d, dist in enumerate(cortical_distances):
	if type(dist) == tuple:
		sorted_align_null_dist[:,d] = np.mean(null_distribution_test\
			[:,dist,idx_start_align:idx_end_align], 1)
		sorted_disentangle_null_dist[:,d] = np.mean(null_distribution_test\
			[:,dist,idx_start_disentangle:idx_end_disentangle], 1)
		sorted_baseline_null_dist[:,d] = np.mean(null_distribution_test\
			[:,dist,idx_start_baseline:idx_end_baseline], 1)
	else:
		sorted_align_null_dist[:,d] = null_distribution_test\
			[:,dist,idx_start_align:idx_end_align]
		sorted_disentangle_null_dist[:,d] = null_distribution_test\
			[:,dist,idx_start_disentangle:idx_end_disentangle]
		sorted_baseline_null_dist[:,d] = null_distribution_test\
			[:,dist,idx_start_baseline:idx_end_baseline]
# Compute the difference in RSA scores between cortical distances
sorted_align_diff_null_dist = sorted_align_null_dist[:,0] - \
	sorted_align_null_dist[:,1]
sorted_align_diff_null_dist += sorted_align_null_dist[:,1] - \
	sorted_align_null_dist[:,2]
sorted_disentangle_diff_null_dist = sorted_disentangle_null_dist[:,0] - \
	sorted_disentangle_null_dist[:,1]
sorted_disentangle_diff_null_dist += sorted_disentangle_null_dist[:,1] - \
	sorted_disentangle_null_dist[:,2]
sorted_baseline_diff_null_dist = sorted_baseline_null_dist[:,0] - \
	sorted_baseline_null_dist[:,1]
sorted_baseline_diff_null_dist += sorted_baseline_null_dist[:,1] - \
	sorted_baseline_null_dist[:,2]

# Compute the within-subject p-values
sorted_align_within_subject_pval = np.zeros((len(all_subjects)),
	dtype=np.float32)
sorted_disentangle_within_subject_pval = np.zeros((len(all_subjects)),
	dtype=np.float32)
sorted_baseline_within_subject_pval = np.zeros((len(all_subjects)),
	dtype=np.float32)
# Compute the p-values
for s, sub in enumerate(all_subjects):
	# Alignment (test for a decreasing trend)
	idx = sum(sorted_align_diff_null_dist[s] > sorted_align_diff[s])
	sorted_align_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
	# Disentanglement (test for a decreasing trend)
	idx = sum(sorted_disentangle_diff_null_dist[s] > sorted_disentangle_diff[s])
	sorted_disentangle_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1)
	# Baseline (test for a decreasing trend)
	idx = sum(sorted_baseline_diff_null_dist[s] > sorted_baseline_diff[s])
	sorted_baseline_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1)

# Benjamini/Hochberg correct the within-subject alphas over 3 comparisons
n_control_conditions = 3
# Empty significance variables
sorted_align_within_subject_sig = np.zeros((len(all_subjects)))
sorted_disentangle_within_subject_sig = np.zeros((len(all_subjects)))
sorted_baseline_within_subject_sig = np.zeros((len(all_subjects)))
# Loop across subjects
for s in range(len(all_subjects)):
	# Append the within-subject p-values across the 3 comparisons
	pvals = np.zeros((n_control_conditions))
	pvals[0] = sorted_align_within_subject_pval[s]
	pvals[1] = sorted_disentangle_within_subject_pval[s]
	pvals[2] = sorted_baseline_within_subject_pval[s]
	# Correct for multiple comparisons
	sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
	# Store the significance scores
	sorted_align_within_subject_sig[s] = sig[0]
	sorted_disentangle_within_subject_sig[s] = sig[1]
	sorted_baseline_within_subject_sig[s] = sig[2]


# =============================================================================
# Compute the between-subject significance
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

n = len(all_subjects) # Total number of subjects
p = 0.05 # probability of success in each trial

# Alignment
k = sum(sorted_align_within_subject_sig) # Number of significant subjects
# We use "k-1" because otherwise we would get the probability of observing
# k+1 or more significant results by chance
sorted_align_between_subject_pval = 1 - binom.cdf(k-1, n, p)

# Disentanglement
k = sum(sorted_disentangle_within_subject_sig) # Number of significant subjects
sorted_disentangle_between_subject_pval = 1 - binom.cdf(k-1, n, p)

# Baseline
k = sum(sorted_baseline_within_subject_sig) # Number of significant subjects
sorted_baseline_between_subject_pval = 1 - binom.cdf(k-1, n, p)


# =============================================================================
# Save the stats
# =============================================================================
results = {
	'roi_comb': roi_comb,
	'r1': r1,
	'r2': r2,
	'roi_comb_names': roi_comb_names,
	'sorted_align': sorted_align,
	'sorted_disentangle': sorted_disentangle,
	'sorted_baseline': sorted_baseline,
	'ci_sorted_align': ci_sorted_align,
	'ci_sorted_disentangle': ci_sorted_disentangle,
	'ci_sorted_baseline': ci_sorted_baseline,
	'sorted_align_within_subject_pval' : sorted_align_within_subject_pval,
	'sorted_align_within_subject_sig' : sorted_align_within_subject_sig,
	'sorted_align_between_subject_pval' : sorted_align_between_subject_pval,
	'sorted_disentangle_within_subject_pval' : sorted_disentangle_within_subject_pval,
	'sorted_disentangle_within_subject_sig' : sorted_disentangle_within_subject_sig,
	'sorted_disentangle_between_subject_pval' : sorted_disentangle_between_subject_pval,
	'sorted_baseline_within_subject_pval' : sorted_baseline_within_subject_pval,
	'sorted_baseline_within_subject_sig' : sorted_baseline_within_subject_sig,
	'sorted_baseline_between_subject_pval' : sorted_baseline_between_subject_pval
	}

save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'cortical_distance_analysis', 'cv-1', 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'cortical_distance_analysis.npy'

np.save(os.path.join(save_dir, file_name), results)
