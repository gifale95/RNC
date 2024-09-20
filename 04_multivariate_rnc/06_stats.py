"""This code tests whether the controlling images found using the synthetic
fMRI responses of the train subjects generalize to the synthetic fMRI responses
for the left-out subject. Stats include confidence intervals and significance.

The code additionally compares the multivariate RNC scores of pairwise ROI
comparisons from different stepwise ROI distances.

This code is available at:
https://github.com/gifale95/RNC/04_multivariate_rnc/06_stats.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
cv : int
	'1' if univariate RNC is cross-validated across subjects, '0' otherwise.
roi_pair : int
	Integer indicating the chosen pairwise ROI combination for which to compute
	the stats. Possible values are '0' (V1-V2), '1' (V1-V3), '2' (V1-hV4), '3'
	(V2-V3), '4' (V2-hV4), '5' (V3-hV4).
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
control_conditions : list
	List of used control conditions.
n_iter : int
	Amount of iterations for creating confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from copy import copy
import random
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

from utils import load_rsms
from utils import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi_pair', type=int, default=0)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--control_conditions', type=list, default=['align', 'disentangle'])
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Multivariate RNC stats <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# ROI pair combinations
# =============================================================================
# 0 --> V1 - V2
# 1 --> V1 - V3
# 2 --> V1 - V4
# 3 --> V2 - V3
# 4 --> V2 - V4
# 5 --> V3 - V4
roi_comb_names = [['V1', 'V2'], ['V1', 'V3'], ['V1', 'hV4'], ['V2', 'V3'],
	['V2', 'hV4'], ['V3', 'hV4']]
roi_1 = roi_comb_names[args.roi_pair][0]
roi_2 = roi_comb_names[args.roi_pair][1]


# =============================================================================
# Load the mutivariate RNC results
# =============================================================================
best_generation_scores_train = {}
best_generation_image_batches = {}

if args.cv == 0:
	for c in args.control_conditions:
		data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
			'best_image_batches', 'cv-'+format(args.cv), 'imageset-'+
			args.imageset, roi_1+'-'+roi_2, 'control_condition-'+c,
			'best_image_batches.npy')
		results = np.load(data_dir, allow_pickle=True).item()
		best_generation_scores_train[c] = \
			results['image_batches_scores']
		best_generation_image_batches[c] = \
			results['best_generation_image_batches']

elif args.cv == 1:
	for c in args.control_conditions:
		best_gen_scores_train = []
		best_gen_chromosomes = []
		for s in args.all_subjects:
			data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
				'best_image_batches', 'cv-'+format(args.cv), 'imageset-'+
				args.imageset, roi_1+'-'+roi_2, 'control_condition-'+c,
				'best_image_batches_subject-'+format(s, '02')+'.npy')
			results = np.load(data_dir, allow_pickle=True).item()
			best_gen_scores_train.append(copy(
				results['image_batches_scores']))
			best_gen_chromosomes.append(copy(
				results['best_generation_image_batches']))
		best_generation_scores_train[c] = np.asarray(best_gen_scores_train)
		best_generation_image_batches[c] = np.asarray(best_gen_chromosomes)


# =============================================================================
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subjects RSA scores for the controlling images from the two
# neural control conditions.

if args.cv == 1:

	# Cross-validate
	best_generation_scores_test = {}
	for c in args.control_conditions:
		scores_test_all = []
		for s in args.all_subjects:
			rsm_1, rsm_2 = load_rsms(args, s, 'test')
			scores_test = np.zeros((best_generation_image_batches[c].shape[1]))
			for g in range(best_generation_image_batches[c].shape[1]):
				image_batch = best_generation_image_batches[c][s-1,g]
				image_batch = np.reshape(image_batch, (1, len(image_batch)))
				scores_test[g] = evaluate(image_batch, rsm_1, rsm_2)[0]
			scores_test_all.append(scores_test)
			del rsm_1, rsm_2
		scores_test_all = np.asarray(scores_test_all)
		best_generation_scores_test[c] = copy(scores_test_all)


# =============================================================================
# Load the multivariate RNC baseline scores
# =============================================================================
data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'baseline',
	'cv-'+format(args.cv), 'imageset-'+args.imageset, roi_1+'-'+roi_2)

if args.cv == 0:
	file_name = 'baseline.npy'
	results = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	baseline_images = results['baseline_images']
	baseline_images_score = results['baseline_images_score']

elif args.cv == 1:
	baseline_images = []
	baseline_images_score_test = []
	baseline_images_score_train = []
	for s in args.all_subjects:
		file_name = 'baseline_cv_subject-' + format(s, '02') + '.npy'
		results = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_images.append(results['baseline_images'])
		baseline_images_score_test.append(results['baseline_images_score_test'])
		baseline_images_score_train.append(
			results['baseline_images_score_train'])
	baseline_images = np.asarray(baseline_images)
	baseline_images_score_test = np.asarray(baseline_images_score_test)
	baseline_images_score_train = np.asarray(baseline_images_score_train)


# =============================================================================
# Compute the 95% confidence intervals (only for cv==1)
# =============================================================================
# Compute the confidence intervals of the cross-validated RSA scores for the
# controlling images, across the 8 (NSD) subjects.


if args.cv == 1:

	# CI arrays of shape: (CI percentiles)
	ci_align = np.zeros((2))
	ci_disentangle = np.zeros(ci_align.shape)
	ci_baseline = np.zeros(ci_align.shape)

	# Empty CI distribution arrays
	align_dist = np.zeros((args.n_iter))
	disentangle_dist = np.zeros((args.n_iter))
	baseline_dist = np.zeros((args.n_iter))

	# Compute the CI distributions
	for i in tqdm(range(args.n_iter), leave=False):
		idx_resample = resample(np.arange(len(args.all_subjects)))
		align_dist[i] = np.mean(
			best_generation_scores_test['align'][idx_resample,-1])
		disentangle_dist[i] = np.mean(
			best_generation_scores_test['disentangle'][idx_resample,-1])
		baseline_dist[i] = np.mean(baseline_images_score_test[idx_resample])

	# Get the 5th and 95th CI distributions percentiles
	ci_align[0] = np.percentile(align_dist, 2.5)
	ci_align[1] = np.percentile(align_dist, 97.5)
	ci_disentangle[0] = np.percentile(disentangle_dist, 2.5)
	ci_disentangle[1] = np.percentile(disentangle_dist, 97.5)
	ci_baseline[0] = np.percentile(baseline_dist, 2.5)
	ci_baseline[1] = np.percentile(baseline_dist, 97.5)


# =============================================================================
# Calculate the significance (only for cv==1)
# =============================================================================
# Compute the significance between the RSA scores of the aligning/disentangling
# controlling images, and the RSA scores of the baseline images, across
# the 8 (NSD) subjects.

if args.cv == 1:

	# Compute significance using a paired samples t-test
	p_value_align = ttest_rel(best_generation_scores_test['align'][:,-1],
		baseline_images_score_test, alternative='greater')[1]
	p_value_disentangle = ttest_rel(
		best_generation_scores_test['disentangle'][:,-1],
		baseline_images_score_test, alternative='less')[1]

	# Correct for multiple comparisons
	p_values_all = np.append(p_value_align, p_value_disentangle)
	significance, p_values_corrected, _, _ = multipletests(p_values_all, 0.05,
		'fdr_bh')

	# Store the significance and corrected p-values
	significance_align = significance[0]
	significance_disentangle = significance[1]
	p_value_corrected_align = p_values_corrected[0]
	p_value_corrected_disentangle = p_values_corrected[1]


# =============================================================================
# Save the stats
# =============================================================================
if args.cv == 0:
	results = {
		'roi_1': roi_1,
		'roi_2': roi_2,
		'best_generation_scores_train': best_generation_scores_train,
		'best_generation_image_batches': best_generation_image_batches,
		'baseline_images': baseline_images,
		'baseline_images_score': baseline_images_score,
		}

elif args.cv == 1:
	results = {
		'roi_1': roi_1,
		'roi_2': roi_2,
		'best_generation_scores_train': best_generation_scores_train,
		'best_generation_scores_test': best_generation_scores_test,
		'best_generation_image_batches': best_generation_image_batches,
		'baseline_images': baseline_images,
		'baseline_images_score_test': baseline_images_score_test,
		'baseline_images_score_train': baseline_images_score_train,
		'ci_align': ci_align,
		'ci_disentangle': ci_disentangle,
		'ci_baseline': ci_baseline,
		'p_value_align': p_value_align,
		'p_value_disentangle': p_value_disentangle,
		'p_value_corrected_align': p_value_corrected_align,
		'p_value_corrected_disentangle': p_value_corrected_disentangle,
		'significance_align': significance_align,
		'significance_disentangle': significance_disentangle
		}

save_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset, roi_1+'-'+roi_2)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'stats'

np.save(os.path.join(save_dir, file_name), results)

