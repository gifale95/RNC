"""This code tests whether the controlling images found using the synthetic
fMRI responses of the train subjects generalize to the synthetic fMRI responses
for the left-out subject. Stats include confidence intervals and significance.

The code additionally compares the univariate RNC scores of pairwise ROI
comparisons from different stepwise ROI distances.

This code is available at:
https://github.com/gifale95/RNC/blob/main/02_univariate_rnc/03_stats.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
cv : int
	'1' if univariate RNC is cross-validated across subjects, '0' otherwise.
rois : list of str
	List of used ROIs.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_images : int
	Number of controlling images kept.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import pearsonr
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import page_trend_test

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_images', type=int, default=25)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Univariate RNC stats <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Pairwise ROI comparisons
# =============================================================================
# 0: V1
# 1: V2
# 2: V3
# 3: hV4
r1 = [0, 0, 0, 1, 1, 2]
r2 = [1, 2, 3, 2, 3, 3]


# =============================================================================
# Load the univariate RNC results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'image_ranking',
	'cv-'+format(args.cv), 'imageset-'+args.imageset)

if args.cv == 0:
	file_name = 'image_ranking.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	uni_resp = np.nanmean(data_dict['uni_resp'], 0)
	high_1_high_2 = np.zeros((len(r1), args.n_images), dtype=np.int32)
	low_1_low_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)
	high_1_low_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)
	low_1_high_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)
	for r in range(len(r1)):
		h1h2 = data_dict['high_1_high_2'][r]
		l1l2 = data_dict['low_1_low_2'][r]
		h1l2 = data_dict['high_1_low_2'][r]
		l1h2 = data_dict['low_1_high_2'][r]
		idx_nan_h1h2 = np.isnan(h1h2)
		idx_nan_l1l2 = np.isnan(l1l2)
		idx_nan_h1l2 = np.isnan(h1l2)
		idx_nan_l1h2 = np.isnan(l1h2)
		high_1_high_2[r] = h1h2[~idx_nan_h1h2][:args.n_images].astype(np.int32)
		low_1_low_2[r] = l1l2[~idx_nan_l1l2][:args.n_images].astype(np.int32)
		high_1_low_2[r] = h1l2[~idx_nan_h1l2][:args.n_images].astype(np.int32)
		low_1_high_2[r] = l1h2[~idx_nan_l1h2][:args.n_images].astype(np.int32)

elif args.cv == 1:
	uni_resp = []
	high_1_high_2 = []
	high_1_low_2 = []
	low_1_high_2 = []
	low_1_low_2 = []
	for s in args.all_subjects:
		file_name = 'image_ranking_cv_subject-' + format(s, '02') + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		uni_resp.append(data_dict['uni_resp'][s-1])
		h1h2_all = np.zeros((len(r1), args.n_images), dtype=np.int32)
		l1l2_all = np.zeros((h1h2_all.shape), dtype=np.int32)
		h1l2_all = np.zeros((h1h2_all.shape), dtype=np.int32)
		l1h2_all = np.zeros((h1h2_all.shape), dtype=np.int32)
		for r in range(len(r1)):
			h1h2 = data_dict['high_1_high_2'][r]
			l1l2 = data_dict['low_1_low_2'][r]
			h1l2 = data_dict['high_1_low_2'][r]
			l1h2 = data_dict['low_1_high_2'][r]
			idx_nan_h1h2 = np.isnan(h1h2)
			idx_nan_l1l2 = np.isnan(l1l2)
			idx_nan_h1l2 = np.isnan(h1l2)
			idx_nan_l1h2 = np.isnan(l1h2)
			h1h2_all[r] = h1h2[~idx_nan_h1h2][:args.n_images].astype(np.int32)
			l1l2_all[r] = l1l2[~idx_nan_l1l2][:args.n_images].astype(np.int32)
			h1l2_all[r] = h1l2[~idx_nan_h1l2][:args.n_images].astype(np.int32)
			l1h2_all[r] = l1h2[~idx_nan_l1h2][:args.n_images].astype(np.int32)
		high_1_high_2.append(h1h2_all)
		low_1_low_2.append(l1l2_all)
		high_1_low_2.append(h1l2_all)
		low_1_high_2.append(l1h2_all)
	uni_resp = np.asarray(uni_resp)
	high_1_high_2 = np.asarray(high_1_high_2)
	low_1_low_2 = np.asarray(low_1_low_2)
	high_1_low_2 = np.asarray(high_1_low_2)
	low_1_high_2 = np.asarray(low_1_high_2)


# =============================================================================
# Load the univariate RNC baseline
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'baseline', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if args.cv == 0:
	file_name = 'baseline.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	baseline_images = data_dict['baseline_images']
	baseline_images_score = data_dict['baseline_images_score']
elif args.cv == 1:
	baseline_images = []
	baseline_images_score_test = []
	baseline_images_score_train = []
	for s in args.all_subjects:
		file_name = 'baseline_cv_subject-' + format(s, '02') + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_images.append(data_dict['baseline_images'])
		baseline_images_score_test.append(
			data_dict['baseline_images_score_test'])
		baseline_images_score_train.append(
			data_dict['baseline_images_score_train'])
	baseline_images = np.asarray(baseline_images)
	baseline_images_score_test = np.asarray(baseline_images_score_test)
	baseline_images_score_train = np.asarray(baseline_images_score_train)


# =============================================================================
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subjects synthetic univariate fMRI responses for the controlling
# images from the four neural control conditions, as well as for the baseline
# images.

if args.cv == 1:

	# Synthetic univariate fMRI responses arrays of shape:
	# (Subjects x ROI comparisons x ROI pair per comparison x Target images)
	high_1_high_2_resp = np.zeros((len(args.all_subjects), len(r1), 2,
		args.n_images))
	high_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
	low_1_high_2_resp = np.zeros(high_1_high_2_resp.shape)
	low_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
	baseline_resp = np.zeros(high_1_high_2_resp.shape)

	for s in range(len(args.all_subjects)):
		for r in range(len(r1)):

			high_1_high_2_resp[s,r,0] = \
				uni_resp[s,r1[r],high_1_high_2[s,r]]
			high_1_high_2_resp[s,r,1] = \
				uni_resp[s,r2[r],high_1_high_2[s,r]]
			high_1_low_2_resp[s,r,0] = \
				uni_resp[s,r1[r],high_1_low_2[s,r]]
			high_1_low_2_resp[s,r,1] = \
				uni_resp[s,r2[r],high_1_low_2[s,r]]
			low_1_high_2_resp[s,r,0] = \
				uni_resp[s,r1[r],low_1_high_2[s,r]]
			low_1_high_2_resp[s,r,1] = \
				uni_resp[s,r2[r],low_1_high_2[s,r]]
			low_1_low_2_resp[s,r,0] = \
				uni_resp[s,r1[r],low_1_low_2[s,r]]
			low_1_low_2_resp[s,r,1] = \
				uni_resp[s,r2[r],low_1_low_2[s,r]]
			baseline_resp[s,r,0] = \
				uni_resp[s,r1[r],baseline_images[s,r1[r]]]
			baseline_resp[s,r,1] = \
				uni_resp[s,r2[r],baseline_images[s,r2[r]]]


# =============================================================================
# Compute the 95% confidence intervals (only for cv==1)
# =============================================================================
# Compute the confidence intervals of the cross-validated synthetic univariate
# fMRI responses for the controlling images (averaged across the N best
# controlling images), across the 8 (NSD) subjects.

if args.cv == 1:

	# CI arrays of shape:
	# (CI percentiles x ROI comparisons x ROI pair per comparison)
	ci_high_1_high_2 = np.zeros((2, len(r1), 2))
	ci_high_1_low_2 = np.zeros(ci_high_1_high_2.shape)
	ci_low_1_high_2 = np.zeros(ci_high_1_high_2.shape)
	ci_low_1_low_2 = np.zeros(ci_high_1_high_2.shape)
	ci_baseline = np.zeros(ci_high_1_high_2.shape)

	for r in tqdm(range(len(r1)), leave=False):

		# Empty CI distribution arrays
		h1h2_resp_roi_1_dist = np.zeros((args.n_iter))
		h1h2_resp_roi_2_dist = np.zeros((args.n_iter))
		h1l2_resp_roi_1_dist = np.zeros((args.n_iter))
		h1l2_resp_roi_2_dist = np.zeros((args.n_iter))
		l1h2_resp_roi_1_dist = np.zeros((args.n_iter))
		l1h2_resp_roi_2_dist = np.zeros((args.n_iter))
		l1l2_resp_roi_1_dist = np.zeros((args.n_iter))
		l1l2_resp_roi_2_dist = np.zeros((args.n_iter))
		control_resp_roi_1_dist = np.zeros((args.n_iter))
		control_resp_roi_2_dist = np.zeros((args.n_iter))

		# Compute the CI distributions
		for i in range(args.n_iter):
			idx_resample = resample(np.arange(len(args.all_subjects)))
			h1h2_resp_roi_1_dist[i] = np.mean(np.mean(
				high_1_high_2_resp[idx_resample,r,0,:], 1))
			h1h2_resp_roi_2_dist[i] = np.mean(np.mean(
				high_1_high_2_resp[idx_resample,r,1,:], 1))
			h1l2_resp_roi_1_dist[i] = np.mean(np.mean(
				high_1_low_2_resp[idx_resample,r,0,:], 1))
			h1l2_resp_roi_2_dist[i] = np.mean(np.mean(
				high_1_low_2_resp[idx_resample,r,1,:], 1))
			l1h2_resp_roi_1_dist[i] = np.mean(np.mean(
				low_1_high_2_resp[idx_resample,r,0,:], 1))
			l1h2_resp_roi_2_dist[i] = np.mean(np.mean(
				low_1_high_2_resp[idx_resample,r,1,:], 1))
			l1l2_resp_roi_1_dist[i] = np.mean(np.mean(
				low_1_low_2_resp[idx_resample,r,0,:], 1))
			l1l2_resp_roi_2_dist[i] = np.mean(np.mean(
				low_1_low_2_resp[idx_resample,r,1,:], 1))
			control_resp_roi_1_dist[i] = np.mean(np.mean(
				baseline_resp[idx_resample,r,0,:], 1))
			control_resp_roi_2_dist[i] = np.mean(np.mean(
				baseline_resp[idx_resample,r,1,:], 1))

		# Get the 5th and 95th CI distributions percentiles
		ci_high_1_high_2[0,r,0] = np.percentile(h1h2_resp_roi_1_dist, 2.5)
		ci_high_1_high_2[1,r,0] = np.percentile(h1h2_resp_roi_1_dist, 97.5)
		ci_high_1_high_2[0,r,1] = np.percentile(h1h2_resp_roi_2_dist, 2.5)
		ci_high_1_high_2[1,r,1] = np.percentile(h1h2_resp_roi_2_dist, 97.5)
		ci_high_1_low_2[0,r,0] = np.percentile(h1l2_resp_roi_1_dist, 2.5)
		ci_high_1_low_2[1,r,0] = np.percentile(h1l2_resp_roi_1_dist, 97.5)
		ci_high_1_low_2[0,r,1] = np.percentile(h1l2_resp_roi_2_dist, 2.5)
		ci_high_1_low_2[1,r,1] = np.percentile(h1l2_resp_roi_2_dist, 97.5)
		ci_low_1_high_2[0,r,0] = np.percentile(l1h2_resp_roi_1_dist, 2.5)
		ci_low_1_high_2[1,r,0] = np.percentile(l1h2_resp_roi_1_dist, 97.5)
		ci_low_1_high_2[0,r,1] = np.percentile(l1h2_resp_roi_2_dist, 2.5)
		ci_low_1_high_2[1,r,1] = np.percentile(l1h2_resp_roi_2_dist, 97.5)
		ci_low_1_low_2[0,r,0] = np.percentile(l1l2_resp_roi_1_dist, 2.5)
		ci_low_1_low_2[1,r,0] = np.percentile(l1l2_resp_roi_1_dist, 97.5)
		ci_low_1_low_2[0,r,1] = np.percentile(l1l2_resp_roi_2_dist, 2.5)
		ci_low_1_low_2[1,r,1] = np.percentile(l1l2_resp_roi_2_dist, 97.5)
		ci_baseline[0,r,0] = np.percentile(control_resp_roi_1_dist, 2.5)
		ci_baseline[1,r,0] = np.percentile(control_resp_roi_1_dist, 97.5)
		ci_baseline[0,r,1] = np.percentile(control_resp_roi_2_dist, 2.5)
		ci_baseline[1,r,1] = np.percentile(control_resp_roi_2_dist, 97.5)


# =============================================================================
# Compute the significance (only for cv==1)
# =============================================================================
# Compute the significance between the synthetic univariate fMRI responses for
# the neural control images (averaged across the N best controlling images), and
# the synthetic univariate fMRI univariate responses for the baseline images,
# across the 8 (NSD) subjects.

if args.cv == 1:

	# p-value arrays of shape:
	# (ROI comparisons x ROI pair per comparison)
	pval_high_1_high_2 = np.ones((len(r1), 2))
	pval_high_1_low_2 = np.ones(pval_high_1_high_2.shape)
	pval_low_1_high_2 = np.ones(pval_high_1_high_2.shape)
	pval_low_1_low_2 = np.ones(pval_high_1_high_2.shape)
	pval_corrected_high_1_high_2 = np.ones(pval_high_1_high_2.shape)
	pval_corrected_high_1_low_2 = np.ones(pval_high_1_high_2.shape)
	pval_corrected_low_1_high_2 = np.ones(pval_high_1_high_2.shape)
	pval_corrected_low_1_low_2 = np.ones(pval_high_1_high_2.shape)
	sig_high_1_high_2 = np.zeros(pval_high_1_high_2.shape, dtype=int)
	sig_high_1_low_2 = np.zeros(pval_high_1_high_2.shape, dtype=int)
	sig_low_1_high_2 = np.zeros(pval_high_1_high_2.shape, dtype=int)
	sig_low_1_low_2 = np.zeros(pval_high_1_high_2.shape, dtype=int)

	for r in range(len(r1)):

		# Compute significance using a paired samples t-test
		pval_high_1_high_2[r,0] = ttest_rel(np.mean(high_1_high_2_resp[:,r,0], 1),
			np.mean(baseline_resp[:,r,0], 1), alternative='greater')[1]
		pval_high_1_high_2[r,1] = ttest_rel(np.mean(high_1_high_2_resp[:,r,1], 1),
			np.mean(baseline_resp[:,r,1], 1), alternative='greater')[1]
		pval_high_1_low_2[r,0] = ttest_rel(np.mean(high_1_low_2_resp[:,r,0], 1),
			np.mean(baseline_resp[:,r,0], 1), alternative='greater')[1]
		pval_high_1_low_2[r,1] = ttest_rel(np.mean(high_1_low_2_resp[:,r,1], 1),
			np.mean(baseline_resp[:,r,1], 1), alternative='less')[1]
		pval_low_1_high_2[r,0] = ttest_rel(np.mean(low_1_high_2_resp[:,r,0], 1),
			np.mean(baseline_resp[:,r,0], 1), alternative='less')[1]
		pval_low_1_high_2[r,1] = ttest_rel(np.mean(low_1_high_2_resp[:,r,1], 1),
			np.mean(baseline_resp[:,r,1], 1), alternative='greater')[1]
		pval_low_1_low_2[r,0] = ttest_rel(np.mean(low_1_low_2_resp[:,r,0], 1),
			np.mean(baseline_resp[:,r,0], 1), alternative='less')[1]
		pval_low_1_low_2[r,1] = ttest_rel(np.mean(low_1_low_2_resp[:,r,1], 1),
			np.mean(baseline_resp[:,r,1], 1), alternative='less')[1]

		# Append all p-values together
		pval_all = []
		pval_all.append(pval_high_1_high_2[r])
		pval_all.append(pval_high_1_low_2[r])
		pval_all.append(pval_low_1_high_2[r])
		pval_all.append(pval_low_1_low_2[r])
		pval_all = np.asarray(pval_all)
		pval_all_shape = pval_all.shape
		pval_all = np.reshape(pval_all, -1)

		# Correct for multiple comparisons
		sig, pval_corrected, _, _ = multipletests(pval_all, 0.05, 'fdr_bh')
		sig = np.reshape(sig, pval_all_shape)
		pval_corrected = np.reshape(pval_corrected, pval_all_shape)

		# Store the significance and corrected p-values
		sig_high_1_high_2[r] = sig[0]
		sig_high_1_low_2[r] = sig[1]
		sig_low_1_high_2[r] = sig[2]
		sig_low_1_low_2[r] = sig[3]
		pval_corrected_high_1_high_2[r] = pval_corrected[0]
		pval_corrected_high_1_low_2[r] = pval_corrected[1]
		pval_corrected_low_1_high_2[r] = pval_corrected[2]
		pval_corrected_low_1_low_2[r] = pval_corrected[3]


# =============================================================================
# Correlate the ROI responses across all images
# =============================================================================
# This will provide the correlation scores between the synthetic fMRI univariate
# responses between each pariwise ROI comparison.

if args.cv == 1:

	# Correlation arrays of shape:
	# (Subjects x ROI comparisons)
	roi_pair_corr = np.zeros((len(args.all_subjects), len(r1)))
	for s in range(len(args.all_subjects)):
		for r in range(len(r1)):
			roi_pair_corr[s,r] = pearsonr(uni_resp[s,r1[r]],
				uni_resp[s,r2[r]])[0]


# =============================================================================
# Compute the difference between the ROI univariate responses for the control
# conditions and the ROI baseline univariate response, and sort these
# differences as a function of cortical distance
# =============================================================================
# There are three cortical distances:
# Cortical distance 1: [V1 vs. V2; V2 vs. V3; V3 vs. V4]
# Cortical distance 2: [V1 vs. V3; V2 vs. V4]
# Cortical distance 3: [V1 vs. V4]

if args.cv == 1:

	# Compute the absolute differences from baseline
	h1h2_base_diff = np.zeros((high_1_high_2_resp.shape))
	l1l2_base_diff = np.zeros((low_1_low_2_resp.shape))
	h1l2_base_diff = np.zeros((high_1_low_2_resp.shape))
	l1h2_base_diff = np.zeros((low_1_high_2_resp.shape))
	for s in range(h1h2_base_diff.shape[0]):
		for r in range(h1h2_base_diff.shape[1]):
			# h1h2
			h1h2_base_diff[s,r,0] = abs(high_1_high_2_resp[s,r,0] - 
				baseline_images_score_test[s,r1[r]])
			h1h2_base_diff[s,r,1] = abs(high_1_high_2_resp[s,r,1] - 
				baseline_images_score_test[s,r2[r]])
			# l1l2
			l1l2_base_diff[s,r,0] = abs(low_1_low_2_resp[s,r,0] - 
				baseline_images_score_test[s,r1[r]])
			l1l2_base_diff[s,r,1] = abs(low_1_low_2_resp[s,r,1] - 
				baseline_images_score_test[s,r2[r]])
			# h1l2
			h1l2_base_diff[s,r,0] = abs(high_1_low_2_resp[s,r,0] - 
				baseline_images_score_test[s,r1[r]])
			h1l2_base_diff[s,r,1] = abs(high_1_low_2_resp[s,r,1] - 
				baseline_images_score_test[s,r2[r]])
			# l1h2
			l1h2_base_diff[s,r,0] = abs(low_1_high_2_resp[s,r,0] - 
				baseline_images_score_test[s,r1[r]])
			l1h2_base_diff[s,r,1] = abs(low_1_high_2_resp[s,r,1] - 
				baseline_images_score_test[s,r2[r]])

	# Sort the absolute baseline differences based on cortical distances
	cortical_distances = [(0, 3, 5), (1, 4), (2)]

	# Sorted univariate response arrays of shape:
	# (Subjects x ROI cortical distances)
	sorted_h1h2_resp = np.zeros((len(args.all_subjects),
		len(cortical_distances)))
	sorted_l1l2_resp = np.zeros((len(args.all_subjects),
		len(cortical_distances)))
	sorted_h1l2_resp = np.zeros((len(args.all_subjects),
		len(cortical_distances)))
	sorted_l1h2_resp = np.zeros((len(args.all_subjects),
		len(cortical_distances)))
	for d, dist in enumerate(cortical_distances):
		# h1h2
		h1h2 = np.reshape(abs(high_1_high_2_resp[:,dist]),
			(len(args.all_subjects), -1))
		sorted_h1h2_resp[:,d] = np.mean(h1h2, 1)
		# l1l2
		l1l2 = np.reshape(abs(low_1_low_2_resp[:,dist]),
			(len(args.all_subjects), -1))
		sorted_l1l2_resp[:,d] = np.mean(l1l2, 1)
		# h1l2
		h1l2 = np.reshape(abs(high_1_low_2_resp[:,dist]),
			(len(args.all_subjects), -1))
		sorted_h1l2_resp[:,d] = np.mean(h1l2, 1)
		# l1h2
		l1h2 = np.reshape(abs(low_1_high_2_resp[:,dist]),
			(len(args.all_subjects), -1))
		sorted_l1h2_resp[:,d] = np.mean(l1h2, 1)

	# Compute the 95% confidence intervals
	# CI arrays of shape:
	# (CI percentiles x Cortical distances)
	ci_sorted_h1h2_resp = np.zeros((2, len(cortical_distances)))
	ci_sorted_l1l2_resp = np.zeros((2, len(cortical_distances)))
	ci_sorted_h1l2_resp = np.zeros((2, len(cortical_distances)))
	ci_sorted_l1h2_resp = np.zeros((2, len(cortical_distances)))
	for d in tqdm(range(len(cortical_distances)), leave=False):
		# Empty CI distribution array
		h1h2_dist = np.zeros((args.n_iter))
		l1l2_dist = np.zeros((args.n_iter))
		h1l2_dist = np.zeros((args.n_iter))
		l1h2_dist = np.zeros((args.n_iter))
		# Compute the CI distribution
		for i in range(args.n_iter):
			idx_resample = resample(np.arange(len(args.all_subjects)))
			h1h2_dist[i] = np.mean(sorted_h1h2_resp[idx_resample,d])
			l1l2_dist[i] = np.mean(sorted_l1l2_resp[idx_resample,d])
			h1l2_dist[i] = np.mean(sorted_h1l2_resp[idx_resample,d])
			l1h2_dist[i] = np.mean(sorted_l1h2_resp[idx_resample,d])
		# Get the 5th and 95th CI distributions percentiles
		ci_sorted_h1h2_resp[0,d] = np.percentile(h1h2_dist, 2.5)
		ci_sorted_h1h2_resp[1,d] = np.percentile(h1h2_dist, 97.5)
		ci_sorted_l1l2_resp[0,d] = np.percentile(l1l2_dist, 2.5)
		ci_sorted_l1l2_resp[1,d] = np.percentile(l1l2_dist, 97.5)
		ci_sorted_h1l2_resp[0,d] = np.percentile(h1l2_dist, 2.5)
		ci_sorted_h1l2_resp[1,d] = np.percentile(h1l2_dist, 97.5)
		ci_sorted_l1h2_resp[0,d] = np.percentile(l1h2_dist, 2.5)
		ci_sorted_l1h2_resp[1,d] = np.percentile(l1h2_dist, 97.5)

	# Test for a increasing trend
	sorted_h1l2_resp_increase = page_trend_test(sorted_h1l2_resp)
	sorted_l1h2_resp_increase = page_trend_test(sorted_l1h2_resp)
	# Correct for multiple comparisons
	pval = []
	pval.append(sorted_h1l2_resp_increase.pvalue)
	pval.append(sorted_l1h2_resp_increase.pvalue)
	pval = np.asarray(pval)
	sig, pval_corrected, _, _ = multipletests(pval, 0.05, 'fdr_bh')


# =============================================================================
# Compute the ROI response correlations as a function of cortical distance
# =============================================================================
# There are three cortical distances:
# Cortical distance 1: [V1 vs. V2; V2 vs. V3; V3 vs. V4]
# Cortical distance 2: [V1 vs. V3; V2 vs. V4]
# Cortical distance 3: [V1 vs. V4]

if args.cv == 1:

	# Sorted Correlation arrays of shape:
	# (Subjects x ROI cortical distances)
	sorted_corr = np.zeros((len(args.all_subjects), len(cortical_distances)))
	for d, dist in enumerate(cortical_distances):
		if type(dist) == tuple:
			sorted_corr[:,d] = np.mean(roi_pair_corr[:,dist], 1)
		else:
			sorted_corr[:,d] = roi_pair_corr[:,dist]

	# Compute the 95% confidence intervals
	# CI arrays of shape:
	# (CI percentiles x Cortical distances)
	ci_sorted_corr = np.zeros((2, len(cortical_distances)))
	for d in tqdm(range(len(cortical_distances)), leave=False):
		# Empty CI distribution array
		sorted_corr_dist = np.zeros((args.n_iter))
		# Compute the CI distribution
		for i in range(args.n_iter):
			idx_resample = resample(np.arange(len(args.all_subjects)))
			sorted_corr_dist[i] = np.mean(sorted_corr[idx_resample,d])
		# Get the 5th and 95th CI distributions percentiles
		ci_sorted_corr[0,d] = np.percentile(sorted_corr_dist, 2.5)
		ci_sorted_corr[1,d] = np.percentile(sorted_corr_dist, 97.5)

	# Test for a decreasing trend (conditions needs to be arranged in order of
	# increasing predicted mean, for the test to work).
	sorted_corr_decrease = page_trend_test(np.flip(sorted_corr, 1))


# =============================================================================
# Save the stats
# =============================================================================
if args.cv == 0:
	stats = {
		'uni_resp': uni_resp,
		'high_1_high_2': high_1_high_2,
		'high_1_low_2': high_1_low_2,
		'low_1_high_2': low_1_high_2,
		'low_1_low_2': low_1_low_2,
		'baseline_images': baseline_images,
		'baseline_images_score': baseline_images_score
		}

elif args.cv == 1:
	stats = {
		'uni_resp': uni_resp,
		'high_1_high_2': high_1_high_2,
		'high_1_low_2': high_1_low_2,
		'low_1_high_2': low_1_high_2,
		'low_1_low_2': low_1_low_2,
		'baseline_images': baseline_images,
		'baseline_images_score_test': baseline_images_score_test,
		'baseline_images_score_train': baseline_images_score_train,
		'high_1_high_2_resp': high_1_high_2_resp,
		'high_1_low_2_resp': high_1_low_2_resp,
		'low_1_high_2_resp': low_1_high_2_resp,
		'low_1_low_2_resp': low_1_low_2_resp,
		'baseline_resp': baseline_resp,
		'ci_high_1_high_2': ci_high_1_high_2,
		'ci_high_1_low_2': ci_high_1_low_2,
		'ci_low_1_high_2': ci_low_1_high_2,
		'ci_low_1_low_2': ci_low_1_low_2,
		'ci_baseline': ci_baseline,
		'pval_high_1_high_2': pval_high_1_high_2,
		'pval_high_1_low_2': pval_high_1_low_2,
		'pval_low_1_high_2': pval_low_1_high_2,
		'pval_low_1_low_2': pval_low_1_low_2,
		'pval_corrected_high_1_high_2': pval_high_1_high_2,
		'pval_corrected_high_1_low_2': pval_high_1_low_2,
		'pval_corrected_low_1_high_2': pval_low_1_high_2,
		'pval_corrected_low_1_low_2': pval_low_1_low_2,
		'sig_high_1_high_2': sig_high_1_high_2,
		'sig_high_1_low_2': sig_high_1_low_2,
		'sig_low_1_high_2': sig_low_1_high_2,
		'sig_low_1_low_2': sig_low_1_low_2,
		'roi_pair_corr': roi_pair_corr,
		'sorted_h1h2_resp': sorted_h1h2_resp,
		'sorted_l1l2_resp': sorted_l1l2_resp,
		'sorted_h1l2_resp': sorted_h1l2_resp,
		'sorted_l1h2_resp': sorted_l1h2_resp,
		'ci_sorted_h1h2_resp': ci_sorted_h1h2_resp,
		'ci_sorted_l1l2_resp': ci_sorted_l1l2_resp,
		'ci_sorted_h1l2_resp': ci_sorted_h1l2_resp,
		'ci_sorted_l1h2_resp': ci_sorted_l1h2_resp,
		'sorted_h1l2_resp_increase': sorted_h1l2_resp_increase,
		'sorted_l1h2_resp_increase': sorted_l1h2_resp_increase,
		'sorted_corr': sorted_corr,
		'ci_sorted_corr': ci_sorted_corr,
		'sorted_corr_decrease' : sorted_corr_decrease

		}

save_dir = os.path.join(args.project_dir, 'univariate_rnc', 'stats', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'stats'

np.save(os.path.join(save_dir, file_name), stats)

