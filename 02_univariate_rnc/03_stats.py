"""This code tests whether the controlling images found using the in silico
fMRI responses of the train subjects generalize to the in silico fMRI responses
for the left-out subject. Stats include confidence intervals and significance.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
cv : int
	If '1' univariate RNC leaves the data of one subject out for
	cross-validation, if '0' univariate RNC uses the data of all subjects.
roi_pair : str
	Used pairwise ROI combination.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_images : int
	Number of controlling images kept.
n_iter : int
	Amount of iterations for the permutation stats.
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
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--cv', type=int, default=1)
parser.add_argument('--roi_pair', type=str, default='V1-V2')
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
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]


# =============================================================================
# Load the univariate RNC univariate responses and controlling images
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'image_ranking',
	'cv-'+format(args.cv), 'imageset-'+args.imageset, args.roi_pair)

if args.cv == 0:
	file_name = 'image_ranking.npy'
	data_dict = np.load(os.path.join(data_dir, file_name),
		allow_pickle=True).item()
	# Get the univariate responses
	uni_resp = np.nanmean(data_dict['uni_resp'], 0)
	# Get the controlling images
	h1h2 = data_dict['high_1_high_2']
	l1l2 = data_dict['low_1_low_2']
	h1l2 = data_dict['high_1_low_2']
	l1h2 = data_dict['low_1_high_2']
	idx_nan_h1h2 = np.isnan(h1h2)
	idx_nan_l1l2 = np.isnan(l1l2)
	idx_nan_h1l2 = np.isnan(h1l2)
	idx_nan_l1h2 = np.isnan(l1h2)
	high_1_high_2 = h1h2[~idx_nan_h1h2][:args.n_images].astype(np.int32)
	low_1_low_2 = l1l2[~idx_nan_l1l2][:args.n_images].astype(np.int32)
	high_1_low_2 = h1l2[~idx_nan_h1l2][:args.n_images].astype(np.int32)
	low_1_high_2 = l1h2[~idx_nan_l1h2][:args.n_images].astype(np.int32)

elif args.cv == 1:
	uni_resp = []
	high_1_high_2 = []
	high_1_low_2 = []
	low_1_high_2 = []
	low_1_low_2 = []
	for s in all_subjects:
		file_name = 'image_ranking_cv_subject-' + format(s, '02') + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		# Get the univariate responses
		uni_resp.append(data_dict['uni_resp'][s-1])
		# Get the controlling images
		h1h2 = data_dict['high_1_high_2']
		l1l2 = data_dict['low_1_low_2']
		h1l2 = data_dict['high_1_low_2']
		l1h2 = data_dict['low_1_high_2']
		idx_nan_h1h2 = np.isnan(h1h2)
		idx_nan_l1l2 = np.isnan(l1l2)
		idx_nan_h1l2 = np.isnan(h1l2)
		idx_nan_l1h2 = np.isnan(l1h2)
		h1h2_all = h1h2[~idx_nan_h1h2][:args.n_images].astype(np.int32)
		l1l2_all = l1l2[~idx_nan_l1l2][:args.n_images].astype(np.int32)
		h1l2_all = h1l2[~idx_nan_h1l2][:args.n_images].astype(np.int32)
		l1h2_all = l1h2[~idx_nan_l1h2][:args.n_images].astype(np.int32)
		high_1_high_2.append(h1h2_all)
		low_1_low_2.append(l1l2_all)
		high_1_low_2.append(h1l2_all)
		low_1_high_2.append(l1h2_all)
	# Convert to numpy arrays
	uni_resp = np.asarray(uni_resp)
	high_1_high_2 = np.asarray(high_1_high_2)
	low_1_low_2 = np.asarray(low_1_low_2)
	high_1_low_2 = np.asarray(high_1_low_2)
	low_1_high_2 = np.asarray(low_1_high_2)


# =============================================================================
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'baseline', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset)

if args.cv == 0:
	baseline_images = np.zeros((len(rois), args.n_images), dtype=np.int32)
	baseline_resp = np.zeros((len(rois)))
	for r, roi in enumerate(rois):
		file_name = 'baseline_roi-' + roi + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_images[r] = data_dict['baseline_images']
		baseline_resp[r] = data_dict['baseline_images_score']

elif args.cv == 1:
	baseline_images = np.zeros((len(all_subjects), len(rois), args.n_images),
		dtype=np.int32)
	for s, sub in enumerate(all_subjects):
		for r, roi in enumerate(rois):
			file_name = 'baseline_cv_subject-' + format(sub, '02') + \
				'_roi-' + roi + '.npy'
			data_dict = np.load(os.path.join(data_dir, file_name),
				allow_pickle=True).item()
			baseline_images[s,r] = data_dict['baseline_images']


# =============================================================================
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subjects in silico univariate fMRI responses for the controlling
# images from the four neural control conditions, as well as for the baseline
# images.

if args.cv == 1:

	# In silico univariate fMRI responses arrays of shape:
	# (Subjects × 2 ROIs per comparison × Target images)
	high_1_high_2_resp = np.zeros((len(all_subjects), 2, args.n_images))
	high_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
	low_1_high_2_resp = np.zeros(high_1_high_2_resp.shape)
	low_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
	baseline_resp = np.zeros(high_1_high_2_resp.shape)

	for s in range(len(all_subjects)):

		high_1_high_2_resp[s,0] = uni_resp[s,0,high_1_high_2[s]]
		high_1_high_2_resp[s,1] = uni_resp[s,1,high_1_high_2[s]]
		high_1_low_2_resp[s,0] = uni_resp[s,0,high_1_low_2[s]]
		high_1_low_2_resp[s,1] = uni_resp[s,1,high_1_low_2[s]]
		low_1_high_2_resp[s,0] = uni_resp[s,0,low_1_high_2[s]]
		low_1_high_2_resp[s,1] = uni_resp[s,1,low_1_high_2[s]]
		low_1_low_2_resp[s,0] = uni_resp[s,0,low_1_low_2[s]]
		low_1_low_2_resp[s,1] = uni_resp[s,1,low_1_low_2[s]]
		baseline_resp[s,0] = uni_resp[s,0,baseline_images[s,0]]
		baseline_resp[s,1] = uni_resp[s,1,baseline_images[s,1]]


# =============================================================================
# Compute the 95% confidence intervals (only for cv==1)
# =============================================================================
# Compute the confidence intervals of the cross-validated in silico univariate
# fMRI responses for the controlling images (averaged across the N best
# controlling images), across the 8 (NSD) subjects.

if args.cv == 1:

	# CI arrays of shape:
	# (CI percentiles × 2 ROIs per comparison)
	ci_high_1_high_2 = np.zeros((2, 2))
	ci_high_1_low_2 = np.zeros(ci_high_1_high_2.shape)
	ci_low_1_high_2 = np.zeros(ci_high_1_high_2.shape)
	ci_low_1_low_2 = np.zeros(ci_high_1_high_2.shape)
	ci_baseline = np.zeros(ci_high_1_high_2.shape)

	# Empty CI distribution arrays
	h1h2_resp_roi_1_dist = np.zeros((args.n_iter))
	h1h2_resp_roi_2_dist = np.zeros((args.n_iter))
	h1l2_resp_roi_1_dist = np.zeros((args.n_iter))
	h1l2_resp_roi_2_dist = np.zeros((args.n_iter))
	l1h2_resp_roi_1_dist = np.zeros((args.n_iter))
	l1h2_resp_roi_2_dist = np.zeros((args.n_iter))
	l1l2_resp_roi_1_dist = np.zeros((args.n_iter))
	l1l2_resp_roi_2_dist = np.zeros((args.n_iter))
	baseline_resp_roi_1_dist = np.zeros((args.n_iter))
	baseline_resp_roi_2_dist = np.zeros((args.n_iter))

	# Compute the CI distributions
	for i in tqdm(range(args.n_iter)):
		idx_resample = resample(np.arange(len(all_subjects)))
		h1h2_resp_roi_1_dist[i] = np.mean(np.mean(
			high_1_high_2_resp[idx_resample,0,:], 1))
		h1h2_resp_roi_2_dist[i] = np.mean(np.mean(
			high_1_high_2_resp[idx_resample,1,:], 1))
		h1l2_resp_roi_1_dist[i] = np.mean(np.mean(
			high_1_low_2_resp[idx_resample,0,:], 1))
		h1l2_resp_roi_2_dist[i] = np.mean(np.mean(
			high_1_low_2_resp[idx_resample,1,:], 1))
		l1h2_resp_roi_1_dist[i] = np.mean(np.mean(
			low_1_high_2_resp[idx_resample,0,:], 1))
		l1h2_resp_roi_2_dist[i] = np.mean(np.mean(
			low_1_high_2_resp[idx_resample,1,:], 1))
		l1l2_resp_roi_1_dist[i] = np.mean(np.mean(
			low_1_low_2_resp[idx_resample,0,:], 1))
		l1l2_resp_roi_2_dist[i] = np.mean(np.mean(
			low_1_low_2_resp[idx_resample,1,:], 1))
		baseline_resp_roi_1_dist[i] = np.mean(np.mean(
			baseline_resp[idx_resample,0,:], 1))
		baseline_resp_roi_2_dist[i] = np.mean(np.mean(
			baseline_resp[idx_resample,1,:], 1))

	# Get the 5th and 95th CI distributions percentiles
	ci_high_1_high_2[0,0] = np.percentile(h1h2_resp_roi_1_dist, 2.5)
	ci_high_1_high_2[1,0] = np.percentile(h1h2_resp_roi_1_dist, 97.5)
	ci_high_1_high_2[0,1] = np.percentile(h1h2_resp_roi_2_dist, 2.5)
	ci_high_1_high_2[1,1] = np.percentile(h1h2_resp_roi_2_dist, 97.5)
	ci_high_1_low_2[0,0] = np.percentile(h1l2_resp_roi_1_dist, 2.5)
	ci_high_1_low_2[1,0] = np.percentile(h1l2_resp_roi_1_dist, 97.5)
	ci_high_1_low_2[0,1] = np.percentile(h1l2_resp_roi_2_dist, 2.5)
	ci_high_1_low_2[1,1] = np.percentile(h1l2_resp_roi_2_dist, 97.5)
	ci_low_1_high_2[0,0] = np.percentile(l1h2_resp_roi_1_dist, 2.5)
	ci_low_1_high_2[1,0] = np.percentile(l1h2_resp_roi_1_dist, 97.5)
	ci_low_1_high_2[0,1] = np.percentile(l1h2_resp_roi_2_dist, 2.5)
	ci_low_1_high_2[1,1] = np.percentile(l1h2_resp_roi_2_dist, 97.5)
	ci_low_1_low_2[0,0] = np.percentile(l1l2_resp_roi_1_dist, 2.5)
	ci_low_1_low_2[1,0] = np.percentile(l1l2_resp_roi_1_dist, 97.5)
	ci_low_1_low_2[0,1] = np.percentile(l1l2_resp_roi_2_dist, 2.5)
	ci_low_1_low_2[1,1] = np.percentile(l1l2_resp_roi_2_dist, 97.5)
	ci_baseline[0,0] = np.percentile(baseline_resp_roi_1_dist, 2.5)
	ci_baseline[1,0] = np.percentile(baseline_resp_roi_1_dist, 97.5)
	ci_baseline[0,1] = np.percentile(baseline_resp_roi_2_dist, 2.5)
	ci_baseline[1,1] = np.percentile(baseline_resp_roi_2_dist, 97.5)


# =============================================================================
# Compute the within-subject significance (only for cv==1)
# =============================================================================
if args.cv == 1:

	# Compute the difference between the mean univariate responses for
	# controlling and baseline images
	h1h2_minus_baseline = {}
	h1l2_minus_baseline = {}
	l1h2_minus_baseline = {}
	l1l2_minus_baseline = {}
	for r, roi in enumerate(rois):
		h1h2_minus_baseline[roi] = np.mean(high_1_high_2_resp[:,r], 1) - \
			np.mean(baseline_resp[:,r], 1)
		h1l2_minus_baseline[roi] = np.mean(high_1_low_2_resp[:,r], 1) - \
			np.mean(baseline_resp[:,r], 1)
		l1h2_minus_baseline[roi] = np.mean(low_1_high_2_resp[:,r], 1) - \
			np.mean(baseline_resp[:,r], 1)
		l1l2_minus_baseline[roi] = np.mean(low_1_low_2_resp[:,r], 1) - \
			np.mean(baseline_resp[:,r], 1)
	# Create the permutation-based null distributions
	h1h2_minus_baseline_null_dist = {}
	l1l2_minus_baseline_null_dist = {}
	h1l2_minus_baseline_null_dist = {}
	l1h2_minus_baseline_null_dist = {}
	baseline_roi_1_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	baseline_roi_2_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	h1h2_roi_1_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	l1l2_roi_1_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	h1l2_roi_1_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	l1h2_roi_1_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	h1h2_roi_2_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	l1l2_roi_2_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	h1l2_roi_2_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	l1h2_roi_2_null_dist = np.zeros((len(all_subjects), args.n_iter),
		dtype=np.float32)
	# Loop across iterations and subjects
	for i in tqdm(range(args.n_iter)):
		for s, sub in enumerate(all_subjects):
			# Shuffle the univariate responses across samples
			idx = np.arange(len(uni_resp[s,0]))
			np.random.shuffle(idx)
			# Images with high univariate responses for both ROIs
			h1h2_roi_1_null_dist[s,i] = np.mean(
				uni_resp[s,0][idx][high_1_high_2[s]])
			h1h2_roi_2_null_dist[s,i] = np.mean(
				uni_resp[s,1][idx][high_1_high_2[s]])
			# Images with low univariate responses for both ROIs
			l1l2_roi_1_null_dist[s,i] = np.mean(
				uni_resp[s,0][idx][low_1_low_2[s]])
			l1l2_roi_2_null_dist[s,i] = np.mean(
				uni_resp[s,1][idx][low_1_low_2[s]])
			# Images with high univariate responses for V1 and low univariate
			# responses for V4
			h1l2_roi_1_null_dist[s,i] = np.mean(
				uni_resp[s,0][idx][high_1_low_2[s]])
			h1l2_roi_2_null_dist[s,i] = np.mean(
				uni_resp[s,1][idx][high_1_low_2[s]])
			# Images with low univariate responses for V1 and high univariate
			# responses for V4
			l1h2_roi_1_null_dist[s,i] = np.mean(
				uni_resp[s,0][idx][low_1_high_2[s]])
			l1h2_roi_2_null_dist[s,i] = np.mean(
				uni_resp[s,1][idx][low_1_high_2[s]])
			# Baseline images
			baseline_roi_1_null_dist[s,i] = np.mean(
				uni_resp[s,0][idx][baseline_images[s,0]])
			baseline_roi_2_null_dist[s,i] = np.mean(
				uni_resp[s,1][idx][baseline_images[s,1]])
	# Store the difference between controlling and and baseline image univariate
	# responses
	h1h2_minus_baseline_null_dist[rois[0]] = h1h2_roi_1_null_dist - \
		baseline_roi_1_null_dist
	l1l2_minus_baseline_null_dist[rois[0]] = l1l2_roi_1_null_dist - \
		baseline_roi_1_null_dist
	h1l2_minus_baseline_null_dist[rois[0]] = h1l2_roi_1_null_dist - \
		baseline_roi_1_null_dist
	l1h2_minus_baseline_null_dist[rois[0]] = l1h2_roi_1_null_dist - \
		baseline_roi_1_null_dist
	h1h2_minus_baseline_null_dist[rois[1]] = h1h2_roi_2_null_dist - \
		baseline_roi_2_null_dist
	l1l2_minus_baseline_null_dist[rois[1]] = l1l2_roi_2_null_dist - \
		baseline_roi_2_null_dist
	h1l2_minus_baseline_null_dist[rois[1]] = h1l2_roi_2_null_dist - \
		baseline_roi_2_null_dist
	l1h2_minus_baseline_null_dist[rois[1]] = l1h2_roi_2_null_dist - \
		baseline_roi_2_null_dist

	# Compute the within-subject p-values
	h1h2_within_subject_pval = {}
	l1l2_within_subject_pval = {}
	h1l2_within_subject_pval = {}
	l1h2_within_subject_pval = {}
	for r in rois:
		h1h2 = np.zeros((len(all_subjects)), dtype=np.float32)
		l1l2 = np.zeros((len(all_subjects)), dtype=np.float32)
		h1l2 = np.zeros((len(all_subjects)), dtype=np.float32)
		l1h2 = np.zeros((len(all_subjects)), dtype=np.float32)
		# Compute the p-values
		for s, sub in enumerate(all_subjects):
			# h1h2
			idx = sum(h1h2_minus_baseline_null_dist[r][s] > \
				h1h2_minus_baseline[r][s])
			h1h2[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
			# l1l2
			idx = sum(l1l2_minus_baseline_null_dist[r][s] < \
				l1l2_minus_baseline[r][s])
			l1l2[s] = (idx + 1) / (args.n_iter + 1)
			if r == rois[0]:
				# h1l2
				idx = sum(h1l2_minus_baseline_null_dist[r][s] > \
					h1l2_minus_baseline[r][s])
				h1l2[s] = (idx + 1) / (args.n_iter + 1)
				# l1h2
				idx = sum(l1h2_minus_baseline_null_dist[r][s] < \
					l1h2_minus_baseline[r][s])
				l1h2[s] = (idx + 1) / (args.n_iter + 1)
			if r == rois[1]:
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
	for roi in rois:
		h1h2_within_subject_sig[roi] = np.zeros((len(all_subjects)))
		l1l2_within_subject_sig[roi] = np.zeros((len(all_subjects)))
		h1l2_within_subject_sig[roi] = np.zeros((len(all_subjects)))
		l1h2_within_subject_sig[roi] = np.zeros((len(all_subjects)))
	# Loop across subjects
	for s in range(len(all_subjects)):
		# Append the within-subject p-values across the 8 comparisons
		pvals = np.zeros((n_control_conditions, n_rois))
		for r, roi in enumerate(rois):
			pvals[0,r] = h1h2_within_subject_pval[roi][s]
			pvals[1,r] = l1l2_within_subject_pval[roi][s]
			pvals[2,r] = h1l2_within_subject_pval[roi][s]
			pvals[3,r] = l1h2_within_subject_pval[roi][s]
		pvals = pvals.flatten()
		# Correct for multiple comparisons
		sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
		sig = np.reshape(sig, (n_control_conditions, n_rois))
		# Store the significance scores
		for r, roi in enumerate(rois):
			h1h2_within_subject_sig[roi][s] = sig[0,r]
			l1l2_within_subject_sig[roi][s] = sig[1,r]
			h1l2_within_subject_sig[roi][s] = sig[2,r]
			l1h2_within_subject_sig[roi][s] = sig[3,r]


# =============================================================================
# Compute the between-subject significance (only for cv==1)
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

if args.cv == 1:

	n = len(all_subjects) # Total number of subjects
	p = 0.05 # probability of success in each trial

	h1h2_between_subject_pval = {}
	l1l2_between_subject_pval = {}
	h1l2_between_subject_pval = {}
	l1h2_between_subject_pval = {}

	for r in rois:

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
# Correlate the ROI responses across all images
# =============================================================================
# This will provide the correlation scores between the in silico fMRI
# univariate responses of the two ROIs within each pariwise ROI comparison.

if args.cv == 0:
	roi_pair_corr = pearsonr(uni_resp[0], uni_resp[1])[0]

if args.cv == 1:
	# Correlation arrays of shape: (Subjects)
	roi_pair_corr = np.zeros((len(all_subjects)))
	for s in range(len(all_subjects)):
		roi_pair_corr[s] = pearsonr(uni_resp[s,0], uni_resp[s,1])[0]


# =============================================================================
# Save the stats
# =============================================================================
if args.cv == 0:
	stats = {
		'roi_1': roi_1,
		'roi_2': roi_2,
		'uni_resp': uni_resp,
		'high_1_high_2': high_1_high_2,
		'high_1_low_2': high_1_low_2,
		'low_1_high_2': low_1_high_2,
		'low_1_low_2': low_1_low_2,
		'baseline_images': baseline_images,
		'baseline_resp': baseline_resp,
		'roi_pair_corr': roi_pair_corr
		}

elif args.cv == 1:
	stats = {
		'roi_1': roi_1,
		'roi_2': roi_2,
		'uni_resp': uni_resp,
		'high_1_high_2': high_1_high_2,
		'high_1_low_2': high_1_low_2,
		'low_1_high_2': low_1_high_2,
		'low_1_low_2': low_1_low_2,
		'baseline_images': baseline_images,
		'baseline_resp': baseline_resp,
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
		'h1h2_within_subject_pval': h1h2_within_subject_pval,
		'l1l2_within_subject_pval': l1l2_within_subject_pval,
		'h1l2_within_subject_pval': h1l2_within_subject_pval,
		'l1h2_within_subject_pval': l1h2_within_subject_pval,
		'h1h2_within_subject_sig': h1h2_within_subject_sig,
		'l1l2_within_subject_sig': l1l2_within_subject_sig,
		'h1l2_within_subject_sig': h1l2_within_subject_sig,
		'l1h2_within_subject_sig': l1h2_within_subject_sig,
		'h1h2_between_subject_pval': h1h2_between_subject_pval,
		'l1l2_between_subject_pval': l1l2_between_subject_pval,
		'h1l2_between_subject_pval': h1l2_between_subject_pval,
		'l1h2_between_subject_pval': l1h2_between_subject_pval,
		'roi_pair_corr': roi_pair_corr
		}

save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-'+
	format(args.cv), 'imageset-'+args.imageset, args.roi_pair)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), stats)
