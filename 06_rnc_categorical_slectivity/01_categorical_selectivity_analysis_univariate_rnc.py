"""This code tests whether the univariate RNC alignment and disentanglement
scores change as a function of two areas being from the same or different
categorical selectivity groups. The two categorical selectivity groups are
animate objects (EBA, FFA), and scenes (PPA, RSC).

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
rois : list of str
	List of used ROIs.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
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
import h5py
from sklearn.utils import resample
from copy import copy
from itertools import combinations
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--rois', type=list, default=['EBA', 'FFA', 'PPA', 'RSC'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_images', type=int, default=25)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Univariate RNC categorical selectivity analysis <<<')
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
# Compute the univariate responses
# =============================================================================
# In silico fMRI univariate responses array of shape:
# (Subjects × ROIs × Images)
if args.imageset == 'nsd':
	images = 73000
elif args.imageset == 'imagenet_val':
	images = 50000
elif args.imageset == 'things':
	images = 26107
uni_resp = np.zeros((len(all_subjects), len(args.rois), images),
	dtype=np.float32)

for s, sub in enumerate(all_subjects):
	for r, roi in enumerate(args.rois):

		# Load the in silico fMRI responses
		data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri', 'imageset-'+args.imageset,
			'insilico_fmri_responses_sub-'+format(sub, '02')+'_roi-'+roi+'.h5')
		betas = h5py.File(data_dir).get('insilico_fmri_responses')

		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri','ncsnr_sub-'+format(sub, '02')+'_roi-'+roi+'.npy')
		ncsnr = np.load(ncsnr_dir)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold.
		best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
		# For subject 4 of the Visual Illusion Reconstruction dataset, lower the
		# ncsnr theshold for ROI hV4 to 0.4, since there are no voxels above a
		# threshold of 0.5
		if args.encoding_models_train_dataset == 'VisualIllusionRecon':
			if roi == 'hV4' and sub == 4 and args.ncsnr_threshold > 0.4:
				best_voxels = np.where(ncsnr > 0.4)[0]
		betas = betas[:,best_voxels]

		# Score the fMRI activity across voxels (there might be NaN values
		# since some subjects have missing data)
		uni_resp[s,r] = np.nanmean(betas, 1)


# =============================================================================
# Load the univariate RNC controlling images
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'image_ranking',
	'cv-1', 'imageset-'+args.imageset)

# Empty variables of shape:
# (Subjects × ROI pairwise comparisons × Target images)
high_1_high_2 = np.zeros((len(all_subjects), len(roi_comb_names),
	args.n_images), dtype=np.int32)
high_1_low_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)
low_1_high_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)
low_1_low_2 = np.zeros((high_1_high_2.shape), dtype=np.int32)

for r, roi_pair in enumerate(roi_comb_names):
	for s, sub in enumerate(all_subjects):

		file_name = 'image_ranking_cv_subject-' + format(sub, '02') + '.npy'
		data_dict = np.load(os.path.join(data_dir, roi_pair, file_name),
			allow_pickle=True).item()
		h1h2 = data_dict['high_1_high_2']
		l1l2 = data_dict['low_1_low_2']
		h1l2 = data_dict['high_1_low_2']
		l1h2 = data_dict['low_1_high_2']
		idx_nan_h1h2 = np.isnan(h1h2)
		idx_nan_l1l2 = np.isnan(l1l2)
		idx_nan_h1l2 = np.isnan(h1l2)
		idx_nan_l1h2 = np.isnan(l1h2)
		high_1_high_2[s,r] = h1h2[~idx_nan_h1h2][:args.n_images].astype(np.int32)
		low_1_low_2[s,r] = l1l2[~idx_nan_l1l2][:args.n_images].astype(np.int32)
		high_1_low_2[s,r] = h1l2[~idx_nan_h1l2][:args.n_images].astype(np.int32)
		low_1_high_2[s,r] = l1h2[~idx_nan_l1h2][:args.n_images].astype(np.int32)


# =============================================================================
# Load the univariate RNC baseline images
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'baseline', 'cv-1',
	'imageset-'+args.imageset)

# Empty variables of shape:
# (Subjects × ROIs × Controlling images)
baseline_images = np.zeros((len(all_subjects), len(args.rois),
	args.n_images), dtype=np.int32)

for s, sub in enumerate(all_subjects):
	for r, roi in enumerate(args.rois):

		file_name = 'baseline_cv_subject-' + format(sub, '02') + \
			'_roi-' + roi + '.npy'
		data_dict = np.load(os.path.join(data_dir, file_name),
			allow_pickle=True).item()
		baseline_images[s,r] = data_dict['baseline_images']


# =============================================================================
# Validate the neural control conditions across subjects
# =============================================================================
# Get the test subjects' in silico univariate fMRI responses for the controlling
# images from the four neural control conditions, as well as for the baseline
# images.

# In silico univariate fMRI responses arrays of shape:
# (Subjects × ROI pairwise comparisons × ROIs per comparison × Target images)
high_1_high_2_resp = np.zeros((len(all_subjects), len(r1), 2,
	args.n_images))
high_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
low_1_high_2_resp = np.zeros(high_1_high_2_resp.shape)
low_1_low_2_resp = np.zeros(high_1_high_2_resp.shape)
baseline_resp = np.zeros(high_1_high_2_resp.shape)

for s in range(len(all_subjects)):
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
# Compute the difference between the ROI univariate responses for the control
# conditions and the ROI baseline univariate response, and sort these
# differences as a function of two areas being from the same or different
# categorical selectivity groups
# =============================================================================
# Compute the absolute difference from baseline
h1h2_base_diff = np.zeros((high_1_high_2_resp.shape))
l1l2_base_diff = np.zeros((low_1_low_2_resp.shape))
h1l2_base_diff = np.zeros((high_1_low_2_resp.shape))
l1h2_base_diff = np.zeros((low_1_high_2_resp.shape))
for s in range(h1h2_base_diff.shape[0]):
	for r in range(h1h2_base_diff.shape[1]):
		# h1h2
		h1h2_base_diff[s,r,0] = abs(high_1_high_2_resp[s,r,0] - 
			np.mean(baseline_resp[s,r,0]))
		h1h2_base_diff[s,r,1] = abs(high_1_high_2_resp[s,r,1] - 
			np.mean(baseline_resp[s,r,1]))
		# l1l2
		l1l2_base_diff[s,r,0] = abs(low_1_low_2_resp[s,r,0] - 
			np.mean(baseline_resp[s,r,0]))
		l1l2_base_diff[s,r,1] = abs(low_1_low_2_resp[s,r,1] - 
			np.mean(baseline_resp[s,r,1]))
		# h1l2
		h1l2_base_diff[s,r,0] = abs(high_1_low_2_resp[s,r,0] - 
			np.mean(baseline_resp[s,r,0]))
		h1l2_base_diff[s,r,1] = abs(high_1_low_2_resp[s,r,1] - 
			np.mean(baseline_resp[s,r,1]))
		# l1h2
		l1h2_base_diff[s,r,0] = abs(low_1_high_2_resp[s,r,0] - 
			np.mean(baseline_resp[s,r,0]))
		l1h2_base_diff[s,r,1] = abs(low_1_high_2_resp[s,r,1] - 
			np.mean(baseline_resp[s,r,1]))

# Aggregate the absolute baseline difference scores as a function of two areas
# being from the same or different categorical selectivity groups
# Within group comparisons: [EBA vs. FFA; PPA vs. RSC]
# Between group comparisons: [EBA vs. PPA; EBA vs. RSC, FFA vs. PPA; FFA vs. RSC]
groups = [(0, 5), (1, 2, 3, 4)]
# Aggregated absolute baseline difference scores arrays of shape:
# (Subjects × 2 categorical selectivity groups comparisons)
sorted_h1h2_base_diff = np.zeros((len(all_subjects), len(groups)))
sorted_l1l2_base_diff = np.zeros((len(all_subjects), len(groups)))
sorted_h1l2_base_diff = np.zeros((len(all_subjects), len(groups)))
sorted_l1h2_base_diff = np.zeros((len(all_subjects), len(groups)))
for s in range(len(all_subjects)):
	for d, dist in enumerate(groups):
		# h1h2
		sorted_h1h2_base_diff[s,d] = np.mean(h1h2_base_diff[s,dist])
		# l1l2
		sorted_l1l2_base_diff[s,d] = np.mean(l1l2_base_diff[s,dist])
		# h1l2
		sorted_h1l2_base_diff[s,d] = np.mean(h1l2_base_diff[s,dist])
		# l1h2
		sorted_l1h2_base_diff[s,d] = np.mean(l1h2_base_diff[s,dist])

# Compute the difference in absolute baseline difference scores between
# categorical selectivity group comparisons
# h1h2
sorted_h1h2_base_diff_diff = sorted_h1h2_base_diff[:,0] - \
	sorted_h1h2_base_diff[:,1]
# l1l2
sorted_l1l2_base_diff_diff = sorted_l1l2_base_diff[:,0] - \
	sorted_l1l2_base_diff[:,1]
# h1l2
sorted_h1l2_base_diff_diff = sorted_h1l2_base_diff[:,0] - \
	sorted_h1l2_base_diff[:,1]
# l1h2
sorted_l1h2_base_diff_diff = sorted_l1h2_base_diff[:,0] - \
	sorted_l1h2_base_diff[:,1]

# Compute the 95% confidence intervals
# CI arrays of shape:
# (CI percentiles × Cortical distances)
ci_sorted_h1h2_base_diff = np.zeros((2, len(groups)))
ci_sorted_l1l2_base_diff = np.zeros((2, len(groups)))
ci_sorted_h1l2_base_diff = np.zeros((2, len(groups)))
ci_sorted_l1h2_base_diff = np.zeros((2, len(groups)))
for d in tqdm(range(len(groups)), leave=False):
	# Empty CI distribution array
	h1h2_dist = np.zeros((args.n_iter))
	l1l2_dist = np.zeros((args.n_iter))
	h1l2_dist = np.zeros((args.n_iter))
	l1h2_dist = np.zeros((args.n_iter))
	# Compute the CI distribution
	for i in range(args.n_iter):
		idx_resample = resample(np.arange(len(all_subjects)))
		h1h2_dist[i] = np.mean(sorted_h1h2_base_diff[idx_resample,d])
		l1l2_dist[i] = np.mean(sorted_l1l2_base_diff[idx_resample,d])
		h1l2_dist[i] = np.mean(sorted_h1l2_base_diff[idx_resample,d])
		l1h2_dist[i] = np.mean(sorted_l1h2_base_diff[idx_resample,d])
	# Get the 5th and 95th CI distributions percentiles
	ci_sorted_h1h2_base_diff[0,d] = np.percentile(h1h2_dist, 2.5)
	ci_sorted_h1h2_base_diff[1,d] = np.percentile(h1h2_dist, 97.5)
	ci_sorted_l1l2_base_diff[0,d] = np.percentile(l1l2_dist, 2.5)
	ci_sorted_l1l2_base_diff[1,d] = np.percentile(l1l2_dist, 97.5)
	ci_sorted_h1l2_base_diff[0,d] = np.percentile(h1l2_dist, 2.5)
	ci_sorted_h1l2_base_diff[1,d] = np.percentile(h1l2_dist, 97.5)
	ci_sorted_l1h2_base_diff[0,d] = np.percentile(l1h2_dist, 2.5)
	ci_sorted_l1h2_base_diff[1,d] = np.percentile(l1h2_dist, 97.5)

# Test for significant differences (within-subject significance)
# Create the permutation-based null distributions
# Absolute baseline difference scores null distribution arrays of shape:
# (Iterations × Subjects × ROI pairwise comparisons × ROIs per comparison × Target images)
h1h2_base_diff_null_dist = np.zeros((args.n_iter, len(all_subjects),
	len(r1), 2, args.n_images), dtype=np.float32)
l1l2_base_diff_null_dist = np.zeros((h1h2_base_diff_null_dist.shape),
	dtype=np.float32)
h1l2_base_diff_null_dist = np.zeros((h1h2_base_diff_null_dist.shape),
	dtype=np.float32)
l1h2_base_diff_null_dist = np.zeros((h1h2_base_diff_null_dist.shape),
	dtype=np.float32)
shuffled_uni_resp = copy(uni_resp)
shape = shuffled_uni_resp.shape
shuffled_uni_resp = np.reshape(shuffled_uni_resp, (len(all_subjects), -1))
idx = np.arange((len(args.rois)*images))
for i in tqdm(range(args.n_iter)):
	# Shuffle the univariate responses across ROIs
	np.random.shuffle(idx)
	shuffled_uni_resp = shuffled_uni_resp[:,idx]
	shuff_uni_resp_array = np.reshape(shuffled_uni_resp, (shape))
	for s in range(len(all_subjects)):
		for r in range(len(r1)):
			# h1h2
			h1h2_base_diff_null_dist[i,s,r,0] = abs(
				shuff_uni_resp_array[s,r1[r],high_1_high_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r1[r],baseline_images[s,r1[r]]]))
			h1h2_base_diff_null_dist[i,s,r,1] = abs(
				shuff_uni_resp_array[s,r2[r],high_1_high_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r2[r],baseline_images[s,r2[r]]]))
			# l1l2
			l1l2_base_diff_null_dist[i,s,r,0] = abs(
				shuff_uni_resp_array[s,r1[r],low_1_low_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r1[r],baseline_images[s,r1[r]]]))
			l1l2_base_diff_null_dist[i,s,r,1] = abs(
				shuff_uni_resp_array[s,r2[r],low_1_low_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r2[r],baseline_images[s,r2[r]]]))
			# h1l2
			h1l2_base_diff_null_dist[i,s,r,0] = abs(
				shuff_uni_resp_array[s,r1[r],high_1_low_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r1[r],baseline_images[s,r1[r]]]))
			h1l2_base_diff_null_dist[i,s,r,1] = abs(
				shuff_uni_resp_array[s,r2[r],high_1_low_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r2[r],baseline_images[s,r2[r]]]))
			# l1h2
			l1h2_base_diff_null_dist[i,s,r,0] = abs(
				shuff_uni_resp_array[s,r1[r],low_1_high_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r1[r],baseline_images[s,r1[r]]]))
			l1h2_base_diff_null_dist[i,s,r,1] = abs(
				shuff_uni_resp_array[s,r2[r],low_1_high_2[s,r]] - \
				np.mean(shuff_uni_resp_array[s,r2[r],baseline_images[s,r2[r]]]))
# Aggregate the absolute baseline difference scores based on categorical
# selectivity group comparisons
# Aggregated absolute baseline difference scores null distribution arrays of shape:
# (Iterations × Subjects × 2 categorical selectivity group comparisons)
sorted_h1h2_base_diff_null_dist = np.zeros((args.n_iter, len(all_subjects),
	len(groups)))
sorted_l1l2_base_diff_null_dist = np.zeros((
	sorted_h1h2_base_diff_null_dist.shape))
sorted_h1l2_base_diff_null_dist = np.zeros((
	sorted_h1h2_base_diff_null_dist.shape))
sorted_l1h2_base_diff_null_dist = np.zeros((
	sorted_h1h2_base_diff_null_dist.shape))
for d, dist in enumerate(groups):
	for s in range(len(all_subjects)):
		# h1h2
		h1h2 = np.reshape(h1h2_base_diff_null_dist[:,s,dist],
			(args.n_iter, -1))
		sorted_h1h2_base_diff_null_dist[:,s,d] = np.mean(h1h2, 1)
		# l1l2
		l1l2 = np.reshape(l1l2_base_diff_null_dist[:,s,dist],
			(args.n_iter, -1))
		sorted_l1l2_base_diff_null_dist[:,s,d] = np.mean(l1l2, 1)
		# h1l2
		h1l2 = np.reshape(h1l2_base_diff_null_dist[:,s,dist],
			(args.n_iter, -1))
		sorted_h1l2_base_diff_null_dist[:,s,d] = np.mean(h1l2, 1)
		# l1h2
		l1h2 = np.reshape(l1h2_base_diff_null_dist[:,s,dist],
			(args.n_iter, -1))
		sorted_l1h2_base_diff_null_dist[:,s,d] = np.mean(l1h2, 1)
# Compute the difference in absolute baseline difference scores between
# categorical selectivity group comparisons
sorted_h1h2_base_diff_diff_null_dist = \
	sorted_h1h2_base_diff_null_dist[:,:,0] - \
	sorted_h1h2_base_diff_null_dist[:,:,1]
sorted_h1l2_base_diff_diff_null_dist = \
	sorted_h1l2_base_diff_null_dist[:,:,0] - \
	sorted_h1l2_base_diff_null_dist[:,:,1]
sorted_l1h2_base_diff_diff_null_dist = \
	sorted_l1h2_base_diff_null_dist[:,:,0] - \
	sorted_l1h2_base_diff_null_dist[:,:,1]
sorted_l1l2_base_diff_diff_null_dist = \
	sorted_l1l2_base_diff_null_dist[:,:,0] - \
	sorted_l1l2_base_diff_null_dist[:,:,1]
# Compute the within-subject p-values
sorted_h1h2_base_diff_within_subject_pval = np.zeros((len(all_subjects)))
sorted_h1l2_base_diff_within_subject_pval = np.zeros((len(all_subjects)))
sorted_l1h2_base_diff_within_subject_pval = np.zeros((len(all_subjects)))
sorted_l1l2_base_diff_within_subject_pval = np.zeros((len(all_subjects)))
for s in range(len(all_subjects)):
	# Test for a significant difference between categorical selectivity group
	# comparisons
	# h1h2
	idx = sum(sorted_h1h2_base_diff_diff_null_dist[:,s] > \
		sorted_h1h2_base_diff_diff[s])
	sorted_h1h2_base_diff_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
	# l1l2
	idx = sum(sorted_l1l2_base_diff_diff_null_dist[:,s] > \
		sorted_l1l2_base_diff_diff[s])
	sorted_l1l2_base_diff_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
	# h1l2
	idx = sum(sorted_h1l2_base_diff_diff_null_dist[:,s] < \
		sorted_h1l2_base_diff_diff[s])
	sorted_h1l2_base_diff_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
	# l1h2
	idx = sum(sorted_l1h2_base_diff_diff_null_dist[:,s] < \
		sorted_l1h2_base_diff_diff[s])
	sorted_l1h2_base_diff_within_subject_pval[s] = (idx + 1) / (args.n_iter + 1) # Add one to avoid p-values of 0
# Benjamini/Hochberg correct the within-subject alphas over:
# 4 control conditions = 4 comparisons
n_control_conditions = 4
# Empty significance variables
sorted_h1h2_base_diff_within_subject_sig = np.zeros((len(all_subjects)))
sorted_l1l2_base_diff_within_subject_sig = np.zeros((len(all_subjects)))
sorted_h1l2_base_diff_within_subject_sig = np.zeros((len(all_subjects)))
sorted_l1h2_base_diff_within_subject_sig = np.zeros((len(all_subjects)))
# Loop across subjects
for s in range(len(all_subjects)):
	# Append the within-subject p-values across the 4 comparisons
	pvals = np.zeros((n_control_conditions))
	pvals[0] = sorted_h1h2_base_diff_within_subject_pval[s]
	pvals[1] = sorted_l1l2_base_diff_within_subject_pval[s]
	pvals[2] = sorted_h1l2_base_diff_within_subject_pval[s]
	pvals[3] = sorted_l1h2_base_diff_within_subject_pval[s]
	# Correct for multiple comparisons
	sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
	# Store the significance scores
	sorted_h1h2_base_diff_within_subject_sig[s] = sig[0]
	sorted_l1l2_base_diff_within_subject_sig[s] = sig[1]
	sorted_h1l2_base_diff_within_subject_sig[s] = sig[2]
	sorted_l1h2_base_diff_within_subject_sig[s] = sig[3]

# Test for a significant difference (between-subject significance)
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.
n = len(all_subjects) # Total number of subjects
p = 0.05 # probability of success in each trial
# h1h2
k = sum(sorted_h1h2_base_diff_within_subject_sig) # Number of significant subjects
# We use "k-1" because otherwise we would get the probability of observing
# k+1 or more significant results by chance
sorted_h1h2_base_diff_between_subject_pval = 1 - binom.cdf(k-1, n, p)
# h1l2
k = sum(sorted_h1l2_base_diff_within_subject_sig) # Number of significant subjects
sorted_h1l2_base_diff_between_subject_pval = 1 - binom.cdf(k-1, n, p)
# l1h2
k = sum(sorted_l1h2_base_diff_within_subject_sig) # Number of significant subjects
sorted_l1h2_base_diff_between_subject_pval = 1 - binom.cdf(k-1, n, p)
# l1l2
k = sum(sorted_l1l2_base_diff_within_subject_sig) # Number of significant subjects
sorted_l1l2_base_diff_between_subject_pval = 1 - binom.cdf(k-1, n, p)


# =============================================================================
# Save the stats
# =============================================================================
stats = {
	'roi_comb': roi_comb,
	'r1': r1,
	'r2': r2,
	'roi_comb_names': roi_comb_names,
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
	'sorted_h1h2_base_diff': sorted_h1h2_base_diff,
	'sorted_l1l2_base_diff': sorted_l1l2_base_diff,
	'sorted_h1l2_base_diff': sorted_h1l2_base_diff,
	'sorted_l1h2_base_diff': sorted_l1h2_base_diff,
	'ci_sorted_h1h2_base_diff': ci_sorted_h1h2_base_diff,
	'ci_sorted_l1l2_base_diff': ci_sorted_l1l2_base_diff,
	'ci_sorted_h1l2_base_diff': ci_sorted_h1l2_base_diff,
	'ci_sorted_l1h2_base_diff': ci_sorted_l1h2_base_diff,
	'sorted_h1h2_base_diff_within_subject_pval': sorted_h1h2_base_diff_within_subject_pval,
	'sorted_l1l2_base_diff_within_subject_pval': sorted_l1l2_base_diff_within_subject_pval,
	'sorted_h1l2_base_diff_within_subject_pval': sorted_h1l2_base_diff_within_subject_pval,
	'sorted_l1h2_base_diff_within_subject_pval': sorted_l1h2_base_diff_within_subject_pval,
	'sorted_h1h2_base_diff_within_subject_sig': sorted_h1h2_base_diff_within_subject_sig,
	'sorted_l1l2_base_diff_within_subject_sig': sorted_l1l2_base_diff_within_subject_sig,
	'sorted_h1l2_base_diff_within_subject_sig': sorted_h1l2_base_diff_within_subject_sig,
	'sorted_l1h2_base_diff_within_subject_sig': sorted_l1h2_base_diff_within_subject_sig,
	'sorted_h1h2_base_diff_between_subject_pval': sorted_h1h2_base_diff_between_subject_pval,
	'sorted_l1l2_base_diff_between_subject_pval': sorted_l1l2_base_diff_between_subject_pval,
	'sorted_h1l2_base_diff_between_subject_pval': sorted_h1l2_base_diff_between_subject_pval,
	'sorted_l1h2_base_diff_between_subject_pval': sorted_l1h2_base_diff_between_subject_pval
	}

save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'categorical_selectivity_analysis', 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'categorical_selectivity_analysis.npy'

np.save(os.path.join(save_dir, file_name), stats)
