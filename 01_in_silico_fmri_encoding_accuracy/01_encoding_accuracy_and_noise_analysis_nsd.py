"""Compute the Neural Encoding Simulation Toolkit (NEST) fMRI encoding models
in-distribution (ID) encoding accuracy, using NSD-core's 515 test images not
used for model training.

This code additionally compares the noise of the in silico fMRI responses
(i.e., the fMRI responses generated from encoding models) with the noise of the
in vivo (i.e., target) responses from the NSD experiment, by comparing how
much variance can these two data types explain for a third, independent split of
NSD responses.

Because the in-silico neural responses did not capture all signal variance for
the in-vivo NSD responses, the in-silico neural responses explaining more
variance than NSD’s in-vivo responses would be indicative of the former not
being affected by noise.

The comparison is carried out through three predictions, using the in-silico and
in-vivo fMRI responses for the 515 test images. Each prediction involves
explaining single NSD in-vivo response trials with a different predictor.
The first predictor consists of the two remaining NSD in-vivo response trials,
each used independently. The evaluation is repeated until each of the three
trials is used as the target to be explained and the remaining two trials as
separate predictors, and the explained variance scores from the different
evaluations (N = 6 evaluations) are then averaged.
The second predictor consists of the average of the two remaining NSD in-vivo
response trials. The evaluation is repeated until each of the three trials is
used as the target to be explained and the average of the remaining two trials
as predictor, and the explained variance scores from the different evaluations
(N = 3 evaluations) are then averaged.
The third predictor consists of the in-silico responses from the trained
encoding models. The evaluation is repeated until each of the three trials is
used as the target to be explained by the same in-silico responses, and the
explained variance scores from the different evaluations (N = 3 evaluations) is
then averaged.
These comparisons are carried out independently for each voxel and subject, and
the results averaged across voxels belonging to the same subject and ROI.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
rois : list of str
	List of used ROIs.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
n_iter : int
	Amount of iterations for the permutation stats.
project_dir : str
	Directory of the project folder.
nest_dir : str
	Directory of the Neural Encoding Simulation Toolkit.
	https://github.com/gifale95/NEST

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
from nest.nest import NEST
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4', 'FFA', 'PPA', 'RSC', 'EBA'])
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--nest_dir', default='../neural_encoding_simulation_toolkit/', type=str)
args = parser.parse_args()

print('>>> Encoding accuracy and SNR analysis - NSD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize the NEST object
# =============================================================================
# https://github.com/gifale95/NEST
nest_object = NEST(args.nest_dir)


# =============================================================================
# Load the in silico fMRI test responses
# =============================================================================
explained_variance = {}
explained_variance_gt1tr_synt = {}
explained_variance_gt1tr_gt1tr = {}
explained_variance_gt1tr_gt2tr = {}
gt1tr_synt_minus_gt1tr_gt2tr_within_subject_pval = {}
gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_pval = {}
gt1tr_synt_minus_gt1tr_gt2tr_within_subject_sig = {}
gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_sig = {}

for s in tqdm(args.all_subjects, leave=False):
	for r in args.rois:

		# Load the in silico fMRI responses
		data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'nsd_encoding_models', 'insilico_fmri', 'imageset-nsd',
			'insilico_fmri_responses_sub-'+format(s, '02')+'_roi-'+r+'.h5')
		insilico_fmri = h5py.File(data_dir).get('insilico_fmri_responses')

		# Load the in silico fMRI responses metadata
		if r in ['FFA']:
			metadata_1 = nest_object.get_metadata(
				modality='fmri',
				train_dataset='nsd',
				model='fwrf',
				subject=s,
				roi=r+'-1'
				)
			metadata_2 = nest_object.get_metadata(
				modality='fmri',
				train_dataset='nsd',
				model='fwrf',
				subject=s,
				roi=r+'-2'
				)
		else:
			metadata = nest_object.get_metadata(
				modality='fmri',
				train_dataset='nsd',
				model='fwrf',
				subject=s,
				roi=r
				)

		# Select the in silico fMRI test responses
		if r in ['FFA']:
			test_img_num = metadata_1['encoding_models']\
				['train_val_test_nsd_image_splits']['test_img_num']
		else:
			test_img_num = metadata['encoding_models']\
				['train_val_test_nsd_image_splits']['test_img_num']
		insilico_fmri = insilico_fmri[test_img_num]


# =============================================================================
# Load the ground truth test data
# =============================================================================
		# The ground truth fMRI data was prepared using this code:
		# https://github.com/gifale95/NEST/blob/main/nest_creation_code/00_prepare_data/train_dataset-nsd/model-fwrf/prepare_nsd_fmri.py

		if r in ['FFA']:
			betas_dir_1 = os.path.join(args.nest_dir, 'model_training_datasets',
				'train_dataset-nsd', 'model-fwrf', 'neural_data',
				'nsd_betas_sub-'+format(s,'02')+'_roi-'+r+'-1.npy')
			betas_dir_2 = os.path.join(args.nest_dir, 'model_training_datasets',
				'train_dataset-nsd', 'model-fwrf', 'neural_data',
				'nsd_betas_sub-'+format(s,'02')+'_roi-'+r+'-2.npy')
			betas_dict_1 = np.load(betas_dir_1, allow_pickle=True).item()
			betas_dict_2 = np.load(betas_dir_2, allow_pickle=True).item()
			# Target fMRI of shape:
			# (515 Test images x 3 Repetitions x N Voxels)
			gt_fmri = np.zeros((insilico_fmri.shape[0], 3,
				insilico_fmri.shape[1]), dtype=np.float32)
			for i, img in enumerate(test_img_num):
				idx = np.where(betas_dict_1['img_presentation_order'] == img)[0]
				gt_fmri[i] = np.append(betas_dict_1['betas'][idx],
					betas_dict_2['betas'][idx], 1)
			del betas_dict_1, betas_dict_2
		else:
			betas_dir = os.path.join(args.nest_dir, 'model_training_datasets',
				'train_dataset-nsd', 'model-fwrf', 'neural_data',
				'nsd_betas_sub-'+format(s,'02')+'_roi-'+r+'.npy')
			betas_dict = np.load(betas_dir, allow_pickle=True).item()
			# Target fMRI of shape:
			# (515 Test images x 3 Repetitions x N Voxels)
			gt_fmri = np.zeros((insilico_fmri.shape[0], 3,
				insilico_fmri.shape[1]), dtype=np.float32)
			for i, img in enumerate(test_img_num):
				idx = np.where(betas_dict['img_presentation_order'] == img)[0]
				gt_fmri[i] = betas_dict['betas'][idx]
			del betas_dict


# =============================================================================
# Voxel selection
# =============================================================================
		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			'nsd_encoding_models', 'insilico_fmri', 'ncsnr_sub-'+
			format(s, '02')+'_roi-'+r+'.npy')
		ncsnr = np.load(ncsnr_dir)

		# Only retain voxels with noise ceiling signal-to-noise ratio scores
		# above the selected threshold.
		best_voxels = np.where(ncsnr > args.ncsnr_threshold)[0]
		insilico_fmri = insilico_fmri[:,best_voxels]
		gt_fmri = gt_fmri[:,:,best_voxels]


# =============================================================================
# Compute the in silico fMRI responses encoding accuracy
# =============================================================================
		# Correlate the insilico and ground truth test fMRI responses
		correlation = np.zeros(insilico_fmri.shape[1])
		for v in range(len(correlation)):
			correlation[v] = pearsonr(np.mean(gt_fmri[:,:,v], 1),
				insilico_fmri[:,v])[0]

		# Convert the ncsnr to noise ceiling
		if r in ['FFA']:	
			ncsnr = np.append(metadata_1['fmri']['ncsnr'],
				metadata_2['fmri']['ncsnr'])
		else:
			ncsnr = metadata['fmri']['ncsnr']
		norm_term = (len(insilico_fmri) / 3) / len(insilico_fmri)
		noise_ceil = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)
		noise_ceil = noise_ceil[best_voxels]

		# Set negative correlation values to 0, so to keep the
		# noise-normalized encoding accuracy positive
		correlation[correlation<0] = 0

		# Square the correlation values
		r2 = correlation ** 2

		# Add a very small number to noise ceiling values of 0, otherwise
		# the noise-normalized encoding accuracy cannot be calculated
		# (division by 0 is not possible)
		noise_ceil[noise_ceil==0] = 1e-14

		# Compute the noise-normalized encoding accuracy
		expl_val = np.divide(r2, noise_ceil)

		# Set the noise-normalized encoding accuracy to 1 for those
		# vertices in which the correlation is higher than the noise
		# ceiling, to prevent encoding accuracy values higher than 100%
		expl_val[expl_val>1] = 1

		# Store the explained variance scores
		explained_variance['s'+str(s)+'_'+r] = np.mean(expl_val)


# =============================================================================
# Correlate in silico and target fMRI for the noise analysis
# =============================================================================
		# Correlate target fMRI single trials with other single trials, and
		# with in silico fRMI responses
		comparisons = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
		corr_gt1tr_synt = np.zeros((len(comparisons), insilico_fmri.shape[1]))
		corr_gt1tr_gt1tr = np.zeros((len(comparisons), 2,
			insilico_fmri.shape[1]))
		for c, comp in enumerate(comparisons):
			for v in range(insilico_fmri.shape[1]):
				corr_gt1tr_synt[c,v] = pearsonr(gt_fmri[:,comp[0],v],
					insilico_fmri[:,v])[0]
				corr_gt1tr_gt1tr[c,0,v] = pearsonr(gt_fmri[:,comp[0],v],
					gt_fmri[:,comp[1],v])[0]
				corr_gt1tr_gt1tr[c,1,v] = pearsonr(gt_fmri[:,comp[0],v],
					gt_fmri[:,comp[2],v])[0]

		# Correlate target fMRI single trials with the average of the two
		# two other trials
		corr_gt1tr_gt2tr = np.zeros((len(comparisons), insilico_fmri.shape[1]))
		for c, comp in enumerate(comparisons):
			for v in range(insilico_fmri.shape[1]):
				corr_gt1tr_gt2tr[c,v] = pearsonr(gt_fmri[:,comp[0],v],
					np.mean(gt_fmri[:,comp[1:],v], 1))[0]


# =============================================================================
# Compute the noise-ceiling-normalized encoding accuracy for the noise analysis
# =============================================================================
		# Convert the ncsnr to noise ceiling
		if r in ['FFA']:
			ncsnr = np.append(metadata_1['fmri']['ncsnr'],
				metadata_2['fmri']['ncsnr'])
		else:
			ncsnr = metadata['fmri']['ncsnr']
		norm_term = (len(insilico_fmri) / 1) / len(insilico_fmri)
		noise_ceil = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)
		noise_ceil = noise_ceil[best_voxels]

		# Set negative correlation values to 0, so to keep the
		# noise-normalized encoding accuracy positive
		corr_gt1tr_synt[corr_gt1tr_synt<0] = 0
		corr_gt1tr_gt1tr[corr_gt1tr_gt1tr<0] = 0
		corr_gt1tr_gt2tr[corr_gt1tr_gt2tr<0] = 0

		# Square the correlation values
		r2_corr_gt1tr_synt = corr_gt1tr_synt ** 2
		r2_corr_gt1tr_gt1tr = corr_gt1tr_gt1tr ** 2
		r2_corr_gt1tr_gt2tr = corr_gt1tr_gt2tr ** 2

		# Add a very small number to noise ceiling values of 0, otherwise
		# the noise-normalized encoding accuracy cannot be calculated
		# (division by 0 is not possible)
		noise_ceil[noise_ceil==0] = 1e-14

		# Compute the noise-normalized encoding accuracy
		expl_var_gt1tr_synt = np.divide(r2_corr_gt1tr_synt, noise_ceil)
		expl_var_gt1tr_gt1tr = np.divide(r2_corr_gt1tr_gt1tr, noise_ceil)
		expl_var_gt1tr_gt2tr = np.divide(r2_corr_gt1tr_gt2tr, noise_ceil)

		# Set the noise-normalized encoding accuracy to 1 for those
		# vertices in which the correlation is higher than the noise
		# ceiling, to prevent encoding accuracy values higher than 100%
		expl_var_gt1tr_synt[expl_var_gt1tr_synt>1] = 1
		expl_var_gt1tr_gt1tr[expl_var_gt1tr_gt1tr>1] = 1
		expl_var_gt1tr_gt2tr[expl_var_gt1tr_gt2tr>1] = 1

		# Store the explained variance scores
		explained_variance_gt1tr_synt['s'+str(s)+'_'+r] = \
			np.mean(expl_var_gt1tr_synt)
		explained_variance_gt1tr_gt1tr['s'+str(s)+'_'+r] = \
			np.mean(expl_var_gt1tr_gt1tr)
		explained_variance_gt1tr_gt2tr['s'+str(s)+'_'+r] = \
			np.mean(expl_var_gt1tr_gt2tr)

		# Delete unused variables
		if r in ['FFA']:
			del metadata_1, metadata_2, gt_fmri, insilico_fmri
		else:
			del metadata, gt_fmri, insilico_fmri


# =============================================================================
# Compute the within-subject significance
# =============================================================================
		# Compute the difference in encoding accuracy between the noise analysis
		# conditions
		gt1tr_synt_minus_gt1tr_gt2tr = np.mean(expl_var_gt1tr_synt) - \
			np.mean(expl_var_gt1tr_gt2tr)
		gt1tr_gt2tr_minus_gt1tr_gt1tr = np.mean(expl_var_gt1tr_gt2tr) - \
			np.mean(expl_var_gt1tr_gt1tr)

		# Create the permutation-based null distributions
		scores_gt1tr_synt_minus_gt1tr_gt2tr = np.append(expl_var_gt1tr_synt,
			expl_var_gt1tr_gt2tr)
		scores_gt1tr_gt2tr_minus_gt1tr_gt1tr = np.append(expl_var_gt1tr_gt2tr,
			expl_var_gt1tr_gt1tr)
		gt1tr_synt_minus_gt1tr_gt2tr_null_dist = np.zeros((args.n_iter))
		gt1tr_gt2tr_minus_gt1tr_gt1tr_null_dist = np.zeros((args.n_iter))
		idx = len(scores_gt1tr_synt_minus_gt1tr_gt2tr) // 2
		# At each iteration, the scores from the noise analysis comparisons are
		# shuffled
		for i in range(args.n_iter):
			np.random.shuffle(scores_gt1tr_synt_minus_gt1tr_gt2tr)
			np.random.shuffle(scores_gt1tr_gt2tr_minus_gt1tr_gt1tr)
			gt1tr_synt_minus_gt1tr_gt2tr_null_dist[i] = \
				np.mean(scores_gt1tr_synt_minus_gt1tr_gt2tr[:idx]) - \
				np.mean(scores_gt1tr_synt_minus_gt1tr_gt2tr[idx:])
			gt1tr_gt2tr_minus_gt1tr_gt1tr_null_dist[i] = \
				np.mean(scores_gt1tr_gt2tr_minus_gt1tr_gt1tr[:idx]) - \
				np.mean(scores_gt1tr_gt2tr_minus_gt1tr_gt1tr[idx:])

		# Compute the within-subject p-values
		# gt1tr_synt minus gt1tr_gt2tr
		idx = sum(gt1tr_synt_minus_gt1tr_gt2tr_null_dist > \
			gt1tr_synt_minus_gt1tr_gt2tr)
		pval_gt1tr_synt_minus_gt1tr_gt2tr = (idx + 1) / (args.n_iter + 1)
		gt1tr_synt_minus_gt1tr_gt2tr_within_subject_pval['s'+str(s)+'_'+r] = \
			pval_gt1tr_synt_minus_gt1tr_gt2tr
		# gt1tr_gt2tr minus gt1tr_gt1tr
		idx = sum(gt1tr_gt2tr_minus_gt1tr_gt1tr_null_dist > \
			gt1tr_gt2tr_minus_gt1tr_gt1tr)
		pval_gt1tr_gt2tr_minus_gt1tr_gt1tr = (idx + 1) / (args.n_iter + 1)
		gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_pval['s'+str(s)+'_'+r] = \
			pval_gt1tr_gt2tr_minus_gt1tr_gt1tr

		# Benjamini/Hochberg correct the within-subject alphas over:
		# 2 noise analysis conditions = 2 comparisons
		# Append the within-subject p-values across the 2 comparisons
		pvals = np.append(pval_gt1tr_synt_minus_gt1tr_gt2tr,
			pval_gt1tr_gt2tr_minus_gt1tr_gt1tr)
		# Correct for multiple comparisons
		sig, _, _, _ = multipletests(pvals, 0.05, 'fdr_bh')
		# Store the significance scores
		gt1tr_synt_minus_gt1tr_gt2tr_within_subject_sig['s'+str(s)+'_'+r] = \
			sig[0]
		gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_sig['s'+str(s)+'_'+r] = \
			sig[1]


# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
# Random seeds
seed = 20200220
random.seed(seed)
np.random.seed(seed)

ci_lower = {}
ci_upper = {}
ci_lower_gt1tr_synt = {}
ci_upper_gt1tr_synt = {}
ci_lower_gt1tr_gt1tr = {}
ci_upper_gt1tr_gt1tr = {}
ci_lower_gt1tr_gt2tr = {}
ci_upper_gt1tr_gt2tr = {}
for r in tqdm(args.rois):
	sample_dist = np.zeros(args.n_iter)
	sample_dist_gt1tr_synt = np.zeros(args.n_iter)
	sample_dist_gt1tr_gt1tr = np.zeros(args.n_iter)
	sample_dist_gt1tr_gt2tr = np.zeros(args.n_iter)
	mean_encoding_acc = []
	mean_encoding_acc_gt1tr_synt = []
	mean_encoding_acc_gt1tr_gt1tr = []
	mean_encoding_acc_gt1tr_gt2tr = []
	for s in args.all_subjects:
		mean_encoding_acc.append(np.mean(explained_variance['s'+str(s)+'_'+r]))
		mean_encoding_acc_gt1tr_synt.append(
			explained_variance_gt1tr_synt['s'+str(s)+'_'+r])
		mean_encoding_acc_gt1tr_gt1tr.append(
			explained_variance_gt1tr_gt1tr['s'+str(s)+'_'+r])
		mean_encoding_acc_gt1tr_gt2tr.append(
			explained_variance_gt1tr_gt2tr['s'+str(s)+'_'+r])
	mean_encoding_acc = np.asarray(mean_encoding_acc)
	mean_encoding_acc_gt1tr_synt = np.asarray(mean_encoding_acc_gt1tr_synt)
	mean_encoding_acc_gt1tr_gt1tr = np.asarray(mean_encoding_acc_gt1tr_gt1tr)
	mean_encoding_acc_gt1tr_gt2tr = np.asarray(mean_encoding_acc_gt1tr_gt2tr)
	for i in range(args.n_iter):
		sample_dist[i] = np.mean(resample(mean_encoding_acc))
		sample_dist_gt1tr_synt[i] = np.mean(resample(
			mean_encoding_acc_gt1tr_synt))
		sample_dist_gt1tr_gt1tr[i] = np.mean(resample(
			mean_encoding_acc_gt1tr_gt1tr))
		sample_dist_gt1tr_gt2tr[i] = np.mean(resample(
			mean_encoding_acc_gt1tr_gt2tr))
	ci_lower[r] = np.percentile(sample_dist, 2.5)
	ci_upper[r] = np.percentile(sample_dist, 97.5)
	ci_lower_gt1tr_synt[r] = np.percentile(sample_dist_gt1tr_synt, 2.5)
	ci_upper_gt1tr_synt[r] = np.percentile(sample_dist_gt1tr_synt, 97.5)
	ci_lower_gt1tr_gt1tr[r] = np.percentile(sample_dist_gt1tr_gt1tr, 2.5)
	ci_upper_gt1tr_gt1tr[r] = np.percentile(sample_dist_gt1tr_gt1tr, 97.5)
	ci_lower_gt1tr_gt2tr[r] = np.percentile(sample_dist_gt1tr_gt2tr, 2.5)
	ci_upper_gt1tr_gt2tr[r] = np.percentile(sample_dist_gt1tr_gt2tr, 97.5)


# =============================================================================
# Compute the between-subject significance
# =============================================================================
# Compute the probability of observing k or more significant results by chance,
# based on the CDF of the binomial distribution of within-subject significances.

n = len(args.all_subjects) # Total number of subjects
p = 0.05 # probability of success in each trial

gt1tr_synt_minus_gt1tr_gt2tr_between_subject_pval = {}
gt1tr_gt2tr_minus_gt1tr_gt1tr_between_subject_pval = {}

for r in args.rois:

	# Aggregate the within-subject significances
	sig_gt1tr_synt_minus_gt1tr_gt2tr = np.zeros((len(args.all_subjects)),
		dtype=int)
	sig_gt1tr_gt2tr_minus_gt1tr_gt1tr = np.zeros((len(args.all_subjects)),
		dtype=int)
	for s, sub in enumerate(args.all_subjects):
		sig_gt1tr_synt_minus_gt1tr_gt2tr[s] = \
			gt1tr_synt_minus_gt1tr_gt2tr_within_subject_sig['s'+str(sub)+'_'+r]
		sig_gt1tr_gt2tr_minus_gt1tr_gt1tr[s] = \
			gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_sig['s'+str(sub)+'_'+r]

	# gt1tr_synt minus gt1tr_gt2tr
	k = sum(sig_gt1tr_synt_minus_gt1tr_gt2tr) # Number of significant subjects
	# We use "k-1" because otherwise we would get the probability of observing
	# k+1 or more significant results by chance
	gt1tr_synt_minus_gt1tr_gt2tr_between_subject_pval[r] = \
		1 - binom.cdf(k-1, n, p)

	# gt1tr_gt2tr minus gt1tr_gt1tr
	k = sum(sig_gt1tr_gt2tr_minus_gt1tr_gt1tr) # Number of significant subjects
	gt1tr_gt2tr_minus_gt1tr_gt1tr_between_subject_pval[r] = \
		1 - binom.cdf(k-1, n, p)


# =============================================================================
# Save the results
# =============================================================================
results = {
	'explained_variance': explained_variance,
	'explained_variance_gt1tr_synt': explained_variance_gt1tr_synt,
	'explained_variance_gt1tr_gt1tr': explained_variance_gt1tr_gt1tr,
	'explained_variance_gt1tr_gt2tr': explained_variance_gt1tr_gt2tr,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'ci_lower_gt1tr_synt': ci_lower_gt1tr_synt,
	'ci_upper_gt1tr_synt': ci_upper_gt1tr_synt,
	'ci_lower_gt1tr_gt1tr': ci_lower_gt1tr_gt1tr,
	'ci_upper_gt1tr_gt1tr': ci_upper_gt1tr_gt1tr,
	'ci_lower_gt1tr_gt2tr': ci_lower_gt1tr_gt2tr,
	'ci_upper_gt1tr_gt2tr': ci_upper_gt1tr_gt2tr,
	'gt1tr_synt_minus_gt1tr_gt2tr_within_subject_pval': gt1tr_synt_minus_gt1tr_gt2tr_within_subject_pval,
	'gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_pval': gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_pval,
	'gt1tr_synt_minus_gt1tr_gt2tr_within_subject_sig': gt1tr_synt_minus_gt1tr_gt2tr_within_subject_sig,
	'gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_sig': gt1tr_gt2tr_minus_gt1tr_gt1tr_within_subject_sig,
	'gt1tr_synt_minus_gt1tr_gt2tr_between_subject_pval': gt1tr_synt_minus_gt1tr_gt2tr_between_subject_pval,
	'gt1tr_gt2tr_minus_gt1tr_gt1tr_between_subject_pval': gt1tr_gt2tr_minus_gt1tr_gt1tr_between_subject_pval
}

save_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'nsd_encoding_models')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)
