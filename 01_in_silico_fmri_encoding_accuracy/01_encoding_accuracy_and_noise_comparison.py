"""Compute the Neural Encoding Dataset (NED) fMRI encoding models encoding
accuracies for areas V1, V2, V3 and V4, and compute stats on them. The encoding
accuracie are calculated using the 515 test images not used for model training.

This code additionally compares the noise of the in-silico fMRI responses (i.e.,
synthetic fMRI responses generated from encoding models) with the noise of the
in-vivo (i.e., target) responses from the NSD experiment, by comparing how
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
https://github.com/gifale95/RNC/01_in_silico_fmri_encoding_accuracy/01_encoding_accuracy_and_noise_comparison.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
rois : list of str
	List of used ROIs.
n_iter : int
	Amount of iterations for creating the confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED

"""

import argparse
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from ned.ned import NED
from scipy.stats import pearsonr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--n_boot_iter', default=100000, type=int)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
args = parser.parse_args()

print('>>> Encoding accuracy and SNR analysis <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize NED object
# =============================================================================
# https://github.com/gifale95/NED
ned_object = NED(args.ned_dir)


# =============================================================================
# Load the synthetic test data
# =============================================================================
explained_variance = {}
explained_variance_gt1tr_synt = {}
explained_variance_gt1tr_gt1tr = {}
explained_variance_gt1tr_gt2tr = {}

for s in tqdm(args.all_subjects, leave=False):
	for r in args.rois:

		# Load the synthetic fMRI responses
		data_dir = os.path.join(args.project_dir, 'synthetic_fmri_responses',
			'imageset-nsd', 'synthetic_fmri_responses_sub-'+format(s, '02')+
                        '_roi-'+r+'.h5')
		synthetic_fmri = h5py.File(data_dir).get('synthetic_fmri_responses')

		# Load the synthetic fMRI responses metadata
		metadata = ned_object.get_metadata(
			modality='fmri',
			train_dataset='nsd',
			model='fwrf',
			subject=s,
			roi=r
			)

		# Select the synthetic fMRI test responses
		test_img_num = metadata['encoding_models']\
			['train_val_test_nsd_image_splits']['test_img_num']
		synthetic_fmri = synthetic_fmri[test_img_num]


# =============================================================================
# Load the ground truth test data
# =============================================================================
		# The ground truth fMRI data was prepared using this code:
		# https://github.com/gifale95/NED/blob/main/ned_creation_code/00_prepare_data/train_dataset-nsd/model-fwrf/prepare_nsd_fmri.py

		betas_dir = os.path.join(args.ned_dir, 'model_training_datasets',
			'train_dataset-nsd', 'model-fwrf', 'neural_data', 'nsd_betas_sub-'+
			format(s,'02')+'_roi-'+r+'.npy')
		betas_dict = np.load(betas_dir, allow_pickle=True).item()

		# target fMRI of shape:
		# (515 Test images x 3 Repetitions x N Voxels)
		gt_fmri = np.zeros((synthetic_fmri.shape[0], 3,
			synthetic_fmri.shape[1]), dtype=np.float32)
		for i, img in enumerate(test_img_num):
			idx = np.where(betas_dict['img_presentation_order'] == img)[0]
			gt_fmri[i] = betas_dict['betas'][idx]
		del betas_dict


# =============================================================================
# Compute the synthetic fMRI responses encoding accuracy
# =============================================================================
		# Correlate the synthetic and ground truth test data
		correlation = np.zeros(synthetic_fmri.shape[1])
		for v in range(len(correlation)):
			correlation[v] = pearsonr(np.mean(gt_fmri[:,:,v], 1),
				synthetic_fmri[:,v])[0]

		# Convert the ncsnr to noise ceiling
		ncsnr = metadata['fmri']['ncsnr']
		norm_term = (len(synthetic_fmri) / 3) / len(synthetic_fmri)
		noise_ceil = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)

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
# Correlate synthetic and target fMRI for the noise analysis
# =============================================================================
		# Correlate target fMRI single trials with other single trials, and with
		# synthetic fRMI responses
		comparisons = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
		corr_gt1tr_synt = np.zeros((len(comparisons), synthetic_fmri.shape[1]))
		corr_gt1tr_gt1tr = np.zeros((len(comparisons), 2,
			synthetic_fmri.shape[1]))
		for c, comp in enumerate(comparisons):
			for v in range(synthetic_fmri.shape[1]):
				corr_gt1tr_synt[c,v] = pearsonr(gt_fmri[:,comp[0],v],
					synthetic_fmri[:,v])[0]
				corr_gt1tr_gt1tr[c,0,v] = pearsonr(gt_fmri[:,comp[0],v],
					gt_fmri[:,comp[1],v])[0]
				corr_gt1tr_gt1tr[c,1,v] = pearsonr(gt_fmri[:,comp[0],v],
					gt_fmri[:,comp[2],v])[0]

		# Correlate target fMRI single trials with the average of the two
		# two other trials
		corr_gt1tr_gt2tr = np.zeros((len(comparisons), synthetic_fmri.shape[1]))
		for c, comp in enumerate(comparisons):
			for v in range(synthetic_fmri.shape[1]):
				corr_gt1tr_gt2tr[c,v] = pearsonr(gt_fmri[:,comp[0],v],
					np.mean(gt_fmri[:,comp[1:],v], 1))[0]


# =============================================================================
# Compute the noise-ceiling-normalized encoding accuracy for the noise analysis
# =============================================================================
		# Convert the ncsnr to noise ceiling
		ncsnr = metadata['fmri']['ncsnr']
		norm_term = (len(synthetic_fmri) / 1) / len(synthetic_fmri)
		noise_ceil = (ncsnr ** 2) / ((ncsnr ** 2) + norm_term)

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

		del metadata, gt_fmri, synthetic_fmri


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
	sample_dist = np.zeros(args.n_boot_iter)
	sample_dist_gt1tr_synt = np.zeros(args.n_boot_iter)
	sample_dist_gt1tr_gt1tr = np.zeros(args.n_boot_iter)
	sample_dist_gt1tr_gt2tr = np.zeros(args.n_boot_iter)
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
	for i in range(args.n_boot_iter):
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
# t-tests & multiple comparisons correction
# =============================================================================
p_values = {}
significance = {}
p_values_mat = np.zeros((len(args.rois)))
for r, roi in enumerate(args.rois):
	_, p_values_mat[r] = ttest_1samp(mean_encoding_acc, 0,
		alternative='greater')
sig, p_val, _, _ = multipletests(p_values_mat, 0.05, 'fdr_bh')
for r, roi in enumerate(args.rois):
	significance[roi] = sig[r]
	p_values[roi] = p_val[r]

significance_gt1tr_gt1tr_vs_gt1tr_gt2tr = {}
significance_gt1tr_gt2tr_vs_gt1tr_synt = {}
p_values_gt1tr_gt1tr_vs_gt1tr_gt2tr = np.zeros((len(args.rois)))
p_values_gt1tr_gt2tr_vs_gt1tr_synt = np.zeros((len(args.rois)))
for r, roi in enumerate(args.rois):
	_, p_values_gt1tr_gt1tr_vs_gt1tr_gt2tr[r] = ttest_rel(
		sample_dist_gt1tr_gt2tr, sample_dist_gt1tr_gt1tr, alternative='greater')
	_, p_values_gt1tr_gt2tr_vs_gt1tr_synt[r] = ttest_rel(
		sample_dist_gt1tr_synt, sample_dist_gt1tr_gt2tr, alternative='greater')
sig_gt1tr_gt1tr_vs_gt1tr_gt2tr, _, _, _ = multipletests(
	p_values_gt1tr_gt1tr_vs_gt1tr_gt2tr, 0.05, 'fdr_bh')
sig_gt1tr_gt2tr_vs_gt1tr_synt, _, _, _ = multipletests(
	p_values_gt1tr_gt2tr_vs_gt1tr_synt, 0.05, 'fdr_bh')
for r, roi in enumerate(args.rois):
	significance_gt1tr_gt1tr_vs_gt1tr_gt2tr[roi] = sig_gt1tr_gt1tr_vs_gt1tr_gt2tr[r]
	significance_gt1tr_gt2tr_vs_gt1tr_synt[roi] = sig_gt1tr_gt2tr_vs_gt1tr_synt[r]


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
	'p_values': p_values,
	'significance': significance,
	'significance_gt1tr_gt1tr_vs_gt1tr_gt2tr': significance_gt1tr_gt1tr_vs_gt1tr_gt2tr,
	'significance_gt1tr_gt2tr_vs_gt1tr_synt': significance_gt1tr_gt2tr_vs_gt1tr_synt
}

save_dir = os.path.join(args.project_dir, 'encoding_accuracy')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy.npy'

np.save(os.path.join(save_dir, file_name), results)

