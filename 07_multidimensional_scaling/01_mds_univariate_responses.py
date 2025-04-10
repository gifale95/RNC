"""Perform MDS on the in silico univariate fMRI responses of all ROIs for all
images, or for the controlling images. The ROIs are used as samples, and the
images as features. MDS's goal is to reduce the images to two dimensions, such
that each ROI can be plotted in 2D space based on its univariate response
similarity to other ROIs.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
ncsnr_threshold : float
	Lower bound ncsnr threshold of the kept voxels: only voxels above this
	threshold are used.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--ncsnr_threshold', type=float, default=0.5)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> MDS univariate responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# ROI selection
# =============================================================================
rois = ['V1', 'V2', 'V3', 'hV4', 'EBA', 'FFA', 'PPA', 'RSC']

roi_comb_names = [
	'V1-V2', 'V1-V3', 'V1-hV4', 'V2-V3', 'V2-hV4', 'V3-hV4',
	'EBA-FFA', 'EBA-PPA', 'EBA-RSC', 'FFA-PPA', 'FFA-RSC', 'PPA-RSC',
	'V1-EBA', 'V1-FFA', 'V1-PPA', 'V1-RSC',
	'V2-EBA', 'V2-FFA', 'V2-PPA', 'V2-RSC',
	'V3-EBA', 'V3-FFA', 'V3-PPA', 'V3-RSC',
	'hV4-EBA', 'hV4-FFA', 'hV4-PPA', 'hV4-RSC'
	]

roi_comb = [
	(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
	(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
	(0, 4), (0, 5), (0, 6), (0, 7),
	(1, 4), (1, 5), (1, 6), (1, 7),
	(2, 4), (2, 5), (2, 6), (2, 7),
	(3, 4), (3, 5), (3, 6), (3, 7)
	]


# =============================================================================
# Load the controlling image indices
# =============================================================================
high_1_high_2 = {}
high_1_low_2 = {}
low_1_high_2 = {}
low_1_low_2 = {}

for roi_pair in roi_comb_names:

	data_dir = os.path.join(args.project_dir, 'univariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-0',
		'imageset-'+args.imageset, roi_pair, 'stats.npy')
	stats = np.load(data_dir, allow_pickle=True).item()

	high_1_high_2[roi_pair] = stats['high_1_high_2']
	high_1_low_2[roi_pair] = stats['high_1_low_2']
	low_1_high_2[roi_pair] = stats['low_1_high_2']
	low_1_low_2[roi_pair] = stats['low_1_low_2']


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
uni_resp = np.zeros((len(rois), len(all_subjects), images),
	dtype=np.float32)

for r, roi in enumerate(rois):
	for s, sub in enumerate(all_subjects):

		# Load the in silico fMRI responses
		data_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri', 'imageset-'+args.imageset,
			'insilico_fmri_responses_sub-'+format(sub, '02')+'_roi-'+roi+'.h5')
		betas = h5py.File(data_dir).get('insilico_fmri_responses')

		# Load the ncsnr
		ncsnr_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
			args.encoding_models_train_dataset+'_encoding_models',
			'insilico_fmri', 'ncsnr_sub-'+format(sub, '02')+'_roi-'+roi+'.npy')
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
		# Average the fMRI activity across voxels (there might be NaN values
		# since some subjects have missing data)
		uni_resp[r,s] = np.nanmean(betas, 1)

# Average the univariate responses across subjects
uni_resp = np.mean(uni_resp, 1)


# =============================================================================
# Perform MDS using all image conditions
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)

mds_all_images = embedding.fit_transform(uni_resp)


# =============================================================================
# Perform MDS using only the controlling images
# =============================================================================
mds_h1h2 = {}
mds_h1l2 = {}
mds_l1h2 = {}
mds_l1l2 = {}

for roi_pair in roi_comb_names:

	# h1h2
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_h1h2[roi_pair] = embedding.fit_transform(
		uni_resp[:,high_1_high_2[roi_pair]])

	# h1l2
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_h1l2[roi_pair] = embedding.fit_transform(
		uni_resp[:,high_1_low_2[roi_pair]])

	# l1h2
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_l1h2[roi_pair] = embedding.fit_transform(
		uni_resp[:,low_1_high_2[roi_pair]])

	# l1l2
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_l1l2[roi_pair] = embedding.fit_transform(
		uni_resp[:,low_1_low_2[roi_pair]])


# =============================================================================
# Save the results
# =============================================================================
results = {
	'mds_all_images': mds_all_images,
	'mds_h1h2': mds_h1h2,
	'mds_h1l2': mds_h1l2,
	'mds_l1h2': mds_l1h2,
	'mds_l1l2': mds_l1l2,
	'roi_comb_names': roi_comb_names,
	'roi_comb': roi_comb
	}

save_dir = os.path.join(args.project_dir, 'multidimensional_scaling',
	args.encoding_models_train_dataset+'_encoding_models', 'imageset-'+
	args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'mds_univariate_responses.npy'

np.save(os.path.join(save_dir, file_name), results)
