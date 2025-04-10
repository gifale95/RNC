"""Perform MDS on the in silico fMRI RSMs of all ROIs for all images, or for the
controlling images. The ROIs are used as samples, and the RSM entries as
features. MDS's goal is to reduce the RSM entries to two dimensions, such that
each ROI can be plotted in 2D space based on its multivariate response
similarity to other ROIs.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.manifold import MDS

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> MDS multivariate responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


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
align = {}
disentangle = {}

for roi_pair in roi_comb_names:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-0',
		'imageset-'+args.imageset, roi_pair, 'stats.npy')
	stats = np.load(data_dir, allow_pickle=True).item()

	align[roi_pair] = stats['best_generation_image_batches']['align'][-1]
	disentangle[roi_pair] = stats['best_generation_image_batches']\
		['disentangle'][-1]


# =============================================================================
# Load the RSMs averaged across subjects
# =============================================================================
# Get the image conditions number
if args.imageset == 'nsd':
	img_cond = 73000
elif args.imageset == 'imagenet_val':
	img_cond = 50000
elif args.imageset == 'things':
	img_cond = 26107

rsms_all_images = []
rsms_align = {}
rsms_disentangle = {}
idx_lower_tr_all = np.tril_indices(img_cond, -1)

for r, roi in enumerate(rois):

	# Load the in silico fMRI RSMs
	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'rsms',
		'imageset-'+args.imageset, 'averaged_rsm_'+roi+'_all_subjects.npy')
	rsm = np.load(data_dir)

	# Store the RSMs for all image conditions
	rsms_all_images.append(rsm[idx_lower_tr_all])

	# Store the RSMs for the controlling images
	for roi_pair in roi_comb_names:
		if r == 0:
			rsms_align[roi_pair] = []
			rsms_disentangle[roi_pair] = []
		idx_lower_tr_cond = np.tril_indices(len(align[roi_pair]), -1)
		# Align
		rsm_cond = rsm[align[roi_pair]]
		rsm_cond = rsm_cond[:,align[roi_pair]]
		rsm_cond = rsm_cond[idx_lower_tr_cond]
		rsms_align[roi_pair].append(rsm_cond)
		# Disentangle
		rsm_cond = rsm[disentangle[roi_pair]]
		rsm_cond = rsm_cond[:,disentangle[roi_pair]]
		rsm_cond = rsm_cond[idx_lower_tr_cond]
		rsms_disentangle[roi_pair].append(rsm_cond)
	del rsm, rsm_cond

# Reformat to numpy
rsms_all_images = np.asarray(rsms_all_images)
for roi_pair in roi_comb_names:
	rsms_align[roi_pair] = np.asarray(rsms_align[roi_pair])
	rsms_disentangle[roi_pair] = np.asarray(rsms_disentangle[roi_pair])


# =============================================================================
# Perform MDS using all image conditions
# =============================================================================
embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)

mds_all_images = embedding.fit_transform(rsms_all_images)


# =============================================================================
# Perform MDS using only the controlling images
# =============================================================================
mds_align = {}
mds_disentangle = {}

for roi_pair in roi_comb_names:

	# Align
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_align[roi_pair] = embedding.fit_transform(rsms_align[roi_pair])

	# Disentangle
	embedding = MDS(n_components=2, n_init=10, max_iter=1000, random_state=seed)
	mds_disentangle[roi_pair] = embedding.fit_transform(
		rsms_disentangle[roi_pair])


# =============================================================================
# Save the results
# =============================================================================
results = {
	'mds_all_images': mds_all_images,
	'mds_align': mds_align,
	'mds_disentangle': mds_disentangle,
	'roi_comb_names': roi_comb_names,
	'roi_comb': roi_comb
	}

save_dir = os.path.join(args.project_dir, 'multidimensional_scaling',
	args.encoding_models_train_dataset+'_encoding_models', 'imageset-'+
	args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'mds_multivariate_responses.npy'

np.save(os.path.join(save_dir, file_name), results)
