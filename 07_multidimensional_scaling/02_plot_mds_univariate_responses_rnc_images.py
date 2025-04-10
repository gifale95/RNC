"""Plot the MDS results on the in silico univariate fMRI responses of all ROIs,
averaged across subjects.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
rois : lisr
	List of used ROIs.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'V4', 'EBA', 'FFA', 'PPA', 'RSC'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='..relational_neural_control', type=str)
args = parser.parse_args()


# =============================================================================
# Load the MDS results
# =============================================================================
data_dir = os.path.join(os.path.join(args.project_dir,
	'multidimensional_scaling', args.encoding_models_train_dataset+
	'_encoding_models', 'imageset-'+args.imageset,
	'mds_univariate_responses.npy'))
results = np.load(data_dir, allow_pickle=True).item()

roi_comb_names = results['roi_comb_names']
roi_comb = results['roi_comb']
mds = {}
for r, roi in enumerate(roi_comb_names):
	mds['h1h2_'+roi] = results['mds_h1h2'][roi]
	mds['h1l2_'+roi] = results['mds_h1l2'][roi]
	mds['l1h2_'+roi] = results['mds_l1h2'][roi]
	mds['l1l2_'+roi] = results['mds_l1l2'][roi]
mds_all_images = results['mds_all_images']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
	(217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]


# =============================================================================
# Min-max alphas normalization to range [.1, 1], based on the MDS-space distance
# between each ROI
# =============================================================================
# Standardize the coordinates within the two MDS dimensions, so that they will
# equally contribute to the distance computation
# Controlling images
for key, val in mds.items():
	for d in range(val.shape[1]):
		mds[key][:,d] = zscore(val[:,d])
# All images
for d in range(mds_all_images.shape[1]):
	mds_all_images[:,d] = zscore(mds_all_images[:,d])

# Compute the alphas for the controlling images
min_alpha = 0.1
max_alpha = 1
alphas = {}
for key, val in mds.items():
	# Compute the distance in MDS-space between each ROI
	dist = []
	for r1, r2 in roi_comb:
		dist.append(abs(val[r1,0] - val[r2,0]) + abs(val[r1,1] - val[r2,1]))
	dist = np.asarray(dist)
	# Compute the alphas based on the distance
	min_dist, max_dist = dist.min(), dist.max()
	a = min_alpha + (dist - min_dist) * (max_alpha - min_alpha) / \
		(max_dist - min_dist)
	# Flip the scores, so that smaller distances are plotted with higher alphas
	alphas[key] = abs(a - max_alpha - min_alpha)

# Compute the alphas on the MDS results for all images
dist = []
for r1, r2 in roi_comb:
	dist.append(abs(mds_all_images[r1,0] - mds_all_images[r2,0]) + \
		abs(mds_all_images[r1,1] - mds_all_images[r2,1]))
dist = np.asarray(dist)
# Compute the alphas based on the distance
min_dist, max_dist = dist.min(), dist.max()
a = min_alpha + (dist - min_dist) * (max_alpha - min_alpha) / \
	(max_dist - min_dist)
# Flip the scores, so that smaller distances are plotted with higher alphas
alphas_all_images = abs(a - max_alpha - min_alpha)


# =============================================================================
# Plot the results
# =============================================================================
# Create the figure
for key, val in tqdm(mds.items()):
	fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 7))
	axs = np.reshape(axs, (-1))
	ax1 = axs[0]
	ax2 = ax1.twinx()
	ax1.set_zorder(2)
	ax2.set_zorder(1)
	ax1.patch.set_visible(False)
	if key[:4] == 'h1h2':
		color = colors[0]
	elif key[:4] == 'l1l2':
		color = colors[1]
	elif key[:4] == 'h1l2':
		color = colors[2]
	elif key[:4] == 'l1h2':
		color = colors[3]

	# Plot the connections between ROIs
	for r, (r1, r2) in enumerate(roi_comb):
		ax2.plot([val[r1,0], val[r2,0]], [val[r1,1], val[r2,1]],
			color=color, linewidth=5, alpha=alphas[key][r])

	# Plot each ROI in MDS space
	for r, roi in enumerate(args.rois):
		if roi in ['V1', 'V2', 'V3', 'V4']:
			ax1.scatter(val[r,0], val[r,1], s=2500, c='w', linewidths=0,
				alpha=1)
		elif roi in ['EBA', 'FFA', 'PPA', 'RSC']:
			ax1.scatter(val[r,0], val[r,1], s=4500, c='w', linewidths=0,
				alpha=1)
		ax1.text(val[r,0], val[r,1], roi, fontsize=fontsize, fontweight='bold',
			ha='center', va='center_baseline', color='k')

	# x-axis
	ax1.set_xticks([])
	ax2.set_xticks([])

	# y-axis
	ax1.set_yticks([])
	ax2.set_yticks([])

	# Save the figure
	file_name = 'mds_univariate_responses_' + key + '_imageset-' + \
		args.imageset + '.svg'
	fig.savefig(file_name, bbox_inches='tight', format='svg')

	# Close the figure
	plt.close()


# =============================================================================
# Plot the results for all images
# =============================================================================
# Create the figure
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 7))
axs = np.reshape(axs, (-1))
ax1 = axs[0]
ax2 = ax1.twinx()
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)

# Plot the connections between ROIs
for r, (r1, r2) in enumerate(roi_comb):
	ax2.plot([mds_all_images[r1,0], mds_all_images[r2,0]],
		[mds_all_images[r1,1], mds_all_images[r2,1]], color='k', linewidth=5,
		alpha=alphas_all_images[r])

# Plot each ROI in MDS space
for r, roi in enumerate(args.rois):
	if roi in ['V1', 'V2', 'V3', 'V4']:
		ax1.scatter(mds_all_images[r,0], mds_all_images[r,1], s=2500, c='w',
			linewidths=0, alpha=1)
	elif roi in ['EBA', 'FFA', 'PPA', 'RSC']:
		ax1.scatter(mds_all_images[r,0], mds_all_images[r,1], s=4500, c='w',
			linewidths=0, alpha=1)
	ax1.text(mds_all_images[r,0], mds_all_images[r,1], roi, fontsize=fontsize,
		fontweight='bold', ha='center', va='center_baseline', color='k')

# x-axis
ax1.set_xticks([])
ax2.set_xticks([])

# y-axis
ax1.set_yticks([])
ax2.set_yticks([])

# Save the figure
file_name = 'mds_univariate_responses_all_images_imageset-' + \
	args.imageset + '.svg'
fig.savefig(file_name, bbox_inches='tight', format='svg')
