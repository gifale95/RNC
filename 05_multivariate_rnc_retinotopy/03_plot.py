"""Plot the multivariate RNC retinotopy effect results.

This code is available at:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/03_plot.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
rois : list of str
	List of used ROIs.
control_condition : str
	Whether to plot RSMs for images that 'align' or 'disentangle' the
	multivariate fMRI responses for the two ROIs being compared.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--control_condition', type=str, default='align')
parser.add_argument('--rois', type=list, default=['V1', 'hV4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='/home/ale/scratch/projects/relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Create fMRI RSMs <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
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
matplotlib.rcParams['axes.grid'] = False


# =============================================================================
# Plot the RSM results
# =============================================================================
# Load the RSM results
data_dir = os.path.join(args.project_dir, 'retinotopy_effect', 'imageset-'+
	args.imageset, 'rsms_control_condition-'+args.control_condition+'.npy')
data = np.load(data_dir, allow_pickle=True).item()
rsm_v1 = data['rsm_v1']
rsm_v4 = data['rsm_v4']

# Plot the V1 RSM
fig = plt.figure()
cax = plt.imshow(np.mean(rsm_v1, 0), cmap='coolwarm', vmin=-1, vmax=1)
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
if args.control_condition == 'disentangle':
	label = 'Disentangling images'
elif args.control_condition == 'align':
	label = 'Aligning images'
plt.xlabel(label, fontsize=fontsize)
plt.ylabel(label, fontsize=fontsize)
plt.title('V1', fontsize=fontsize)
# Colorbar
cbar = plt.colorbar(cax, shrink=0.75, ticks=[-1, 0, 1], label='Pearson\'s $r$',
	location='left')
# Place colorbar labels and ticks on the left
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
#fig.savefig('multivariate_rnc_aligning_images_nsd_v1.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_disentangling_images_nsd_v1.png', dpi=100, bbox_inches='tight')

# Plot the V4 RSM
fig = plt.figure()
cax = plt.imshow(np.mean(rsm_v4, 0), cmap='coolwarm', vmin=-1, vmax=1)
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
if args.control_condition == 'disentangle':
	label = 'Disentangling images'
elif args.control_condition == 'align':
	label = 'Aligning images'
plt.xlabel(label, fontsize=fontsize)
plt.ylabel(label, fontsize=fontsize)
plt.title('V4', fontsize=fontsize)
# Colorbar
cbar = plt.colorbar(cax, shrink=0.75, ticks=[-1, 0, 1], label='Pearson\'s $r$',
	location='left')
# Place colorbar labels and ticks on the left
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
#fig.savefig('multivariate_rnc_aligning_images_nsd_v4.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_disentangling_images_nsd_v4.png', dpi=100, bbox_inches='tight')

# Plot the V1 zoomed-in RSM
rsm_avg = np.mean(rsm_v1, 0)
y_range = np.arange(12, 18)
x_range = np.arange(37, 43)
rsm_zoom = rsm_avg[y_range]
rsm_zoom = rsm_zoom[:,x_range]
fig = plt.figure()
plt.imshow(rsm_zoom, cmap='coolwarm', vmin=-1, vmax=1)
for i1 in range(len(y_range)):
	for i2 in range(len(x_range)):
		s = '$r$=' + str(np.round(rsm_zoom[i1,i2], 2))
		plt.text(i2-.4, i1, s=s, fontsize=27, va='center', ha='left')
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
#fig.savefig('multivariate_rnc_aligning_images_nsd_v1_zoomed.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the RSM average correlation between different aligning image types
# =============================================================================
# Load the results
data_dir = os.path.join(args.project_dir, 'retinotopy_effect', 'imageset-'+
	args.imageset, 'rsms_control_condition-align.npy')
data = np.load(data_dir, allow_pickle=True).item()

# Plot parameters
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
alpha = 0.2
x_coord = np.asarray((1, 2, 3))
s = 600
s_mean = 800
x_lim_min = .5
x_lim_max = 3.5
y_lim_min = -1
y_lim_max = 1

# Plot
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(9, 12))
axs = np.reshape(axs, (-1))

for r, roi in enumerate(args.rois):

	# Sky vs. sky
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[0], len(args.all_subjects))
	if roi == 'V1':
		y = data['v1_corr_sky_sky']
	elif roi == 'hV4':
		y = data['v4_corr_sky_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data['ci_v1_corr_sky_sky'][0]
		ci_up = data['ci_v1_corr_sky_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data['ci_v4_corr_sky_sky'][0]
		ci_up = data['ci_v4_corr_sky_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Sky vs. non sky
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[1], len(args.all_subjects))
	if roi == 'V1':
		y = data['v1_corr_sky_non_sky']
	elif roi == 'hV4':
		y = data['v4_corr_sky_non_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data['ci_v1_corr_sky_non_sky'][0]
		ci_up = data['ci_v1_corr_sky_non_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data['ci_v4_corr_sky_non_sky'][0]
		ci_up = data['ci_v4_corr_sky_non_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Non sky vs. non sky
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[2], len(args.all_subjects))
	if roi == 'V1':
		y = data['v1_corr_non_sky_non_sky']
	elif roi == 'hV4':
		y = data['v4_corr_non_sky_non_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data['ci_v1_corr_non_sky_non_sky'][0]
		ci_up = data['ci_v1_corr_non_sky_non_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data['ci_v4_corr_non_sky_non_sky'][0]
		ci_up = data['ci_v4_corr_non_sky_non_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# x-axis parameters
	xticks = x_coord
	xlabels = ['', '', '']
	axs[r].set_xticks(ticks=xticks, labels=xlabels)
	axs[r].set_xlim(left=x_lim_min, right=x_lim_max)

	# y-axis parameters
	ylabel = 'Pearson\'s $r$'
	axs[r].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-1, 0, 1]
	ylabels = [-1, 0, 1]
	plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=y_lim_min, top=y_lim_max)

	# Title
	if roi == 'V1':
		title = 'V1'
	elif r == 'hV4':
		title = 'V4'
	axs[r].set_title(title, fontsize=fontsize)

#fig.savefig('multivariate_rnc_retinotopy_effect_aligning_images_pearson.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the univariate responses for V1 and V4 dorsal and ventral voxels
# =============================================================================
# Plot univariate responses (i.e., voxel-average response) of V1 and V4 voxels
# tuned to the upper and lower portion of the visual field, for aligning images
# containing or not containing skies on their upper half.

# Load the results
data = {}
for roi in args.rois:
	data_dir = os.path.join(args.project_dir, 'retinotopy_effect', 'imageset-'+
		args.imageset, 'retinotopy_effect-'+roi+'.npy')
	data[roi] = np.load(data_dir, allow_pickle=True).item()

# Plot parameters
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
alpha = 0.2
x_coord = np.asarray((1, 2, 4, 5))
s = 600
s_mean = 800
sig_offset = 0.1
sig_bar_length = 0.03
linewidth_sig_bar = 1.5
sig_star_offset_top = 0.025
x_lim_min = 0
x_lim_max = 6
y_lim_min = -0.75
y_lim_max = 0.55

# Plot
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(9, 12))
axs = np.reshape(axs, (-1))

for r, roi in enumerate(args.rois):

	# Sky images, lower visual field
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[0], len(args.all_subjects))
	if roi == 'V1':
		y = data[roi]['mean_dorsal_sky']
	elif roi == 'hV4':
		y = data[roi]['mean_ventral_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data[roi]['ci_dorsal_sky'][0]
		ci_up = data[roi]['ci_dorsal_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data[roi]['ci_ventral_sky'][0]
		ci_up = data[roi]['ci_ventral_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Sky images, upper visual field
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[1], len(args.all_subjects))
	if roi == 'V1':
		y = data[roi]['mean_ventral_sky']
	elif roi == 'hV4':
		y = data[roi]['mean_dorsal_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data[roi]['ci_ventral_sky'][0]
		ci_up = data[roi]['ci_ventral_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data[roi]['ci_dorsal_sky'][0]
		ci_up = data[roi]['ci_dorsal_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Plot significance within sky image conditions
	if data[roi]['significance_1'] == True:
		responses = np.append(data[roi]['mean_ventral_sky'],
			data[roi]['mean_dorsal_sky'])
		y_max = max(responses) + sig_offset
		axs[r].plot([x_coord[0], x_coord[0]], [y_max, y_max+sig_bar_length],
			'k-', [x_coord[0], x_coord[1]],
			[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
			[x_coord[1], x_coord[1]], [y_max+sig_bar_length, y_max], 'k-',
			linewidth=linewidth_sig_bar)
		x_mean = np.mean(np.asarray((x_coord[0], x_coord[1])))
		y = y_max + sig_bar_length + sig_star_offset_top
		axs[r].text(x_mean, y, s='*', fontsize=20, color='k',
			fontweight='bold', ha='center', va='center')

	# Non-sky images, lower visual field
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[2], len(args.all_subjects))
	if roi == 'V1':
		y = data[roi]['mean_dorsal_non_sky']
	elif roi == 'hV4':
		y = data[roi]['mean_ventral_non_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data[roi]['ci_dorsal_non_sky'][0]
		ci_up = data[roi]['ci_dorsal_non_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data[roi]['ci_ventral_non_sky'][0]
		ci_up = data[roi]['ci_ventral_non_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Non-sky images, upper visual field
	# Plot the single subjects univariate responses
	x = np.repeat(x_coord[3], len(args.all_subjects))
	if roi == 'V1':
		y = data[roi]['mean_ventral_non_sky']
	elif roi == 'hV4':
		y = data[roi]['mean_dorsal_non_sky']
	axs[r].scatter(x, y, s, c='k', alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(y)
	axs[r].scatter(x, y, s_mean, c='k')
	# Plot the CIs
	if roi == 'V1':
		ci_low = y - data[roi]['ci_ventral_non_sky'][0]
		ci_up = data[roi]['ci_ventral_non_sky'][1] - y
	elif roi == 'hV4':
		ci_low = y - data[roi]['ci_dorsal_non_sky'][0]
		ci_up = data[roi]['ci_dorsal_non_sky'][1] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	axs[r].errorbar(x, y, yerr=conf_int, fmt="none", ecolor='k', elinewidth=5,
		capsize=0)

	# Plot significance within non sky image conditions
	if data[roi]['significance_3'] == True:
		responses = np.append(data[roi]['mean_ventral_non_sky'],
			data[roi]['mean_dorsal_non_sky'])
		y_max = max(responses) + sig_offset
		axs[r].plot([x_coord[2], x_coord[2]], [y_max, y_max+sig_bar_length],
			'k-', [x_coord[2], x_coord[3]],
			[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
			[x_coord[3], x_coord[3]], [y_max+sig_bar_length, y_max], 'k-',
			linewidth=linewidth_sig_bar)
		x_mean = np.mean(np.asarray((x_coord[2], x_coord[3])))
		y = y_max + sig_bar_length + sig_star_offset_top
		axs[r].text(x_mean, y, s='*', fontsize=20, color='k', fontweight='bold',
			ha='center', va='center')

	# x-axis parameters
	xticks = x_coord
	xlabels = ['', '', '', '']
	axs[r].set_xticks(ticks=xticks, labels=xlabels, rotation=45)
	axs[r].set_xlim(left=x_lim_min, right=x_lim_max)

	# y-axis parameters
	ylabel = 'Univariate\nresponse'
	axs[r].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-0.5, 0, 0.5]
	ylabels = [-0.5, 0, 0.5]
	plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=y_lim_min, top=y_lim_max)

	# Title
	if roi == 'V1':
		title = 'V1'
	elif roi == 'hV4':
		title = 'V4'
	axs[r].set_title(title, fontsize=fontsize)

#fig.savefig('multivariate_rnc_retinotopy_effect_univariate_responses.png', dpi=100, bbox_inches='tight')
