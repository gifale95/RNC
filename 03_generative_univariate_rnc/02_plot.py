"""Plot the generative univariate RNC results.

This code is available at:
https://github.com/gifale95/RNC/blob/main/03_generative_univariate_rnc/02_plot.py

Parameters
----------
roi_pair : int
	Integer indicating the chosen pairwise ROI combination on which to perform
	multivariate RNC. Possible values are '0' (V1-V2), '1' (V1-V3), '2'
	(V1-hV4), '3' (V2-V3), '4' (V2-hV4), '5' (V3-hV4).
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
parser.add_argument('--roi_pair', type=int, default=2)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()


# =============================================================================
# Pairwise ROI combinations and neural control types
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

# 0: V1
# 1: V2
# 2: V3
# 3: hV4
r1 = [0, 0, 0, 1, 1, 2]
r2 = [1, 2, 3, 2, 3, 3]

# Neural control types
control_types = ['high_1_high_2', 'high_1_low_2', 'low_1_high_2', 'low_1_low_2']


# =============================================================================
# Load the generative univariate RNC results
# =============================================================================
fmri_roi_1 = {}
fmri_roi_2 = {}
baseline_penalty_train = {}
images_complexity = {}

for c in control_types:

	data_dir = os.path.join(args.project_dir, 'generative_univariate_rnc',
		'optimization_scores', 'cv-0', roi_1+'-'+roi_2, 'control_condition-'+c,
		'optimization_scores.npy')

	data_dict = np.load(data_dir, allow_pickle=True).item()

	fmri_roi_1[c] = np.mean(np.squeeze(data_dict['best_fmri_roi_1']), 1)
	fmri_roi_2[c] = np.mean(np.squeeze(data_dict['best_fmri_roi_2']), 1)
	baseline_penalty_train[c] = np.squeeze(
		data_dict['best_baseline_penalty_train'])
	images_complexity[c] = np.squeeze(
		data_dict['best_images_complexity']) / 1024 # Convert bytes to kB


# =============================================================================
# Get the null distribution threshold generation
# =============================================================================
threshold = {}

for key, val in baseline_penalty_train.items():
	idx = np.where(val == 1e+10)[0]
	threshold[key] = idx[-1] + 1


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'stats', 'cv-0',
	'imageset-'+args.imageset, 'stats.npy')

data_dict = np.load(data_dir, allow_pickle=True).item()

uni_resp = data_dict['uni_resp']
baseline_images_score = data_dict['baseline_images_score']


# =============================================================================
# Set the plot parameters
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
matplotlib.rcParams['axes.grid'] = False
colors = [(4/255, 178/255, 153/255), (217/255, 214/255, 111/255),
	(214/255, 83/255, 117/255), (130/255, 201/255, 240/255)]


# =============================================================================
# Plot the generative univariate RNC optimization curves
# =============================================================================
for c, control in enumerate(control_types):

	fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(5, 3))
	axs = np.reshape(axs, (-1))

	# Plot
	ax1 = axs[0]
	ax2 = ax1.twinx()
	ax1.set_zorder(2)
	ax2.set_zorder(1)
	ax1.patch.set_visible(False)
	x = np.arange(len(fmri_roi_1[control]))
	# Plot the baseline threshold
	ax1.plot([threshold[control], threshold[control]], [-1e+10, 1e+10], '--',
		color='grey', label='__no_label__', linewidth=3)
	# Plot the image complexity
	ax2.plot(x, images_complexity[control], linewidth=3, color='k',
		label='Image complexity', zorder=1)
	# Plot the fMRI univariate responses
	ax1.plot(x, fmri_roi_1[control], linewidth=3, color=colors[c],
		label=roi_1, zorder=2)
	ax1.plot(x, fmri_roi_2[control], '--', linewidth=3, color=colors[c],
		label=roi_2, zorder=2)

	# y-axis parameters
	ax1.set_ylabel('Univariate\nresponse', fontsize=fontsize, color=colors[c])
	yticks = [-1, 0, 1]
	ylabels = [-1, 0, 1]
	ax1.set_yticks(ticks=yticks, labels=ylabels, color=colors[c])
	ax1.set_ylim(bottom=-1.25, top=1.25)
	ax2.set_ylabel('PNG compression\nfile size (kB)', fontsize=fontsize)
	yticks = [50, 100, 150]
	ylabels = [50, 100, 150]
	ax2.set_yticks(ticks=yticks, labels=ylabels)
	ax2.set_ylim(bottom=0, top=175)

	# x-axis parameters
	ax1.set_xlabel('Generations', fontsize=fontsize)
	xticks = [0, 250, 499]
	xlabels = [0, 250, 500]
	plt.xticks(ticks=xticks, labels=xlabels)
	plt.xlim(min(x), max(x))

	fig_name = 'univariate_rnc_optimization_curves' + control + '.png'
	#fig.savefig(fig_name, dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the univariate responses for the generative univariate RNC controlling
# images on scatterplots
# =============================================================================
fig = plt.figure(figsize=(6,6))

# Diagonal dashed line
plt.plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2,
	alpha=.4, label='_nolegend_')

# Null distribution dashed lines
baseline_roi_1 = baseline_images_score[r1[args.roi_pair]]
plt.plot([baseline_roi_1, baseline_roi_1], [-3, 3], '--w', linewidth=2,
	alpha=.6, label='_nolegend_')
baseline_roi_2 = baseline_images_score[r1[args.roi_pair]]
plt.plot([-3, 3], [baseline_roi_2, baseline_roi_2], '--w', linewidth=2,
	alpha=.6, label='_nolegend_')

# All images
plt.scatter(uni_resp[r1[args.roi_pair]], uni_resp[r2[args.roi_pair]], c='w',
	alpha=.1, edgecolors='k', label='_nolegend_')

# Controlling images
for c, control in enumerate(control_types):
	plt.scatter(fmri_roi_1[control][-1], fmri_roi_2[control][-1], s=1000,
		c=colors[c], alpha=0.8)

# x-axis
xlabel = roi_1 + ' univariate response'
plt.xlabel(xlabel, fontsize=fontsize)
xticks = [-2, -1, 0, 1, 2]
xlabels = [-2, -1, 0, 1, 2]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=-1.5, right=1)

# y-axis
ylabel = roi_2 + ' univariate response'
plt.ylabel(ylabel, fontsize=fontsize)
yticks = [-2, -1, 0, 1, 2]
ylabels = [-2, -1, 0, 1, 2]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-1.5, top=1)

fig_name = 'univariate_rnc_scatterplots_cv-0_' + args.imageset + '_' + \
	roi_1 + '-' + roi_2 + '.png'

#fig.savefig(fig_name, dpi=100, bbox_inches='tight')

