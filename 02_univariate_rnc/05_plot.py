"""Plot the univariate RNC cross-subject validated results.

This code is available at:
https://github.com/gifale95/RNC/blob/main/02_univariate_rnc/05_plot.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
rois : list of str
	List of used ROIs.
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
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()


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
# Load the univariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'stats', 'cv-1',
	'imageset-'+args.imageset, 'stats.npy')

data_dict = np.load(data_dir, allow_pickle=True).item()


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
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
	(217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]


# =============================================================================
# Plot the univariate responses for the controlling images on scatterplots
# =============================================================================
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

for r in range(len(r1)):

	# Diagonal dashed line
	axs[r].plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2,
		alpha=.4, label='_nolegend_')

	# Baseline images dashed lines
	baseline_roi_1 = np.mean(
		data_dict['baseline_images_score_test'][:,r1[r]])
	axs[r].plot([baseline_roi_1, baseline_roi_1], [-3, 3], '--w', linewidth=2,
		alpha=.6, label='_nolegend_')
	baseline_roi_2 = np.mean(
		data_dict['baseline_images_score_test'][:,r1[r]])
	axs[r].plot([-3, 3], [baseline_roi_2, baseline_roi_2], '--w', linewidth=2,
		alpha=.6, label='_nolegend_')

	# Univariate responses for all images
	for s, sub in enumerate(args.all_subjects):
		axs[r].scatter(data_dict['uni_resp'][s,r1[r]],
			data_dict['uni_resp'][s,r2[r]], c='w', alpha=.1,
			edgecolors='k', label='_nolegend_')

	# Univariate responses for the controlling images
	for s, sub in enumerate(args.all_subjects):
		# 1. high_1_high_2
		idx_imgs = data_dict['high_1_high_2'][s,r]
		axs[r].scatter(np.mean(data_dict['uni_resp'][s,r1[r],idx_imgs]),
			np.mean(data_dict['uni_resp'][s,r2[r],idx_imgs]), c=colors[0],
			s=400, alpha=0.8)
		# 2. low_1_low_2
		idx_imgs = data_dict['low_1_low_2'][s,r]
		axs[r].scatter(np.mean(data_dict['uni_resp'][s,r1[r],idx_imgs]),
			np.mean(data_dict['uni_resp'][s,r2[r],idx_imgs]), c=colors[1],
			s=400, alpha=0.8)
		# 3. high_1_low_2
		idx_imgs = data_dict['high_1_low_2'][s,r]
		axs[r].scatter(np.mean(data_dict['uni_resp'][s,r1[r],idx_imgs]),
			np.mean(data_dict['uni_resp'][s,r2[r],idx_imgs]), c=colors[2],
			s=400, alpha=0.8)
		# 4. low_1_high_2
		idx_imgs = data_dict['low_1_high_2'][s,r]
		axs[r].scatter(np.mean(data_dict['uni_resp'][s,r1[r],idx_imgs]),
			np.mean(data_dict['uni_resp'][s,r2[r],idx_imgs]), c=colors[3],
			s=400, alpha=0.8)

	# Add the correlation scores the two ROI responses for all images
	x = -1.6
	y = 1
	s = '$r$=' + str(np.round(np.mean(data_dict['roi_pair_corr'][:,r]), 2))
	axs[r].text(x, y, s, fontsize=fontsize)

	# x-axis parameters
	xlabel = 'Univariate\nresponse'
	if r in [3, 4, 5]:
		axs[r].set_xlabel(xlabel, fontsize=fontsize)
	xticks = [-2, -1, 0, 1, 2]
	xlabels = [-2, -1, 0, 1, 2]
	axs[r].set_xticks(ticks=xticks, labels=xlabels)
	axs[r].set_xlim(left=-1.75, right=1.25)

	# y-axis parameters
	ylabel = 'Univariate\nresponse'
	if r in [0, 3]:
		axs[r].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-2, -1, 0, 1, 2]
	ylabels = [-2, -1, 0, 1, 2]
	axs[r].set_yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=-1.75, top=1.25)

	# Title
	title = args.rois[r1[r]] + '-' + args.rois[r2[r]]
	axs[r].set_title(title, fontsize=fontsize)

	# Legend
	if s == 0 and r == 0:
		legend = ['High ROI-x, High ROI-y', 'High ROI-x, Low ROI-y', 
			'Low ROI-x, High ROI-y', 'Low ROI-x, Low ROI-y',
			'Null distribution ROI-x', 'Null distribution ROI-y']
		axs[r].legend(legend, loc=2, ncol=6, fontsize=15, markerscale=3,
			bbox_to_anchor=(0, -6.4))

	# Aspect
	axs[r].set_aspect('equal')

#fig.savefig('univariate_rnc_scatterplots_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_scatterplots_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_scatterplots_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the univariate responses significance for the controlling images
# =============================================================================
# Format the results for plotting
control_responses = []
control_responses.append(data_dict['high_1_high_2_resp'])
control_responses.append(data_dict['low_1_low_2_resp'])
control_responses.append(data_dict['high_1_low_2_resp'])
control_responses.append(data_dict['low_1_high_2_resp'])
ci_control_responses = []
ci_control_responses.append(data_dict['ci_high_1_high_2'])
ci_control_responses.append(data_dict['ci_low_1_low_2'])
ci_control_responses.append(data_dict['ci_high_1_low_2'])
ci_control_responses.append(data_dict['ci_low_1_high_2'])
sig_control_responses = []
sig_control_responses.append(data_dict['sig_high_1_high_2'])
sig_control_responses.append(data_dict['sig_low_1_low_2'])
sig_control_responses.append(data_dict['sig_high_1_low_2'])
sig_control_responses.append(data_dict['sig_low_1_high_2'])
baseline_responses = data_dict['baseline_images_score_test']

# Plot parameters
lim_min = -1.75
lim_max = 1.25
padding = 0.4
x_dist = (abs(lim_min - lim_max) - (padding*2)) / 3
x_dist_within = float(0.25)
x_start = lim_min + padding
xticks = np.asarray((x_start, x_start+x_dist*1, x_start+x_dist*2, x_start+x_dist*3))
x_coord = xticks - (x_dist_within / 2)
alpha = 0.2
sig_bar_length = 0.1
sig_star_offset_top = 0.13
sig_star_offset_bottom = 0.26
fontsize_sig = 20
marker_roi_1 = 'd'
marker_roi_2 = 's'
null_width = 0.1

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))

for r in range(len(r1)):
	for c in range(len(control_responses)):

		# ROI 1 (baseline images univariate responses)
		x_null = np.repeat(x_coord[c], len(args.all_subjects))
		y = baseline_responses[:,r1[r]]
		axs[r].plot([x_null[0]-null_width, x_null[0]+null_width],
			[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
			alpha=.4)

		# ROI 1 (controlling images univariate responses)
		x = np.repeat(x_coord[c], len(args.all_subjects))
		x_score = x[0]
		y = np.mean(control_responses[c][:,r,0], 1)
		axs[r].scatter(x, y, marker=marker_roi_1, s=200, color=colors[c],
			alpha=alpha)
		axs[r].scatter(x[0], np.mean(y), marker=marker_roi_1, s=400,
			color=colors[c])
		# ROI 1 (controlling images univariate responses CIs)
		ci_low = np.mean(y) - ci_control_responses[c][0,r,0]
		ci_up = ci_control_responses[c][1,r,0] - np.mean(y)
		conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
		axs[r].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
			ecolor=colors[c], elinewidth=5, capsize=0)

		# ROI 1 (controlling images univariate responses significance)
		if sig_control_responses[c][r,0] == 1:
			if c in [0, 2]:
				y = max(np.mean(control_responses[c][:,r,0], 1)) + \
					sig_star_offset_top
				axs[r].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')
			elif c in [1, 3]:
				y = min(np.mean(control_responses[c][:,r,0], 1)) - \
					sig_star_offset_bottom
				axs[r].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')

		# ROI 2 (baseline images univariate responses)
		x_null = x + x_dist_within
		y = baseline_responses[:,r2[r]]
		axs[r].plot([x_null[0]-null_width, x_null[0]+null_width],
			[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
			alpha=.4)

		# ROI 2 (controlling images univariate responses)
		x += x_dist_within
		x_score = x[0]
		y = np.mean(control_responses[c][:,r,1], 1)
		axs[r].scatter(x, y, marker=marker_roi_2, s=200, color=colors[c],
			alpha=alpha)
		axs[r].scatter(x[0], np.mean(y), marker=marker_roi_2, s=400,
			color=colors[c])
		# ROI 2 (controlling images univariate responses CIs)
		ci_low = np.mean(y) - ci_control_responses[c][0,r,1]
		ci_up = ci_control_responses[c][1,r,1] - np.mean(y)
		conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
		axs[r].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
			ecolor=colors[c], elinewidth=5, capsize=0)

		# ROI 2 (controlling images univariate responses significance)
		if sig_control_responses[c][r,1] == 1:
			if c in [0, 3]:
				y = max(np.mean(control_responses[c][:,r,1], 1)) + \
					sig_star_offset_top
				axs[r].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')
			elif c in [1, 2]:
				y = min(np.mean(control_responses[c][:,r,1], 1)) - \
					sig_star_offset_bottom
				axs[r].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')

	# x-axis parameters
	xlabels = ['', '', '', '']
	axs[r].set_xticks(ticks=xticks, labels=xlabels, rotation=45)
	xlabel = 'Neural control\nconditions'
	if r in [3, 4, 5]:
		axs[r].set_xlabel(xlabel, fontsize=fontsize)
	axs[r].set_xlim(left=lim_min, right=lim_max)

	# y-axis parameters
	ylabel = 'Univariate\nresponse'
	if r in [0, 3]:
		axs[r].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-1, 0, 1]
	ylabels = [-1, 0, 1]
	plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=lim_min, top=lim_max)

	# Title
	title = args.rois[r1[r]] + '-' + args.rois[r2[r]]
	axs[r].set_title(title, fontsize=fontsize)

	# Aspect
	axs[r].set_aspect('equal')

#fig.savefig('univariate_rnc_significance_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_significance_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_significance_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the difference between the ROI univariate responses for the control
# conditions and the ROI baseline univariate response, and sort these
# differences as a function of cortical distance
# =============================================================================
# Plot parameters
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
x_dist_within = float(0.325)
sig_offset = 0.1
sig_bar_length = 0.03
linewidth_sig_bar = 1.5
sig_star_offset_top = 0.025

fig = plt.figure(figsize=(5, 7))

for d, dist in enumerate(area_distance):

	# high_1_high_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist-x_dist_within*0.5, len(args.all_subjects))
	y = data_dict['sorted_h1h2_resp'][:,d]
	plt.scatter(x, y, s, c=colors[0], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(data_dict['sorted_h1h2_resp'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[0], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[0], label='_nolegend_')
	# Plot the CIs
	ci_low = y - data_dict['ci_sorted_h1h2_resp'][0,d]
	ci_up = data_dict['ci_sorted_h1h2_resp'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[0],
		elinewidth=5, capsize=0)

	# low_1_low_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist+x_dist_within*0.5, len(args.all_subjects))
	y = data_dict['sorted_l1l2_resp'][:,d]
	plt.scatter(x, y, s, c=colors[1], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(data_dict['sorted_l1l2_resp'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[1], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[1], label='_nolegend_')
	# Plot the CIs
	ci_low = y - data_dict['ci_sorted_l1l2_resp'][0,d]
	ci_up = data_dict['ci_sorted_l1l2_resp'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[1],
		elinewidth=5, capsize=0)

	# high_1_low_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist-x_dist_within*0.5, len(args.all_subjects))
	y = data_dict['sorted_h1l2_resp'][:,d]
	plt.scatter(x, y, s, c=colors[2], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(data_dict['sorted_h1l2_resp'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[2], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[2], label='_nolegend_')
	# Plot the CIs
	ci_low = y - data_dict['ci_sorted_h1l2_resp'][0,d]
	ci_up = data_dict['ci_sorted_h1l2_resp'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[2],
		elinewidth=5, capsize=0)

	# low_1_high_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist+x_dist_within*0.5, len(args.all_subjects))
	y = data_dict['sorted_l1h2_resp'][:,d]
	plt.scatter(x, y, s, c=colors[3], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(data_dict['sorted_l1h2_resp'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[3], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[3], label='_nolegend_')
	# Plot the CIs
	ci_low = y - data_dict['ci_sorted_l1h2_resp'][0,d]
	ci_up = data_dict['ci_sorted_l1h2_resp'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[3],
		elinewidth=5, capsize=0)

# Plot trend connectors
# high_1_low_2
if data_dict['sorted_h1l2_resp_increase'].pvalue < 0.05:
	x = area_distance - (x_dist_within * 0.5)
	y = np.mean(data_dict['sorted_h1l2_resp'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[2],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[2],
		alpha=0.5)
# low_1_high_2
if data_dict['sorted_l1h2_resp_increase'].pvalue < 0.05:
	x = area_distance + (x_dist_within * 0.5)
	y = np.mean(data_dict['sorted_l1h2_resp'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[3],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[3],
		alpha=0.5)

# x-axis parameters
xticks = [1, 2, 3]
xlabels = [1, 2, 3]
plt.xticks(ticks=xticks, labels=xlabels)
xlabel = 'Stepwise area distance'
plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=0.5, right=3.5)

# y-axis parameters
yticks = [0, 1]
ylabels = [0, 1]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Absolute Δ univariate response'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=1.5)

#fig.savefig('univariate_rnc_cortical_distance_responses_nsd', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_cortical_distance_responses_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_cortical_distance_responses_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the pairwise ROI combinations univariate response correlations as a
# function of cortical distance
# =============================================================================
# Plot parameters
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
color = 'k'

fig = plt.figure(figsize=(5, 7))

for d, dist in enumerate(area_distance):

	# Plot the single subjects correlation scores
	x = np.repeat(dist, len(args.all_subjects))
	y = data_dict['sorted_corr'][:,d]
	plt.scatter(x, y, s, c=color, alpha=alpha)

	# Plot the subject-average correlation scores
	x = dist
	y = np.mean(data_dict['sorted_corr'][:,d])
	plt.scatter(x, y, s_mean, c=color)

	# Plot the CIs
	ci_low = y - data_dict['ci_sorted_corr'][0,d]
	ci_up = data_dict['ci_sorted_corr'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=color, elinewidth=5,
		capsize=0)

	# x-axis parameters
	xticks = [1, 2, 3]
	xlabels = [1, 2, 3]
	plt.xticks(ticks=xticks, labels=xlabels)
	xlabel = 'Stepwise area distance'
	plt.xlabel(xlabel, fontsize=fontsize)
	plt.xlim(left=0.5, right=3.5)

	# y-axis parameters
	ylabel = 'Pearson\'s $r$'
	plt.ylabel(ylabel, fontsize=fontsize)
	if args.imageset == 'nsd':
		plt.ylim(bottom=0.5, top=1)
	else:
		plt.ylim(bottom=0.4, top=1)

# Plot trend connectors
# high_1_high_2
if data_dict['sorted_corr_decrease'].pvalue < 0.05:
	x = area_distance
	y = np.mean(data_dict['sorted_corr'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=color, alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=color, alpha=0.5)

#fig.savefig('univariate_rnc_cortical_distance_corr_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_cortical_distance_corr_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('univariate_rnc_cortical_distance_corr_things.png', dpi=100, bbox_inches='tight')

