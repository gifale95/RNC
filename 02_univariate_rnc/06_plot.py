"""Plot the univariate RNC cross-subject validated results.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
roi_pairs : list
	List of used pairwise ROI combinations.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
# ['V1-V2', 'V1-V3', 'V1-hV4', 'V2-V3', 'V2-hV4', 'V3-hV4']
# ['EBA-FFA', 'EBA-PPA', 'EBA-RSC', 'FFA-PPA', 'FFA-RSC', 'PPA-RSC']
# ['FFA-V1', 'FFA-V2', 'FFA-V3', 'FFA-hV4']
# ['EBA-V1', 'EBA-V2', 'EBA-V3', 'EBA-hV4']
# ['PPA-V1', 'PPA-V2', 'PPA-V3', 'PPA-hV4']
# ['RSC-V1', 'RSC-V2', 'RSC-V3', 'RSC-hV4']
parser.add_argument('--roi_pairs', type=list, default=['V1-V2', 'V1-V3', 'V1-hV4', 'V2-V3', 'V2-hV4', 'V3-hV4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()


# =============================================================================
# Get the total dataset subjects
# =============================================================================
if args.encoding_models_train_dataset == 'nsd':
	all_subjects = [1, 2, 3, 4, 5, 6, 7, 8]

elif args.encoding_models_train_dataset == 'VisualIllusionRecon':
	all_subjects = [1, 2, 3, 4, 5, 6, 7]


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
stats = {}

for r in args.roi_pairs:

	data_dir = os.path.join(args.project_dir, 'univariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-1',
		'imageset-'+args.imageset, r, 'stats.npy')

	stats[r] = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 40
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
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
# Plot the univariate responses for the controlling images on scatterplots
# =============================================================================
for roi_pair, stats_roi in tqdm(stats.items()):

	fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6,6))
	axs = np.reshape(axs, (-1))

	# Diagonal dashed line
	axs[0].plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2,
		alpha=.4, label='_nolegend_')

	# Baseline images dashed lines
	baseline_roi_1 = np.mean(stats_roi['baseline_resp'][:,0])
	axs[0].plot([baseline_roi_1, baseline_roi_1], [-3, 3], '--w', linewidth=2,
		alpha=.6, label='_nolegend_')
	baseline_roi_2 = np.mean(stats_roi['baseline_resp'][:,1])
	axs[0].plot([-3, 3], [baseline_roi_2, baseline_roi_2], '--w', linewidth=2,
		alpha=.6, label='_nolegend_')

	# Univariate responses for all images
	for s, sub in enumerate(all_subjects):
		axs[0].scatter(stats_roi['uni_resp'][s,0], stats_roi['uni_resp'][s,1],
			c='w', alpha=.1, edgecolors='k', label='_nolegend_')

	# Univariate responses for the controlling images
	for s, sub in enumerate(all_subjects):
		# 1. high_1_high_2
		axs[0].scatter(np.mean(stats_roi['high_1_high_2_resp'][s,0]),
			np.mean(stats_roi['high_1_high_2_resp'][s,1]), c=colors[0],
			s=400, alpha=0.8)
		# 2. low_1_low_2
		axs[0].scatter(np.mean(stats_roi['low_1_low_2_resp'][s,0]),
			np.mean(stats_roi['low_1_low_2_resp'][s,1]), c=colors[1],
			s=400, alpha=0.8)
		# 3. high_1_low_2
		axs[0].scatter(np.mean(stats_roi['high_1_low_2_resp'][s,0]),
			np.mean(stats_roi['high_1_low_2_resp'][s,1]), c=colors[2],
			s=400, alpha=0.8)
		# 4. low_1_high_2
		axs[0].scatter(np.mean(stats_roi['low_1_high_2_resp'][s,0]),
			np.mean(stats_roi['low_1_high_2_resp'][s,1]), c=colors[3],
			s=400, alpha=0.8)

	# Add the correlation scores the two ROI responses for all images
	x = -1.6
	y = 1
	s = '$r$=' + str(np.round(np.mean(stats_roi['roi_pair_corr']), 2))
	axs[0].text(x, y, s, fontsize=fontsize)

	# x-axis parameters
	xlabel = 'Univariate\nresponse'
	axs[0].set_xlabel(xlabel, fontsize=fontsize)
	xticks = [-2, -1, 0, 1, 2]
	xlabels = [-2, -1, 0, 1, 2]
	axs[0].set_xticks(ticks=xticks, labels=xlabels)
	axs[0].set_xlim(left=-1.75, right=1.25)

	# y-axis parameters
	ylabel = 'Univariate\nresponse'
	axs[0].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-2, -1, 0, 1, 2]
	ylabels = [-2, -1, 0, 1, 2]
	axs[0].set_yticks(ticks=yticks, labels=ylabels)
	axs[0].set_ylim(bottom=-1.75, top=1.25)

	# Aspect
	axs[0].set_aspect('equal')

	# Save the figure
	file_name = 'univariate_rnc_scatterplots_encoding_models_train_dataset-' + \
		args.encoding_models_train_dataset + '_imageset-' + args.imageset + \
		'_' + roi_pair + '.svg'
	fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
	plt.close()


# =============================================================================
# Plot the univariate responses significance for the controlling images
# =============================================================================
# Format the results for plotting
control_responses = {}
ci_control_responses = {}
sig_control_responses = {}
baseline_responses = {}
for roi_pair, stats_roi in stats.items():
	# Control responses
	control_resp = []
	control_resp.append(stats_roi['high_1_high_2_resp'])
	control_resp.append(stats_roi['low_1_low_2_resp'])
	control_resp.append(stats_roi['high_1_low_2_resp'])
	control_resp.append(stats_roi['low_1_high_2_resp'])
	control_responses[roi_pair] = control_resp
	del control_resp
	# Confidence intervals
	ci_control_resp = []
	ci_control_resp.append(stats_roi['ci_high_1_high_2'])
	ci_control_resp.append(stats_roi['ci_low_1_low_2'])
	ci_control_resp.append(stats_roi['ci_high_1_low_2'])
	ci_control_resp.append(stats_roi['ci_low_1_high_2'])
	ci_control_responses[roi_pair] = ci_control_resp
	del ci_control_resp
	# Significance
	sig_control_resp = []
	sig_control_resp.append(stats_roi['h1h2_between_subject_pval'])
	sig_control_resp.append(stats_roi['l1l2_between_subject_pval'])
	sig_control_resp.append(stats_roi['h1l2_between_subject_pval'])
	sig_control_resp.append(stats_roi['l1h2_between_subject_pval'])
	sig_control_responses[roi_pair] = sig_control_resp
	del sig_control_resp
	# Baseline responses
	baseline_responses[roi_pair] = stats_roi['baseline_resp']

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

for roi_pair in args.roi_pairs:

	fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6,6))
	axs = np.reshape(axs, (-1))

	for c in range(len(control_responses[roi_pair])):

		# ROI 1 (baseline images univariate responses)
		x_null = np.repeat(x_coord[c], len(all_subjects))
		y = baseline_responses[roi_pair][:,0]
		axs[0].plot([x_null[0]-null_width, x_null[0]+null_width],
			[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
			alpha=.4)

		# ROI 1 (controlling images univariate responses)
		x = np.repeat(x_coord[c], len(all_subjects))
		x_score = x[0]
		y = np.mean(control_responses[roi_pair][c][:,0], 1)
		axs[0].scatter(x, y, marker=marker_roi_1, s=200, color=colors[c],
			alpha=alpha)
		axs[0].scatter(x[0], np.mean(y), marker=marker_roi_1, s=400,
			color=colors[c])
		# ROI 1 (controlling images univariate responses CIs)
		ci_low = np.mean(y) - ci_control_responses[roi_pair][c][0,0]
		ci_up = ci_control_responses[roi_pair][c][1,0] - np.mean(y)
		conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
		axs[0].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
			ecolor=colors[c], elinewidth=5, capsize=0)

		# ROI 1 (controlling images univariate responses significance)
		idx = roi_pair.find('-')
		roi_1 = roi_pair[:idx]
		roi_2 = roi_pair[idx+1:]
		if sig_control_responses[roi_pair][c][roi_1] < 0.05:
			if c in [0, 2]:
				y = max(np.mean(control_responses[roi_pair][c][:,0], 1)) + \
					sig_star_offset_top
				axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')
			elif c in [1, 3]:
				y = min(np.mean(control_responses[roi_pair][c][:,0], 1)) - \
					sig_star_offset_bottom
				axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')

		# ROI 2 (baseline images univariate responses)
		x_null = x + x_dist_within
		y = baseline_responses[roi_pair][:,1]
		axs[0].plot([x_null[0]-null_width, x_null[0]+null_width],
			[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
			alpha=.4)

		# ROI 2 (controlling images univariate responses)
		x += x_dist_within
		x_score = x[0]
		y = np.mean(control_responses[roi_pair][c][:,1], 1)
		axs[0].scatter(x, y, marker=marker_roi_2, s=200, color=colors[c],
			alpha=alpha)
		axs[0].scatter(x[0], np.mean(y), marker=marker_roi_2, s=400,
			color=colors[c])
		# ROI 2 (controlling images univariate responses CIs)
		ci_low = np.mean(y) - ci_control_responses[roi_pair][c][0,1]
		ci_up = ci_control_responses[roi_pair][c][1,1] - np.mean(y)
		conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
		axs[0].errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
			ecolor=colors[c], elinewidth=5, capsize=0)

		# ROI 2 (controlling images univariate responses significance)
		if sig_control_responses[roi_pair][c][roi_2] < 0.05:
			if c in [0, 3]:
				y = max(np.mean(control_responses[roi_pair][c][:,1], 1)) + \
					sig_star_offset_top
				axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')
			elif c in [1, 2]:
				y = min(np.mean(control_responses[roi_pair][c][:,1], 1)) - \
					sig_star_offset_bottom
				axs[0].text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
					fontweight='bold', ha='center', va='center')

	# x-axis parameters
	xlabels = ['', '', '', '']
	axs[0].set_xticks(ticks=xticks, labels=xlabels, rotation=45)
	xlabel = 'Neural control\nconditions'
	axs[0].set_xlabel(xlabel, fontsize=fontsize)
	axs[0].set_xlim(left=lim_min, right=lim_max)

	# y-axis parameters
	ylabel = 'Univariate\nresponse'
	axs[0].set_ylabel(ylabel, fontsize=fontsize)
	yticks = [-1, 0, 1, 2]
	ylabels = [-1, 0, 1, 2]
	plt.yticks(ticks=yticks, labels=ylabels)
	axs[0].set_ylim(bottom=lim_min, top=lim_max)

	# Aspect
	axs[0].set_aspect('equal')

	# Save the figure
	file_name = 'univariate_rnc_significance_encoding_models_train_dataset-' + \
		args.encoding_models_train_dataset + '_imageset-' + args.imageset + \
		'_' + roi_pair + '.svg'
	fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
	plt.close()


# =============================================================================
# Load the cortical distance analysis results
# =============================================================================
data_dir = save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'cortical_distance_analysis', 'imageset-'+args.imageset,
	'cortical_distance_analysis.npy')

results = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot the difference of the univariate response for the controlling images
# from baseline, as a function of cortical distance difference (only for ROIs:
# V1, V2, V3, hV4).
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

fig = plt.figure(figsize=(5, 8))

for d, dist in enumerate(area_distance):

	# high_1_low_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist-x_dist_within*0.5, len(all_subjects))
	y = results['sorted_h1l2_base_diff'][:,d]
	plt.scatter(x, y, s, c=colors[2], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(results['sorted_h1l2_base_diff'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[2], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[2], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_h1l2_base_diff'][0,d]
	ci_up = results['ci_sorted_h1l2_base_diff'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[2],
		elinewidth=5, capsize=0)

	# low_1_high_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist+x_dist_within*0.5, len(all_subjects))
	y = results['sorted_l1h2_base_diff'][:,d]
	plt.scatter(x, y, s, c=colors[3], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(results['sorted_l1h2_base_diff'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[3], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[3], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_l1h2_base_diff'][0,d]
	ci_up = results['ci_sorted_l1h2_base_diff'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[3],
		elinewidth=5, capsize=0)

	# high_1_high_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist-x_dist_within*0.5, len(all_subjects))
	y = results['sorted_h1h2_base_diff'][:,d]
	plt.scatter(x, y, s, c=colors[0], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(results['sorted_h1h2_base_diff'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[0], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[0], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_h1h2_base_diff'][0,d]
	ci_up = results['ci_sorted_h1h2_base_diff'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[0],
		elinewidth=5, capsize=0)

	# low_1_low_2
	# Plot the single subjects univariate responses
	x = np.repeat(dist+x_dist_within*0.5, len(all_subjects))
	y = results['sorted_l1l2_base_diff'][:,d]
	plt.scatter(x, y, s, c=colors[1], alpha=alpha, label='_nolegend_')
	# Plot the subject-average univariate responses
	x = x[0]
	y = np.mean(results['sorted_l1l2_base_diff'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[1], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[1], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_l1l2_base_diff'][0,d]
	ci_up = results['ci_sorted_l1l2_base_diff'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[1],
		elinewidth=5, capsize=0)

# Plot trend connectors
# high_1_high_2
if results['sorted_h1h2_base_diff_between_subject_pval'] < 0.05:
	x = area_distance - (x_dist_within * 0.5)
	y = np.mean(results['sorted_h1h2_base_diff'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[0],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[0],
		alpha=0.5)
# low_1_low_2
if results['sorted_l1l2_base_diff_between_subject_pval'] < 0.05:
	x = area_distance + (x_dist_within * 0.5)
	y = np.mean(results['sorted_l1l2_base_diff'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[1],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[1],
		alpha=0.5)
# high_1_low_2
if results['sorted_h1l2_base_diff_between_subject_pval'] < 0.05:
	x = area_distance - (x_dist_within * 0.5)
	y = np.mean(results['sorted_h1l2_base_diff'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[2],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[2],
		alpha=0.5)
# low_1_high_2
if results['sorted_l1h2_base_diff_between_subject_pval'] < 0.05:
	x = area_distance + (x_dist_within * 0.5)
	y = np.mean(results['sorted_l1h2_base_diff'], 0)
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
ylabel = 'Absolute Δ\nunivariate response'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=1.5)

# Save the figure
file_name = 'univariate_rnc_cortical_distance_responses_encoding_models_' + \
	'train_dataset-' + args.encoding_models_train_dataset + '_imageset-' + \
	args.imageset + '.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the pairwise ROI combinations univariate response correlations as a
# function of cortical distance (only for ROIs: V1, V2, V3, hV4)
# =============================================================================
# Plot parameters
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
color = 'k'

fig = plt.figure(figsize=(5, 8))

for d, dist in enumerate(area_distance):

	# Plot the single subjects correlation scores
	x = np.repeat(dist, len(all_subjects))
	y = results['sorted_corr'][:,d]
	plt.scatter(x, y, s, c=color, alpha=alpha)

	# Plot the subject-average correlation scores
	x = dist
	y = np.mean(results['sorted_corr'][:,d])
	plt.scatter(x, y, s_mean, c=color)

	# Plot the CIs
	ci_low = y - results['ci_sorted_corr'][0,d]
	ci_up = results['ci_sorted_corr'][1,d] - y
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
if results['sorted_corr_between_subject_pval'] < 0.05:
	x = area_distance
	y = np.mean(results['sorted_corr'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=color, alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=color, alpha=0.5)

# Save the figure
file_name = 'univariate_rnc_cortical_distance_corr_encoding_models_train_' + \
	'dataset-' + args.encoding_models_train_dataset + '_imageset-' + \
	args.imageset + '.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
