"""Plot the univariate RNC controlling images effect on the in vivo fMRI data.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
subjects : list
	List of used subjects.
rois : list
	List of used ROIs.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=str)
parser.add_argument('--rois', default=['V1', 'V4'], type=list)
parser.add_argument('--project_dir', default='./relational_neural_control_old/', type=str)
args = parser.parse_args()


# =============================================================================
# Load the univariate RNC results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'in_vivo_validation',
	'univariate_rnc_experiment', 'univariate_rnc_experiment_results.npy')

data = np.load(data_dir, allow_pickle=True).item()


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
s = 1000
s_all = 100
alpha = 0.8

fig = plt.figure(figsize=(6,6))

# Diagonal dashed line
plt.plot(np.arange(-3,3), np.arange(-3,3), '--k', linewidth=2, alpha=.4,
	label='_nolegend_')

# Baseline dashed lines
baseline_v1 = np.mean(
	data['control_scores']['baseline_v1'])
plt.plot([baseline_v1, baseline_v1], [-3, 3], '--w', linewidth=2, alpha=.6,
	label='Baseline V1')
baseline_v4 = np.mean(
	data['control_scores']['baseline_v4'])
plt.plot([-3, 3], [baseline_v4, baseline_v4], '--w', linewidth=2, alpha=.6,
	label='Baseline V4')

# Univariate responses for all images
for sub in range(len(args.subjects)):
	plt.scatter(np.mean(data['betas']['V1'][sub], 1),
		np.mean(data['betas']['V4'][sub], 1), s=s_all, color='w', alpha=.1,
		edgecolors='k', linewidths=4, label='_nolegend_')

# Univariate responses for the controlling images
for sub in range(len(args.subjects)):
	# 1. high_V1_high_V4
	plt.scatter(data['control_scores']['h1h2_scores']['V1'][sub],
		data['control_scores']['h1h2_scores']['V4'][sub], s=s, color=colors[0],
		alpha=alpha, label='High V1, High V4')
	# 2. low_V1_low_V4
	plt.scatter(data['control_scores']['l1l2_scores']['V1'][sub],
		data['control_scores']['l1l2_scores']['V4'][sub], s=s, color=colors[1],
		alpha=alpha, label='Low V1, Low V4')
	# 3. high_V1_low_V4
	plt.scatter(data['control_scores']['h1l2_scores']['V1'][sub],
		data['control_scores']['h1l2_scores']['V4'][sub], s=s, color=colors[2],
		alpha=alpha, label='High V1, Low V4')
	# 4. low_V1_high_V4
	plt.scatter(data['control_scores']['l1h2_scores']['V1'][sub],
		data['control_scores']['l1h2_scores']['V4'][sub], s=s, color=colors[3],
		alpha=alpha, label='Low V1, High V4')

# x-axis parameters
plt.xlabel('V1 univariate\nresponse', fontsize=fontsize)
xticks = [-1, 0, 1]
xlabels = [-1, 0, 1]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=-1.6, right=1.6)

# y-axis parameters
plt.ylabel('V4 univariate\nresponse', fontsize=fontsize)
yticks = [-1, 0, 1]
ylabels = [-1, 0, 1]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-1.6, top=1.6)

fig.savefig('univariate_rnc_scatterplots_v1_v4_in_vivo_validation.svg',
	format='svg', transparent=True, bbox_inches='tight')


# =============================================================================
# Plot the univariate responses significance for the controlling images
# =============================================================================
fig = plt.figure(figsize=(6,6))

s_mean = 800
s = 400
lim_min = -1.75
lim_max = 1.25
padding = 0.4
x_dist = (abs(lim_min - lim_max) - (padding*2)) / 3
x_dist_within = float(0.25)
x_start = lim_min + padding
xticks = np.asarray((x_start, x_start+x_dist*1, x_start+x_dist*2,
	x_start+x_dist*3))
x_coord = xticks - (x_dist_within / 2)
alpha = 0.2
sig_star_offset_top = 0.11
sig_star_offset_bottom = 0.16
fontsize_sig = 25
marker_roi_1 = 'd'
marker_roi_2 = 's'
baseline_width = 0.1
control_responses = ['h1h2', 'l1l2', 'h1l2', 'l1h2']

for c, cond in enumerate(control_responses):

	# V1 (baseline)
	x_baseline = np.repeat(x_coord[c], len(args.subjects))
	y = data['control_scores']['baseline_v1']
	plt.plot([x_baseline[0]-baseline_width, x_baseline[0]+baseline_width],
		[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
		alpha=.4)

	# V1 (control scores)
	x = np.repeat(x_coord[c], len(args.subjects))
	x_score = x[0]
	y = data['control_scores'][cond+'_scores']['V1']
	plt.scatter(x, y, marker=marker_roi_1, s=s, color=colors[c],
		alpha=alpha)
	plt.scatter(x[0], np.mean(y), marker=marker_roi_1, s=s_mean,
		color=colors[c])
	# V1 (control CIs)
	ci_low = np.mean(y) - data['confidence_intervals'][cond+'_ci']['V1'][0]
	ci_up = data['confidence_intervals'][cond+'_ci']['V1'][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[c], elinewidth=5, capsize=0)

	# V1 (significance)
	if data['significance'][cond+'_between_subject_pval']['V1'] < 0.05:
		if c in [0, 2]:
			y = max(data['control_scores'][cond+'_scores']['V1']) + \
				sig_star_offset_top
			plt.text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
				fontweight='bold', ha='center', va='center')
		elif c in [1, 3]:
			y = min(data['control_scores'][cond+'_scores']['V1']) - \
				sig_star_offset_bottom
			plt.text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
				fontweight='bold', ha='center', va='center')

	# V4 (baseline)
	x_baseline = x + x_dist_within
	y = data['control_scores']['baseline_v4']
	plt.plot([x_baseline[0]-baseline_width, x_baseline[0]+baseline_width],
		[np.mean(y), np.mean(y)], color='k', linestyle='--', linewidth=2,
		alpha=.4)

	# V4 (control scores)
	x += x_dist_within
	x_score = x[0]
	y = data['control_scores'][cond+'_scores']['V4']
	plt.scatter(x, y, marker=marker_roi_2, s=s, color=colors[c],
		alpha=alpha)
	plt.scatter(x[0], np.mean(y), marker=marker_roi_2, s=s_mean,
		color=colors[c])
	# V4 (control CIs)
	ci_low = np.mean(y) - data['confidence_intervals'][cond+'_ci']['V4'][0]
	ci_up = data['confidence_intervals'][cond+'_ci']['V4'][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[c], elinewidth=5, capsize=0)

	# V4 (significance)
	if data['significance'][cond+'_between_subject_pval']['V4'] < 0.05:
		if c in [0, 3]:
			y = max(data['control_scores'][cond+'_scores']['V4']) + \
				sig_star_offset_top
			plt.text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
				fontweight='bold', ha='center', va='center')
		elif c in [1, 2]:
			y = min(data['control_scores'][cond+'_scores']['V4']) - \
				sig_star_offset_bottom
			plt.text(x_score, y, s='*', fontsize=fontsize_sig, color='k',
				fontweight='bold', ha='center', va='center')

# x-axis parameters
xlabels = ['', '', '', '']
plt.xticks(ticks=xticks, labels=xlabels, rotation=45)
xlabel = 'Neural control\nconditions'
plt.xlabel(xlabel, fontsize=fontsize)

# y-axis parameters
ylabel = 'Univariate\nresponse'
plt.ylabel(ylabel, fontsize=fontsize)
yticks = [-0.5, 0, 0.5]
ylabels = [-0.5, 0, 0.5]
plt.yticks(ticks=yticks, labels=ylabels)

# Limits
plt.ylim(bottom=-.95, top=.78)
plt.xlim(left=lim_min, right=lim_max)

fig.savefig('univariate_rnc_significance_v1_v4_in_vivo_validation.svg',
	format='svg', transparent=True, bbox_inches='tight')
