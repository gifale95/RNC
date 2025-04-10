"""Plot the multivariate RNC controlling images effect on the in vivo fMRI data.

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
parser.add_argument('--project_dir', default='../relational_neural_control_old/', type=str)
args = parser.parse_args()


# =============================================================================
# Load the multivariate RNC results
# =============================================================================
data_dir = os.path.join(args.project_dir, 'in_vivo_validation',
	'multivariate_rnc_experiment', 'multivariate_rnc_experiment_results.npy')

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
colors = [(108/255, 0/255, 158/255), (153/255, 153/255, 153/255),
	(240/255, 84/255, 0/255)]


# =============================================================================
# Plot the multivariate control scores
# =============================================================================
x_dist_within = float(0.2)
alpha = 0.2
sig_offset = 0.1
sig_bar_length = 0.03
linewidth_sig_bar = 1.5
sig_star_offset_top = 0.03
sig_star_offset_bottom = 0.07
s = 800
fontsize_sig = 25

fig = plt.figure(figsize=(4,6))

# Alignment images (scores)
x = np.repeat(1-x_dist_within, len(args.subjects))
y = data['control_scores']['rsa_alignemnt_scores']
plt.scatter(x, y, s=s, color=colors[0], alpha=alpha)
label = '↑ $r$ (Align)'
plt.scatter(x[0], np.mean(y), s=s, color=colors[0], label=label)

# Alignment images (CIs)
ci_low = np.mean(y) - data['confidence_intervals']['rsa_alignment_ci'][0]
ci_up = data['confidence_intervals']['rsa_alignment_ci'][1] - np.mean(y)
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
	ecolor=colors[0], elinewidth=5, capsize=0)

# Alignment images (significance)
if data['significance']['rsa_alignment_between_subject_pval'] < 0.05:
	y_max = max(y) + sig_offset
	plt.plot([x[0], x[0]], [y_max, y_max+sig_bar_length], 'k-',
		[x[0], 1], [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
		[1, 1], [y_max+sig_bar_length, y_max], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x[0], 1)))
	y = y_max + sig_bar_length + sig_star_offset_top
	plt.text(x_mean, y, s='*', fontsize=fontsize_sig, color='k',
		fontweight='bold', ha='center', va='center')

# Baseline images (scores)
x = np.repeat(1, len(args.subjects))
y = data['control_scores']['rsa_baseline_scores']
plt.scatter(x, y, s=s, color=colors[1], alpha=alpha)
label = 'Baseline'
plt.scatter(x[0], np.mean(y), s=s, color=colors[1], label=label)

# Baseline images (CIs)
ci_low = np.mean(y) - data['confidence_intervals']['rsa_baseline_ci'][0]
ci_up = data['confidence_intervals']['rsa_baseline_ci'][1] - np.mean(y)
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
	ecolor=colors[1], elinewidth=5, capsize=0)

# Disentanglement images (scores)
x = np.repeat(1+x_dist_within, len(args.subjects))
y = data['control_scores']['rsa_disentanglement_scores']
plt.scatter(x, y, s=s, color=colors[2], alpha=alpha)
label = '↓ $r$ (Disentangle)'
plt.scatter(x[0], np.mean(y), s=s, color=colors[2], label=label)

# Disentanglement images (CIs)
ci_low = np.mean(y) - data['confidence_intervals']['rsa_disentanglement_ci'][0]
ci_up = data['confidence_intervals']['rsa_disentanglement_ci'][1] - np.mean(y)
conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
	ecolor=colors[2], elinewidth=5, capsize=0)

# Disentanglement images (significance)
if data['significance']['rsa_disentanglement_between_subject_pval'] < 0.05:
	y_min = min(y) - sig_offset
	plt.plot([x[0], x[0]], [y_min, y_min-sig_bar_length], 'k-',
		[x[0], 1], [y_min-sig_bar_length, y_min-sig_bar_length], 'k-',
		[1, 1], [y_min-sig_bar_length, y_min], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x[0], 1)))
	y = y_min - sig_bar_length - sig_star_offset_bottom
	plt.text(x_mean, y, s='*', fontsize=fontsize_sig, color='k',
		fontweight='bold', ha='center', va='center')

# y-axis parameters
plt.ylabel('Pearson\'s $r$', fontsize=fontsize)
plt.ylim(top=1.15, bottom=-0.05)

# x-axis parameters
xticks = [1]
labels = ['V1-V4']
plt.xticks(ticks=xticks, labels=labels, fontsize=fontsize)
plt.xlim(left=0.6, right=1.4)

# Legend
plt.legend(loc=3, ncol=1, fontsize=fontsize)

fig.savefig('multivariate_rnc_v1_v4_in_vivo_validation.svg', format='svg',
	transparent=True, bbox_inches='tight')
