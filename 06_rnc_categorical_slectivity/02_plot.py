"""Plot the categorical selectiviy analysis results.

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
n_iter : int
	Amount of iterations for creating confidence intervals bootstrapped
	distribution.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
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
# Set the plot parameters
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


# =============================================================================
# Plot the difference of the univariate response for the controlling images
# from baseline, as a function of two areas being from the same or different
# categorical selectivity groups
# =============================================================================
# Load the categorical selectivity analysis results
data_dir = save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'categorical_selectivity_analysis', 'imageset-'+args.imageset,
	'categorical_selectivity_analysis.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Plot parameters
alpha = 0.2
groups = np.asarray((1, 2))
s = 600
s_mean = 800
x_dist_within = float(0.325)
sig_offset = 0.1
sig_bar_length = 0.03
sig_star_offset_top = 0.025
colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
	(217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]

fig = plt.figure(figsize=(8, 8))

for d, group in enumerate(groups):

	# high_1_high_2
	# Plot the single subjects univariate responses
	if d == 0:
		x = np.repeat(group-x_dist_within*0.75, len(all_subjects))
	else:
		x = np.repeat(group+x_dist_within*0.25, len(all_subjects))
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
	if d == 0:
		x = np.repeat(group-x_dist_within*0.25, len(all_subjects))
	else:
		x = np.repeat(group+x_dist_within*0.75, len(all_subjects))
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


	# high_1_low_2
	# Plot the single subjects univariate responses
	if d == 0:
		x = np.repeat(group+x_dist_within*0.25, len(all_subjects))
	else:
		x = np.repeat(group-x_dist_within*0.75, len(all_subjects))
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
	if d == 0:
		x = np.repeat(group+x_dist_within*0.75, len(all_subjects))
	else:
		x = np.repeat(group-x_dist_within*0.25, len(all_subjects))
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

# Plot trend connectors
# high_1_high_2
if results['sorted_h1h2_base_diff_between_subject_pval'] < 0.05:
	y = np.mean(results['sorted_h1h2_base_diff'], 0)
	x1 = groups[0] - (x_dist_within * 0.75)
	x2 = groups[1] + (x_dist_within * 0.25)
	plt.plot([x1, x2], [y[0], y[1]], linewidth=2, color=colors[0],
		alpha=0.5)
# low_1_low_2
if results['sorted_l1l2_base_diff_between_subject_pval'] < 0.05:
	x1 = groups[0] - (x_dist_within * 0.25)
	x2 = groups[1] + (x_dist_within * 0.75)
	y = np.mean(results['sorted_l1l2_base_diff'], 0)
	plt.plot([x1, x2], [y[0], y[1]], linewidth=2, color=colors[1],
		alpha=0.5)
# high_1_low_2
if results['sorted_h1l2_base_diff_between_subject_pval'] < 0.05:
	x1 = groups[0] + (x_dist_within * 0.25)
	x2 = groups[1] - (x_dist_within * 0.75)
	y = np.mean(results['sorted_h1l2_base_diff'], 0)
	plt.plot([x1, x2], [y[0], y[1]], linewidth=2, color=colors[2],
		alpha=0.5)
# low_1_high_2
if results['sorted_l1h2_base_diff_between_subject_pval'] < 0.05:
	x1 = groups[0] + (x_dist_within * 0.75)
	x2 = groups[1] - (x_dist_within * 0.25)
	y = np.mean(results['sorted_l1h2_base_diff'], 0)
	plt.plot([x1, x2], [y[0], y[1]], linewidth=2, color=colors[3],
		alpha=0.5)

# x-axis parameters
xticks = [1, 2]
xlabels = ['Within\ngroup', 'Between\ngroup']
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=0.6, right=2.4)

# y-axis parameters
yticks = [0, 0.5, 1]
ylabels = [0, 0.5, 1]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Absolute Δ\nunivariate response'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=1.2)

# Save the figure
file_name = 'univariate_rnc_categorical_selectivity_analysis_encoding_' + \
	'models_train_dataset-' + args.encoding_models_train_dataset + \
	'_imageset-' + args.imageset + '.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the multivariate RNC RSA scores as a function of two areas being from the
# same or different categorical selectivity groups
# =============================================================================
# Load the categorical selectivity analysis results
data_dir = save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'categorical_selectivity_analysis', 'cv-1', 'imageset-'+args.imageset,
	'categorical_selectivity_analysis.npy')
results = np.load(data_dir, allow_pickle=True).item()

# Plot parameters
alpha = 0.2
groups = np.asarray((1, 2))
s = 600
s_mean = 800
x_dist_within = float(0.175)
colors = [(108/255, 0/255, 158/255), (153/255, 153/255, 153/255),
	(240/255, 84/255, 0/255)]

# Plot
fig = plt.figure(figsize=(8, 8))

for d, group in enumerate(groups):

	# Alignment neural control condition
	# Plot the single subjects RSA scores
	x = np.repeat(group-x_dist_within*1, len(all_subjects))
	y = results['sorted_align'][:,d]
	plt.scatter(x, y, s, c=colors[0], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(results['sorted_align'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[0], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[0], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_align'][0,d]
	ci_up = results['ci_sorted_align'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[0],
		elinewidth=5, capsize=0)

	# Baseline images
	# Plot the single subjects RSA scores
	x = np.repeat(group, len(all_subjects))
	y = results['sorted_baseline'][:,d]
	plt.scatter(x, y, s, c=colors[1], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(results['sorted_baseline'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[1], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[1], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_baseline'][0,d]
	ci_up = results['ci_sorted_baseline'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[1],
		elinewidth=5, capsize=0)

	# Disentanglement neural control condition
	# Plot the single subjects RSA scores
	x = np.repeat(group+x_dist_within*1, len(all_subjects))
	y = results['sorted_disentangle'][:,d]
	plt.scatter(x, y, s, c=colors[2], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(results['sorted_disentangle'][:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[2], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[2], label='_nolegend_')
	# Plot the CIs
	ci_low = y - results['ci_sorted_disentangle'][0,d]
	ci_up = results['ci_sorted_disentangle'][1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[2],
		elinewidth=5, capsize=0)

# Plot trend connectors
# Alignment
if results['sorted_align_between_subject_pval'] < 0.05:
	x = groups - (x_dist_within * 1)
	y = np.mean(results['sorted_align'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[0],
		alpha=0.5)
# Baseline
if results['sorted_baseline_between_subject_pval'] < 0.05:
	x = groups
	y = np.mean(results['sorted_baseline'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[1],
		alpha=0.5)
# Disentanglement
if results['sorted_disentangle_between_subject_pval'] < 0.05:
	x = groups + (x_dist_within * 1)
	y = np.mean(results['sorted_disentangle'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[2],
		alpha=0.5)

# x-axis parameters
xticks = [1, 2]
xlabels = ['Within\ngroup', 'Between\ngroup']
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=0.6, right=2.4)

# y-axis parameters
yticks = [0, 0.5, 1]
ylabels = [0, 0.5, 1]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Pearson\'s $r$'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(top=1.1, bottom=-0.1)

# Save the figure
file_name = 'multivariate_rnc_categorical_selectivity_analysis_encoding_' + \
	'models_train_dataset-' + args.encoding_models_train_dataset + \
	'_imageset-' + args.imageset + '.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
