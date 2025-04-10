"""Plot the multivariate RNC cross-subject validated results.

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
from matplotlib.lines import Line2D

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

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc',
		args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-1',
		'imageset-'+args.imageset, r, 'stats.npy')

	stats[r] = np.load(data_dir, allow_pickle=True).item()


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
colors = [(108/255, 0/255, 158/255), (153/255, 153/255, 153/255),
	(240/255, 84/255, 0/255)]


# =============================================================================
# Plot the multivariate RNC results from the final genetic optimization
# generation
# =============================================================================
# Plot parameters
x_dist_within = float(0.2)
alpha = 0.2
sig_offset = 0.05
sig_bar_length = 0.03
linewidth_sig_bar = 1
sig_star_offset_top = 0.02
sig_star_offset_bottom = 0.04
s = 600
s_mean = 800

# Plot
fig = plt.figure(figsize=(10,12))

for r, stats_roi in enumerate(stats.values()):

	# Aligning images RSA (scores)
	x = np.repeat(r+1-x_dist_within, len(all_subjects))
	y = stats_roi['best_generation_scores_test']['align'][:,-1]
	plt.scatter(x, y, s=s, color=colors[0], alpha=alpha)
	if r == 0:
		label = '↑ $r$ (Align)'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0])
	# Aligning images RSA (CIs)
	ci_low = np.mean(y) - stats_roi['ci_align'][0]
	ci_up = stats_roi['ci_align'][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[0], elinewidth=5, capsize=0)
	# Aligning images RSA (significance)
	if stats_roi['rsa_alignment_between_subject_pval'] < 0.05:
		y_max = max(y) + sig_offset
		plt.plot([x[0], x[0]], [y_max, y_max+sig_bar_length], 'k-',
			[x[0], r+1], [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
			[r+1, r+1], [y_max+sig_bar_length, y_max], 'k-',
			linewidth=linewidth_sig_bar)
		x_mean = np.mean(np.asarray((x[0], r+1)))
		y = y_max + sig_bar_length + sig_star_offset_top
		plt.text(x_mean, y, s='*', fontsize=30, color='k',
			fontweight='bold', ha='center', va='center')

	# Baseline images RSA (scores)
	x = np.repeat(r+1, len(all_subjects))
	y = stats_roi['baseline_images_score_test']
	plt.scatter(x, y, s=s, color=colors[1], alpha=alpha)
	if r == 0:
		label = 'Baseline'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1])
	# Baseline images RSA (CIs)
	ci_low = np.mean(y) - stats_roi['ci_baseline'][0]
	ci_up = stats_roi['ci_baseline'][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[1], elinewidth=5, capsize=0)

	# Disentangling images RSA (scores)
	x = np.repeat(r+1+x_dist_within, len(all_subjects))
	y = stats_roi['best_generation_scores_test']['disentangle'][:,-1]
	plt.scatter(x, y, s=s, color=colors[2], alpha=alpha)
	if r == 0:
		label = '↓ $r$ (Disentangle)'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2])
	# Disentangling images RSA (CIs)
	ci_low = np.mean(y) - stats_roi['ci_disentangle'][0]
	ci_up = stats_roi['ci_disentangle'][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[2], elinewidth=5, capsize=0)
	# Disentangling images RSA (significance)
	if stats_roi['rsa_disentanglement_between_subject_pval'] < 0.05:
		y_min = min(y) - sig_offset
		plt.plot([x[0], x[0]], [y_min, y_min-sig_bar_length], 'k-',
			[x[0], r+1], [y_min-sig_bar_length, y_min-sig_bar_length], 'k-',
			[r+1, r+1], [y_min-sig_bar_length, y_min], 'k-',
			linewidth=linewidth_sig_bar)
		x_mean = np.mean(np.asarray((x[0], r+1)))
		y = y_min - sig_bar_length - sig_star_offset_bottom
		plt.text(x_mean, y, s='*', fontsize=30, color='k',
			fontweight='bold', ha='center', va='center')

# y-axis parameters
plt.ylabel('Pearson\'s $r$', fontsize=fontsize)
plt.ylim(top=1.15, bottom=-0.19)

# x-axis parameters
xticks = np.arange(1, len(args.roi_pairs)+1)
labels = args.roi_pairs
plt.xticks(ticks=xticks, labels=labels, rotation=45, fontsize=fontsize)
plt.xlim(left=0.5, right=6.5)

# Legend
plt.legend(loc=3, ncol=3, fontsize=fontsize)

# Save the figure
file_name = 'multivariate_rnc_significance_encoding_models_train_dataset-' + \
	args.encoding_models_train_dataset + '_imageset-' + args.imageset + '_.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the multivariate RNC optimization curves
# =============================================================================
# Plot parameters
fontsize = 20
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)

# Make the figure
fig, axs = plt.subplots(len(args.roi_pairs), len(all_subjects), sharex=True,
	sharey=True)

# Plot
for r, stats_roi in enumerate(stats.values()):

	for s, sub in enumerate(all_subjects):

		# Title
		if r == 0:
			title = 'Subject ' + str(sub)
			axs[r,s].set_title(title, fontsize=fontsize)

		# x-axis
		if r == len(args.roi_pairs)-1:
			axs[r,s].set_xlabel('Generations', fontsize=fontsize)
			xticks = [1000]
			xlabels = ['1,000']
			axs[r,s].set_xticks(ticks=xticks, labels=xlabels)

		# x-axis
		if s == 0:
			y_label = args.roi_pairs[r] + '\nPearson\'s $r$'
			axs[r,s].set_ylabel(y_label, fontsize=fontsize)

		x = np.arange(stats_roi['best_generation_scores_test']['align'].shape[1])

		# Plot the training curves (alignment)
		axs[r,s].plot(x, stats_roi['best_generation_scores_train']['align'][s],
			linewidth=1, color=colors[0])

		# Plot the test curves (alignment)
		axs[r,s].plot(x, stats_roi['best_generation_scores_test']['align'][s],
			'--', linewidth=1, color=colors[0])

		# Plot the baseline images scores
		control_scores = stats_roi['baseline_images_score_test'][s]
		axs[r,s].plot([x[0], x[-1]], [control_scores, control_scores], '--',
			linewidth=1, color=colors[1])

		# Plot the train curves (disentanglement)
		axs[r,s].plot(x,
			stats_roi['best_generation_scores_train']['disentangle'][s],
			linewidth=1, color=colors[2])

		# Plot the test curves (disentanglement)
		axs[r,s].plot(x,
			stats_roi['best_generation_scores_test']['disentangle'][s],
			'--', linewidth=1, color=colors[2])

		# Legend
		if r == 0 and s == 0:
			# Create custom lines with increased line width for the legend
			custom_lines = [
				Line2D([0], [0], color=colors[0], lw=4),
				Line2D([0], [0], linewidth=4, color=colors[0], linestyle='--'),
				Line2D([0], [0], linewidth=4, color=colors[1], linestyle='--'),
				Line2D([0], [0], linewidth=4, color=colors[2]),
				Line2D([0], [0], linewidth=4, color=colors[2], linestyle='--')
				]
			legend = [
				'↑ $r$ (train)',
				'↑ $r$ (test)',
				'Baseline',
				'↓ $r$ (train)',
				'↓ $r$ (test)'
				]
			axs[r,s].legend(custom_lines, legend, loc=2, ncol=5,
				fontsize=fontsize, bbox_to_anchor=(1.525, -6.5), frameon=False,
				markerscale=2)

		# Limits
		axs[r,s].set_xlim(min(x), max(x))
		axs[r,s].set_ylim(bottom=-.05, top=1)

# Save the figure
file_name = 'multivariate_rnc_optimization_curves_encoding_models_train_' + \
	'dataset-' + args.encoding_models_train_dataset + '_imageset-' + \
	args.imageset + '_.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Load the cortical distance analysis results
# =============================================================================
data_dir = save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'cortical_distance_analysis', 'cv-1', 'imageset-'+args.imageset,
	'cortical_distance_analysis.npy')

results = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot the multivariate RNC RSA scores as a function of cortical distance
# =============================================================================
# Plot parameters
fontsize = 40
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
x_dist_within = float(0.175)

# Plot
fig = plt.figure(figsize=(10,12))

for d, dist in enumerate(area_distance):

	# Alignment neural control condition
	# Plot the single subjects RSA scores
	x = np.repeat(dist-x_dist_within*1, len(all_subjects))
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
	x = np.repeat(dist, len(all_subjects))
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
	x = np.repeat(dist+x_dist_within*1, len(all_subjects))
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
	x = area_distance - (x_dist_within * 1)
	y = np.mean(results['sorted_align'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[0],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[0],
		alpha=0.5)
# Baseline
if results['sorted_baseline_between_subject_pval'] < 0.05:
	x = area_distance
	y = np.mean(results['sorted_baseline'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[1],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[1],
		alpha=0.5)
# Disentanglement
if results['sorted_disentangle_between_subject_pval'] < 0.05:
	x = area_distance + (x_dist_within * 1)
	y = np.mean(results['sorted_disentangle'], 0)
	plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[2],
		alpha=0.5)
	plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[2],
		alpha=0.5)

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
plt.ylim(top=1.15, bottom=-0.19)

# Save the figure
file_name = 'multivariate_rnc_cortical_distance_analysis_encoding_models_' + \
	'train_dataset-' + args.encoding_models_train_dataset + '_imageset-' + \
	args.imageset + '_.svg'
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
