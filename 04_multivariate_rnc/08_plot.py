"""Plot the multivariate RNC cross-subject validated results.

This code is available at:
https://github.com/gifale95/RNC/04_multivariate_rnc/08_plot.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
roi_pair : list
	List of ROI pairs for which to plot the results.
control_conditions : list
	List of used control conditions.
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
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import page_trend_test

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--roi_pairs', type=list, default=['V1-V2', 'V1-V3', 'V1-hV4', 'V2-V3', 'V2-hV4', 'V3-hV4'])
parser.add_argument('--control_conditions', type=list, default=['align', 'disentangle'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()


# =============================================================================
# Set the plot parameters
# =============================================================================
fontsize = 50
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
colors = [(108/255, 0/255, 158/255), (153/255, 153/255, 153/255),
	(240/255, 84/255, 0/255)]


# =============================================================================
# Plot the multivariate RNC results from the final genetic optimization
# generation
# =============================================================================
# Load the stats
scores_align = {}
scores_disentangle = {}
baseline_scores = {}
ci_align = {}
ci_disentangle = {}
ci_baseline = {}
sig_align = {}
sig_disentangle = {}

for r in args.roi_pairs:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats',
		'cv-1', 'imageset-'+args.imageset, r, 'stats.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()

	scores_align_all_sub = []
	scores_disentangle_all_sub = []
	baseline_scores_all_sub = []

	for s, sub in enumerate(args.all_subjects):
		scores_align_all_sub.append(
			data_dict['best_generation_scores_test']['align'][s,-1])
		scores_disentangle_all_sub.append(
			data_dict['best_generation_scores_test']['disentangle'][s,-1])
		baseline_scores_all_sub.append(data_dict['baseline_images_score_test'][s])

	scores_align[r] = np.asarray(scores_align_all_sub)
	scores_disentangle[r] = np.asarray(scores_disentangle_all_sub)
	baseline_scores[r] = np.asarray(baseline_scores_all_sub)
	ci_align[r] = data_dict['ci_align']
	ci_disentangle[r] = data_dict['ci_disentangle']
	ci_baseline[r] = data_dict['ci_baseline']
	sig_align[r] = data_dict['significance_align']
	sig_disentangle[r] = data_dict['significance_disentangle']

# Plot parameters
x_dist_within = float(0.2)
alpha = 0.2
sig_offset = 0.05
sig_bar_length = 0.03
linewidth_sig_bar = 1.5
sig_star_offset_top = 0.02
sig_star_offset_bottom = 0.04
s = 600
s_mean = 800

# Plot
plt.figure()

for r, roi in enumerate(args.roi_pairs):

	# Aligning images RSA (scores)
	x = np.repeat(r+1-x_dist_within, len(args.all_subjects))
	y = scores_align[roi]
	plt.scatter(x, y, s=s, color=colors[0], alpha=alpha)
	if r == 0:
		label = '↑ $r$ (Align)'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[0])
	# Aligning images RSA (CIs)
	ci_low = np.mean(y) - ci_align[roi][0]
	ci_up = ci_align[roi][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[0], elinewidth=5, capsize=0)
	# Aligning images RSA (significance)
	if sig_align[roi] == True:
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
	x = np.repeat(r+1, len(args.all_subjects))
	y = baseline_scores[roi]
	plt.scatter(x, y, s=s, color=colors[1], alpha=alpha)
	if r == 0:
		label = 'Baseline'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[1])
	# Baseline images RSA (CIs)
	ci_low = np.mean(y) - ci_baseline[roi][0]
	ci_up = ci_baseline[roi][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[1], elinewidth=5, capsize=0)

	# Disentangling images RSA (scores)
	x = np.repeat(r+1+x_dist_within, len(args.all_subjects))
	y = scores_disentangle[roi]
	plt.scatter(x, y, s=s, color=colors[2], alpha=alpha)
	if r == 0:
		label = '↓ $r$ (Disentangle)'
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2],
			label=label)
	else:
		plt.scatter(x[0], np.mean(y), s=s_mean, color=colors[2])
	# Disentangling images RSA (CIs)
	ci_low = np.mean(y) - ci_disentangle[roi][0]
	ci_up = ci_disentangle[roi][1] - np.mean(y)
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x[0], np.mean(y), yerr=conf_int, fmt="none",
		ecolor=colors[2], elinewidth=5, capsize=0)
	# Disentangling images RSA (significance)
	if sig_disentangle[roi] == True:
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
xticks = [1, 2, 3, 4, 5, 6]
labels = ['V1-V2', 'V1-V3', 'V1-V4', 'V2-V3', 'V2-V4', 'V3-V4']
plt.xticks(ticks=xticks, labels=labels, rotation=45, fontsize=fontsize)
plt.xlim(left=0.5, right=6.5)

# Legend
plt.legend(loc=3, ncol=3, fontsize=fontsize)

#fig.savefig('multivariate_rnc_significance_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_significance_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_significance_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the multivariate RNC RSA scores as a function of cortical distance
# =============================================================================
# Load the stats
scores_align = []
scores_disentangle = []
baseline_scores = []

for r in args.roi_pairs:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats',
		'cv-1', 'imageset-'+args.imageset, r, 'stats.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()

	scores_align_all_sub = []
	scores_disentangle_all_sub = []
	baseline_scores_all_sub = []
	for s, sub in enumerate(args.all_subjects):
		scores_align_all_sub.append(
			data_dict['best_generation_scores_test']['align'][s,-1])
		scores_disentangle_all_sub.append(
			data_dict['best_generation_scores_test']['disentangle'][s,-1])
		baseline_scores_all_sub.append(
			data_dict['baseline_images_score_test'][s])

	scores_align.append(np.asarray(scores_align_all_sub))
	scores_disentangle.append(np.asarray(scores_disentangle_all_sub))
	baseline_scores.append(np.asarray(baseline_scores_all_sub))

scores_align = np.transpose(np.asarray(scores_align))
scores_disentangle = np.transpose(np.asarray(scores_disentangle))
baseline_scores = np.transpose(np.asarray(baseline_scores))

# There are three cortical distances:
# Cortical distance 1: [V1 vs. V2; V2 vs. V3; V3 vs. V4]
# Cortical distance 2: [V1 vs. V3; V2 vs. V4]
# Cortical distance 3: [V1 vs. V4]
cortical_distances = [(0, 3, 5), (1, 4), (2)]

# Sorted scores arrays of shape:
# (Subjects x ROI cortical distances)
sorted_align = np.zeros((len(args.all_subjects), len(cortical_distances)))
sorted_disentangle = np.zeros((len(args.all_subjects), len(cortical_distances)))
sorted_baseline = np.zeros((len(args.all_subjects), len(cortical_distances)))

# Group the multivariate RNC correlation scores as a function of cortical
# distance
for d, dist in enumerate(cortical_distances):
	if type(dist) == tuple:
		sorted_align[:,d] = np.mean(scores_align[:,dist], 1)
		sorted_disentangle[:,d] = np.mean(scores_disentangle[:,dist], 1)
		sorted_baseline[:,d] = np.mean(baseline_scores[:,dist], 1)
	else:
		sorted_align[:,d] = scores_align[:,dist]
		sorted_disentangle[:,d] = scores_disentangle[:,dist]
		sorted_baseline[:,d] = baseline_scores[:,dist]

# Compute the 95% confidence intervals
# CI arrays of shape:
# (CI percentiles x Cortical distances)
ci_sorted_corr = np.zeros((2, len(cortical_distances)))
ci_sorted_dec = np.zeros((2, len(cortical_distances)))
ci_sorted_null = np.zeros((2, len(cortical_distances)))
for d in tqdm(range(len(cortical_distances)), leave=False):
	# Empty CI distribution array
	sorted_corr_dist = np.zeros((args.n_iter))
	sorted_dec_dist = np.zeros((args.n_iter))
	sorted_null_dist = np.zeros((args.n_iter))
	# Compute the CI distribution
	for i in range(args.n_iter):
		idx_resample = resample(np.arange(len(args.all_subjects)))
		sorted_corr_dist[i] = np.mean(sorted_align[idx_resample,d])
		sorted_dec_dist[i] = np.mean(sorted_disentangle[idx_resample,d])
		sorted_null_dist[i] = np.mean(sorted_baseline[idx_resample,d])
	# Get the 5th and 95th CI distributions percentiles
	ci_sorted_corr[0,d] = np.percentile(sorted_corr_dist, 2.5)
	ci_sorted_corr[1,d] = np.percentile(sorted_corr_dist, 97.5)
	ci_sorted_dec[0,d] = np.percentile(sorted_dec_dist, 2.5)
	ci_sorted_dec[1,d] = np.percentile(sorted_dec_dist, 97.5)
	ci_sorted_null[0,d] = np.percentile(sorted_null_dist, 2.5)
	ci_sorted_null[1,d] = np.percentile(sorted_null_dist, 97.5)

# Test for a decreasing trend (conditions needs to be arranged in order of
# increasing predicted mean, for the test to work).
sorted_align_decrease = page_trend_test(np.flip(sorted_align, 1))
sorted_disentangle_decrease = page_trend_test(np.flip(sorted_disentangle, 1))
sorted_baseline_decrease = page_trend_test(np.flip(sorted_baseline, 1))

# Plot parameters
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
x_dist_within = float(0.175)

# Plot
fig = plt.figure(figsize=(9, 7))

for d, dist in enumerate(area_distance):

	# Alignment neural control condition
	# Plot the single subjects RSA scores
	x = np.repeat(dist-x_dist_within*1, len(args.all_subjects))
	y = sorted_align[:,d]
	plt.scatter(x, y, s, c=colors[0], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(sorted_align[:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[0], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[0], label='_nolegend_')
	# Plot the CIs
	ci_low = y - ci_sorted_corr[0,d]
	ci_up = ci_sorted_corr[1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[0],
		elinewidth=5, capsize=0)

	# Baseline images
	# Plot the single subjects RSA scores
	x = np.repeat(dist, len(args.all_subjects))
	y = sorted_baseline[:,d]
	plt.scatter(x, y, s, c=colors[1], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(sorted_baseline[:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[1], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[1], label='_nolegend_')
	# Plot the CIs
	ci_low = y - ci_sorted_null[0,d]
	ci_up = ci_sorted_null[1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[1],
		elinewidth=5, capsize=0)

	# Disentanglement neural control condition
	# Plot the single subjects RSA scores
	x = np.repeat(dist+x_dist_within*1, len(args.all_subjects))
	y = sorted_disentangle[:,d]
	plt.scatter(x, y, s, c=colors[2], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA scores
	x = x[0]
	y = np.mean(sorted_disentangle[:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[2], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[2], label='_nolegend_')
	# Plot the CIs
	ci_low = y - ci_sorted_dec[0,d]
	ci_up = ci_sorted_dec[1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[2],
		elinewidth=5, capsize=0)

# Plot trend connectors
# Baseline
x = area_distance
y = np.mean(sorted_baseline, 0)
plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[1], alpha=0.5)
plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[1], alpha=0.5)
# Disentanglement
x = area_distance + (x_dist_within * 1)
y = np.mean(sorted_disentangle, 0)
plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=2, color=colors[2], alpha=0.5)
plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=2, color=colors[2], alpha=0.5)

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

#fig.savefig('multivariate_rnc_cortical_distance_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_cortical_distance_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_cortical_distance_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the multivariate RNC RSA scores distance from the target scores (i.e.,
# r=1 for the alignment neural control condition, and r=0 for the
# disentanglement neural control condition), as a function of cortical distance
# =============================================================================
# Load the stats
scores_diff_align = []
scores_diff_disentangle = []

for r in args.roi_pairs:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats',
		'cv-1', 'imageset-'+args.imageset, r, 'stats.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()

	scores_align_all_sub = []
	scores_disentangle_all_sub = []
	for s, sub in enumerate(args.all_subjects):
		scores_align_all_sub.append(
			data_dict['best_generation_scores_test']['align'][s,-1])
		scores_disentangle_all_sub.append(
			data_dict['best_generation_scores_test']['disentangle'][s,-1])
	scores_diff_align.append(np.asarray(scores_align_all_sub))
	scores_diff_disentangle.append(np.asarray(scores_disentangle_all_sub))

# Compute the absolute difference from the target scores
scores_diff_align = abs(1 - np.transpose(np.asarray(scores_diff_align)))
scores_diff_disentangle = abs(0 - np.transpose(np.asarray(
	scores_diff_disentangle)))

# There are three cortical distances:
# Cortical distance 1: [V1 vs. V2; V2 vs. V3; V3 vs. V4]
# Cortical distance 2: [V1 vs. V3; V2 vs. V4]
# Cortical distance 3: [V1 vs. V4]
cortical_distances = [(0, 3, 5), (1, 4), (2)]

# Sorted scores arrays of shape:
# (Subjects x ROI cortical distances)
sorted_diff_align = np.zeros((len(args.all_subjects), len(cortical_distances)))
sorted_diff_disentangle = np.zeros((len(args.all_subjects),
	len(cortical_distances)))

# Group the difference scores based on cortical distance
for d, dist in enumerate(cortical_distances):
	if type(dist) == tuple:
		sorted_diff_align[:,d] = np.mean(scores_diff_align[:,dist], 1)
		sorted_diff_disentangle[:,d] = np.mean(scores_diff_disentangle[:,dist],
			1)
	else:
		sorted_diff_align[:,d] = scores_diff_align[:,dist]
		sorted_diff_disentangle[:,d] = scores_diff_disentangle[:,dist]

# Compute the 95% confidence intervals
# CI arrays of shape:
# (CI percentiles x ROI pairwise comparisons)
ci_sorted_corr_diff = np.zeros((2, len(args.roi_pairs)))
ci_dorted_dec_diff = np.zeros((2, len(args.roi_pairs)))
for d in tqdm(range(len(cortical_distances)), leave=False):
	# Empty CI distribution array
	corr_dist = np.zeros((args.n_iter))
	dec_dist = np.zeros((args.n_iter))
	# Compute the CI distribution
	for i in range(args.n_iter):
		idx_resample = resample(np.arange(len(args.all_subjects)))
		corr_dist[i] = np.mean(sorted_diff_align[idx_resample,d])
		dec_dist[i] = np.mean(sorted_diff_disentangle[idx_resample,d])
	# Get the 5th and 95th CI distributions percentiles
	ci_sorted_corr_diff[0,d] = np.percentile(corr_dist, 2.5)
	ci_sorted_corr_diff[1,d] = np.percentile(corr_dist, 97.5)
	ci_dorted_dec_diff[0,d] = np.percentile(dec_dist, 2.5)
	ci_dorted_dec_diff[1,d] = np.percentile(dec_dist, 97.5)

# Test for a significant difference
pval = np.zeros((len(cortical_distances)))
for d in range(len(cortical_distances)):
	pval[d] = ttest_rel(sorted_diff_disentangle[:,d], sorted_diff_align[:,d],
		alternative='greater')[1]

# Correct for multiple comparisons
sig, pval_corrected, _, _ = multipletests(pval, 0.05, 'fdr_bh')

# Plot parameters
alpha = 0.2
area_distance = np.asarray((1, 2, 3))
s = 600
s_mean = 800
x_dist_within = float(0.175)
sig_offset = 0.03
sig_bar_length = 0.015
linewidth_sig_bar = 1.5
sig_star_offset_top = 0.02

# Plot
fig = plt.figure(figsize=(9, 7))

for d, dist in enumerate(area_distance):

	# Alignment neural control condition
	# Plot the single subjects RSA difference scores
	x = np.repeat(dist-x_dist_within*0.5, len(args.all_subjects))
	y = sorted_diff_align[:,d]
	plt.scatter(x, y, s, c=colors[0], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA difference scores
	x = x[0]
	y = np.mean(sorted_diff_align[:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[0], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[0], label='_nolegend_')
	# Plot the CIs
	ci_low = y - ci_sorted_corr_diff[0,d]
	ci_up = ci_sorted_corr_diff[1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[0],
		elinewidth=5, capsize=0)

	# Disentanglement neural control condition
	# Plot the single subjects RSA difference scores
	x = np.repeat(dist+x_dist_within*0.5, len(args.all_subjects))
	y = sorted_diff_disentangle[:,d]
	plt.scatter(x, y, s, c=colors[2], alpha=alpha, label='_nolegend_')
	# Plot the subject-average RSA difference scores
	x = x[0]
	y = np.mean(sorted_diff_disentangle[:,d])
	if d == 0:
		plt.scatter(x, y, s_mean, c=colors[2], label='')
	else:
		plt.scatter(x, y, s_mean, c=colors[2], label='_nolegend_')
	# Plot the CIs
	ci_low = y - ci_dorted_dec_diff[0,d]
	ci_up = ci_dorted_dec_diff[1,d] - y
	conf_int = np.reshape(np.append(ci_low, ci_up), (-1,1))
	plt.errorbar(x, y, yerr=conf_int, fmt="none", ecolor=colors[2],
		elinewidth=5, capsize=0)

	# Significant difference between alignment and disentanglement neural
	# control conditions
	if sig[d] == True:
		x = [dist-x_dist_within*0.5, dist+x_dist_within*0.5]
		y_max = max(sorted_diff_disentangle[:,d]) + sig_offset
		plt.plot([x[0], x[0]], [y_max, y_max+sig_bar_length], 'k-',
			[x[0], x[1]], [y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
			[x[1], x[1]], [y_max+sig_bar_length, y_max], 'k-',
			linewidth=linewidth_sig_bar)
		y = y_max + sig_bar_length + sig_star_offset_top
		plt.text(dist, y, s='*', fontsize=30, color='k',
			fontweight='bold', ha='center', va='center')

# x-axis parameters
xticks = [1, 2, 3]
xlabels = [1, 2, 3]
plt.xticks(ticks=xticks, labels=xlabels)
xlabel = 'Stepwise area distance'
plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=0.5, right=3.5)

# y-axis parameters
ylabel = 'Absolute $Δ$ Pearson\'s $r$'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(top=.5, bottom=-0.05)

#fig.savefig('multivariate_rnc_target_scores_cortical_distance_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_target_scores_cortical_distance_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_target_scores_cortical_distance_things.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the multivariate RNC optimization curves
# =============================================================================
# Load the stats
best_generation_scores_train = {}
best_generation_scores_test = {}
baseline_images_score_test = {}

for r in args.roi_pairs:

	data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats',
		'cv-1', 'imageset-'+args.imageset, r, 'stats.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()

	for c in args.control_conditions:
		for s, sub in enumerate(args.all_subjects):
			best_generation_scores_train['s'+str(sub)+'_'+r+'_'+c] = \
				data_dict['best_generation_scores_train'][c][s]
			best_generation_scores_test['s'+str(sub)+'_'+r+'_'+c] = \
				data_dict['best_generation_scores_test'][c][s]
			baseline_images_score_test['s'+str(sub)+'_'+r] = \
				data_dict['baseline_images_score_test'][s]

# Plot
fontsize = 20
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
roi_pairs = ['V1-V2', 'V1-V3', 'V1-V4', 'V2-V3', 'V2-V4', 'V3-V4']
fig, axs = plt.subplots(6, 8, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
idx = 0

for r, roi in enumerate(args.roi_pairs):

	for s in args.all_subjects:

		# Title
		if idx in [0, 1, 2, 3, 4, 5, 6, 7]:
			title = 'Subject ' + str(s)
			axs[idx].set_title(title, fontsize=fontsize)

		# x-axis
		if idx in [40, 41, 42, 43, 44, 45, 46, 47]:
			axs[idx].set_xlabel('Generations', fontsize=fontsize)
			xticks = [1000]
			xlabels = ['1,000']
			axs[r].set_xticks(ticks=xticks, labels=xlabels)

		# x-axis
		if idx in [0, 8, 16, 24, 32, 40]:
			y_label = roi_pairs[r] + '\nPearson\'s $r$'
			axs[idx].set_ylabel(y_label, fontsize=fontsize)

		x = np.arange(len(best_generation_scores_train\
			['s'+str(s)+'_'+roi+'_align']))

		# Plot the training curves (alignment)
		axs[idx].plot(x,
			best_generation_scores_train['s'+str(s)+'_'+roi+'_align'],
			linewidth=1, color=colors[0])

		# Plot the test curves (alignment)
		axs[idx].plot(x,
			best_generation_scores_test['s'+str(s)+'_'+roi+'_align'], '--',
			linewidth=1, color=colors[0])

		# Plot the baseline images scores
		control_scores = baseline_images_score_test['s'+str(s)+'_'+roi]
		axs[idx].plot([x[0], x[-1]], [control_scores, control_scores], '--',
			linewidth=1, color=colors[1])

		# Plot the train curves (disentanglement)
		axs[idx].plot(x,
			best_generation_scores_train['s'+str(s)+'_'+roi+'_disentangle'],
			linewidth=1, color=colors[2])

		# Plot the test curves (disentanglement)
		axs[idx].plot(x,
			best_generation_scores_test['s'+str(s)+'_'+roi+'_disentangle'],
			'--', linewidth=1, color=colors[2])

		# Legend
		if idx in [0]:
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
			axs[idx].legend(custom_lines, legend, loc=2, ncol=5, fontsize=fontsize,
				bbox_to_anchor=(1.525, -6.5), frameon=False, markerscale=2)

		# Limits
		axs[idx].set_xlim(min(x), max(x))
		axs[idx].set_ylim(bottom=-.05, top=1)

		idx += 1

#fig.savefig('multivariate_rnc_optimization_curves_nsd.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_optimization_curves_imagenet.png', dpi=100, bbox_inches='tight')
#fig.savefig('multivariate_rnc_optimization_curves_things.png', dpi=100, bbox_inches='tight')

