"""Plot the Neural Encoding Dataset (NED) fMRI encoding models explained
variance for areas V1, V2, V3 and V4 (subject-average), as well as the noise
analysis.

This code is available at:
https://github.com/gifale95/RNC/blob/main/01_in_silico_fmri_encoding_accuracy/02_plot.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	in silico fMRI responses.
rois : list of str
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
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding accuracy and SNR analysis results
# =============================================================================
results_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'encoding_accuracy.npy')

results = np.load(results_dir, allow_pickle=True).item()


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
colors = [(249/255, 177/255, 66/255), (240/255, 99/255, 51/255),
	(218/255, 83/255, 128/255), (124/255, 108/255, 237/255)]


# =============================================================================
# Plot the subject-average encoding accuracy results
# =============================================================================
# Format the results for plotting
acc = np.zeros((len(args.all_subjects), len(args.rois)))
ci = np.zeros((2, len(args.rois)))
sig = np.zeros((len(args.rois)))
for r, roi in enumerate(args.rois):
	# Prediction accuracy
	for s, sub in enumerate(args.all_subjects):
		acc[s,r] = np.mean(results['explained_variance']\
			['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci[0,r] = results['ci_lower'][roi] * 100
	ci[1,r] = results['ci_upper'][roi] * 100
	# Aggregate the significance
	sig[r] = results['significance'][roi]
# Difference bewteen prediction accuracy and confidence intervals
ci[0] = np.mean(acc, 0) - ci[0]
ci[1] = ci[1] - np.mean(acc, 0)

# Plot parameters
alpha = 0.2
sig_offset = 10
fontsize_sig = 20
s = 1000
color = 'k'

# Plot
fig = plt.figure(figsize=(4,7))

for r, roi in enumerate(args.rois):

	# Prediction accuracy scores
	x = np.repeat(r+1, len(args.all_subjects))
	plt.scatter(x, acc[:,r], s=s, color=color, alpha=alpha)
	plt.scatter(x[0], np.mean(acc[:,r]), s=s, color=color)

	# Confidence intervals
	plt.errorbar(x[0], np.mean(acc[:,r]), yerr=np.reshape(ci[:,r], (-1,1)),
		fmt="none", ecolor=color, elinewidth=5, capsize=0)

	# Significance
	if sig[0] == 1:
		y = max(acc[:,r]) + sig_offset
		plt.text(x[0], y, s='*', fontsize=fontsize_sig, color=color,
			fontweight='bold', ha='center', va='center')

# x-axis parameters
xticks = [1, 2, 3, 4]
xlabels = ['V1', 'V2', 'V3', 'V4']
plt.xticks(ticks=xticks, labels=xlabels, rotation=0)
xlabel = 'Areas'
plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=0.25, right=4.75)

# y-axis parameters
yticks = [0, 20, 40, 60, 80, 100]
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Explained variance (%)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=100)

#fig.savefig('encoding_accuracy_subject_average.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the single subjects encoding accuracy results
# =============================================================================
# Format the results for plotting
acc = np.zeros((len(args.all_subjects), len(args.rois)))
ci = np.zeros((2, len(args.rois)))
sig = np.zeros((len(args.rois)))
for r, roi in enumerate(args.rois):
	# Prediction accuracy
	for s, sub in enumerate(args.all_subjects):
		acc[s,r] = np.mean(results['explained_variance']\
			['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci[0,r] = results['ci_lower'][roi] * 100
	ci[1,r] = results['ci_upper'][roi] * 100
	# Aggregate the significance
	sig[r] = results['significance'][roi]
# Difference bewteen prediction accuracy and confidence intervals
ci[0] = np.mean(acc, 0) - ci[0]
ci[1] = ci[1] - np.mean(acc, 0)

# Plot parameters
rois = ['V1', 'V2', 'V3', 'V4']
x = np.arange(len(acc))
width = 0.4

# Plot
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
	figsize=(5, 3))
axs = np.reshape(axs, (-1))

for r, roi in enumerate(rois):

	# Plot the encoding accuracies
	axs[r].bar(x, acc[:,r], width=width, color='k')

	# y-axis parameters
	if r in [0, 2]:
		axs[r].set_ylabel('Explained variance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)
		plt.yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=0, top=100)

	# x-axis parameters
	if r in [2, 3]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)
		xticks = x
		xlabels = ['1', '2', '3', '4', '5', '6', '7', '8']
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=fontsize)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

#fig.savefig('encoding_accuracy_single_subjects.png', dpi=100, bbox_inches='tight')


# =============================================================================
# Plot the noise analysis results
# =============================================================================
# Format the results for plotting
acc_gt1tr_synt = np.zeros((len(args.rois), len(args.all_subjects)))
acc_gt1tr_gt1tr = np.zeros((len(args.rois), len(args.all_subjects)))
acc_gt1tr_gt2tr = np.zeros((len(args.rois), len(args.all_subjects)))
ci_gt1tr_synt = np.zeros((len(args.rois), 2))
ci_gt1tr_gt1tr = np.zeros((len(args.rois), 2))
ci_gt1tr_gt2tr = np.zeros((len(args.rois), 2))
sig_gt1tr_gt1tr_vs_gt1tr_gt2tr = np.zeros((len(args.rois)))
sig_gt1tr_gt2tr_vs_gt1tr_synt = np.zeros((len(args.rois)))
for r, roi in enumerate(args.rois):
	# Prediction accuracy
	for s, sub in enumerate(args.all_subjects):
		acc_gt1tr_synt[r,s] = np.mean(results['explained_variance_gt1tr_synt']\
			['s'+str(sub)+'_'+roi]) * 100
		acc_gt1tr_gt1tr[r,s] = np.mean(results['explained_variance_gt1tr_gt1tr']\
			['s'+str(sub)+'_'+roi]) * 100
		acc_gt1tr_gt2tr[r,s] = np.mean(results['explained_variance_gt1tr_gt2tr']\
			['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci_gt1tr_synt[r,0] = results['ci_lower_gt1tr_synt'][roi] * 100
	ci_gt1tr_synt[r,1] = results['ci_upper_gt1tr_synt'][roi] * 100
	ci_gt1tr_gt1tr[r,0] = results['ci_lower_gt1tr_gt1tr'][roi] * 100
	ci_gt1tr_gt1tr[r,1] = results['ci_upper_gt1tr_gt1tr'][roi] * 100
	ci_gt1tr_gt2tr[r,0] = results['ci_lower_gt1tr_gt2tr'][roi] * 100
	ci_gt1tr_gt2tr[r,1] = results['ci_upper_gt1tr_gt2tr'][roi] * 100
	# Aggregate the significance
	sig_gt1tr_gt1tr_vs_gt1tr_gt2tr[r] = results\
		['significance_gt1tr_gt1tr_vs_gt1tr_gt2tr'][roi]
	sig_gt1tr_gt2tr_vs_gt1tr_synt[r] = results\
		['significance_gt1tr_gt2tr_vs_gt1tr_synt'][roi]
# Difference bewteen prediction accuracy and confidence intervals
ci_gt1tr_synt[:,0] = np.mean(acc_gt1tr_synt, 1) - ci_gt1tr_synt[:,0]
ci_gt1tr_synt[:,1] = ci_gt1tr_synt[:,1] - np.mean(acc_gt1tr_synt, 1)
ci_gt1tr_gt1tr[:,0] = np.mean(acc_gt1tr_gt1tr, 1) - ci_gt1tr_gt1tr[:,0]
ci_gt1tr_gt1tr[:,1] = ci_gt1tr_gt1tr[:,1] - np.mean(acc_gt1tr_gt1tr, 1)
ci_gt1tr_gt2tr[:,0] = np.mean(acc_gt1tr_gt2tr, 1) - ci_gt1tr_gt2tr[:,0]
ci_gt1tr_gt2tr[:,1] = ci_gt1tr_gt2tr[:,1] - np.mean(acc_gt1tr_gt2tr, 1)

# Plot parameters
x_coord = [1, 2, 3]
dist = 0.15
x_dist = np.asarray((-1.5, -0.5, 0.5, 1.5)) * dist
x_dist_sig = np.asarray((-.75, -0.25, 0.25, .75)) * dist
alpha = 0.2
fontsize_sig = 20
marker = 'o'
s = 1000
rois = ['V1', 'V2', 'V3', 'V4']
sig_offset = 10
sig_bar_length = 3
linewidth_sig_bar = 1.5
sig_star_offset_top = 2

# Plot
fig = plt.figure(figsize=(10,9))

for i, xc in enumerate(x_coord):
	for r, roi in enumerate(args.rois):

		# Encoding accuracy scores
		x = np.repeat(xc+x_dist[r], len(args.all_subjects))
		if i == 0:
			plt.scatter(x, acc_gt1tr_gt1tr[r], s=s, color=colors[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_gt1tr[r]), s=s, color=colors[r],
				label=rois[r])
		elif i == 1:
			plt.scatter(x, acc_gt1tr_gt2tr[r], s=s, color=colors[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_gt2tr[r]), s=s, color=colors[r],
				label='_nolegend_')
		elif i == 2:
			plt.scatter(x, acc_gt1tr_synt[r], s=s, color=colors[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_synt[r]), s=s, color=colors[r],
				label='_nolegend_')

		# Confidence intervals
		if i == 0:
			plt.errorbar(x[0], np.mean(acc_gt1tr_gt1tr[r]),
				yerr=np.reshape(ci_gt1tr_gt1tr[r], (-1,1)), fmt="none",
				ecolor=colors[r], elinewidth=5, capsize=0)
		elif i == 1:
			plt.errorbar(x[0], np.mean(acc_gt1tr_gt2tr[r]),
				yerr=np.reshape(ci_gt1tr_gt2tr[r], (-1,1)), fmt="none",
				ecolor=colors[r], elinewidth=5, capsize=0)
		elif i == 2:
			plt.errorbar(x[0], np.mean(acc_gt1tr_synt[r]),
				yerr=np.reshape(ci_gt1tr_synt[r], (-1,1)), fmt="none",
				ecolor=colors[r], elinewidth=5, capsize=0)

# Significance 1
if all(sig_gt1tr_gt1tr_vs_gt1tr_gt2tr == True):
	res = np.append(acc_gt1tr_gt1tr, acc_gt1tr_gt2tr)
	y_max = max(res) + sig_offset
	plt.plot([x_coord[0], x_coord[0]], [y_max, y_max+sig_bar_length],
		'k-', [x_coord[0], x_coord[1]],
		[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
		[x_coord[1], x_coord[1]], [y_max+sig_bar_length, y_max], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x_coord[0], x_coord[1])))
	y = y_max + sig_bar_length + sig_star_offset_top
	for r, roi in enumerate(args.rois):
		plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
			color=colors[r], fontweight='bold', ha='center', va='center')

# Significance 2
if all(sig_gt1tr_gt2tr_vs_gt1tr_synt == True):
	res = np.append(acc_gt1tr_gt2tr, acc_gt1tr_synt)
	y_max = max(res) + sig_offset
	plt.plot([x_coord[1], x_coord[1]], [y_max, y_max+sig_bar_length],
		'k-', [x_coord[1], x_coord[2]],
		[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
		[x_coord[2], x_coord[2]], [y_max+sig_bar_length, y_max], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x_coord[1], x_coord[2])))
	y = y_max + sig_bar_length + sig_star_offset_top
for r, roi in enumerate(args.rois):
	plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
		color=colors[r], fontweight='bold', ha='center', va='center')

# x-axis parameters
xticks = x_coord
xlabels = ['NSD single\ntrials', 'NSD trials\naverage', '$In-silico$\nresponses']
plt.xticks(ticks=xticks, labels=xlabels, rotation=0)
xlabel = 'Predictors'
#	plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=0.5, right=3.5)

# y-axis parameters
yticks = [0, 20, 40, 60, 80, 100]
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Explained variance (%)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=100)

# Legend
plt.legend(loc=2, ncol=2, fontsize=fontsize, frameon=False)

#fig.savefig('noise_analysis.png', dpi=100, bbox_inches='tight')

