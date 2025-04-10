"""Plot the encoding models' encoiding accuracy, as well as the results of the
noise analysis.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
nsd_subjects : list of int
	List of all NSD subjects.
VisualIllusionRecon_subjects : list of int
	List of all Visual Illusion Reconstuction dataset subjects.
nsd_rois : list of str
	List of used NSD ROIs.
VisualIllusionRecon_rois : list of str
	List of used Visual Illusion Reconstuction dataset ROIs.
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
parser.add_argument('--nsd_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--VisualIllusionRecon_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7])
parser.add_argument('--nsd_rois', type=list, default=['V1', 'V2', 'V3', 'hV4', 'EBA', 'FFA', 'PPA', 'RSC'])
parser.add_argument('--VisualIllusionRecon_rois', type=list, default=['V1', 'V2', 'V3', 'hV4'])
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()


# =============================================================================
# Load the encoding accuracy and SNR analysis results
# =============================================================================
# NSD encoding accuracy
results_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'nsd_encoding_models', 'encoding_accuracy.npy')
results_nsd = np.load(results_dir, allow_pickle=True).item()

# NSD OOD encoding accuracy
results_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'nsd_encoding_models', 'ood_encoding_accuracy.npy')
results_nsd_ood = np.load(results_dir, allow_pickle=True).item()

# VisualIllusionRecon encoding accuracy
results_dir = os.path.join(args.project_dir, 'encoding_accuracy',
	'VisualIllusionRecon_encoding_models', 'encoding_accuracy.npy')
results_VisualIllusionRecon = np.load(results_dir, allow_pickle=True).item()


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 30
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
colors_1 = [(221/255, 204/255, 119/255), (204/255, 102/255, 119/255)]
colors_2 = [(249/255, 177/255, 66/255), (240/255, 99/255, 51/255),
	(218/255, 83/255, 128/255), (124/255, 108/255, 237/255)]


# =============================================================================
# Plot the subject-average encoding accuracy results (NSD)
# =============================================================================
# Only plot early visual cortex ROIs
evc_rois = ['V1', 'V2', 'V3', 'hV4']

# Format the results for plotting
acc = np.zeros((len(args.nsd_subjects), len(evc_rois)))
ci = np.zeros((2, len(evc_rois)))
sig = np.zeros((len(evc_rois)))
for r, roi in enumerate(evc_rois):
	# Prediction accuracy
	for s, sub in enumerate(args.nsd_subjects):
		acc[s,r] = np.mean(results_nsd['explained_variance']\
			['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci[0,r] = results_nsd['ci_lower'][roi] * 100
	ci[1,r] = results_nsd['ci_upper'][roi] * 100
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

for r, roi in enumerate(evc_rois):

	# Prediction accuracy scores
	x = np.repeat(r+1, len(args.nsd_subjects))
	plt.scatter(x, acc[:,r], s=s, color=color, alpha=alpha)
	plt.scatter(x[0], np.mean(acc[:,r]), s=s, color=color)

	# Confidence intervals
	plt.errorbar(x[0], np.mean(acc[:,r]), yerr=np.reshape(ci[:,r], (-1,1)),
		fmt="none", ecolor=color, elinewidth=5, capsize=0)

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
ylabel = 'Noise-ceiling-normalized\nexplained variance (%)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=100)

# Save the figure
fig_name = 'encoding_accuracy_nsd_subject_average.svg'
fig.savefig(fig_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the single subjects encoding accuracy results (NSD)
# =============================================================================
# Format the results for plotting
acc = np.zeros((len(args.nsd_subjects), len(args.nsd_rois)))
ci = np.zeros((2, len(args.nsd_rois)))
acc_ood = np.zeros((len(args.nsd_subjects), len(args.nsd_rois)))
ci_ood = np.zeros((2, len(args.nsd_rois)))
for r, roi in enumerate(args.nsd_rois):
	# Prediction accuracy
	for s, sub in enumerate(args.nsd_subjects):
		acc[s,r] = np.mean(results_nsd['explained_variance']\
			['s'+str(sub)+'_'+roi]) * 100
		acc_ood[s,r] = np.mean(results_nsd_ood['explained_variance']\
			['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci[0,r] = results_nsd['ci_lower'][roi] * 100
	ci[1,r] = results_nsd['ci_upper'][roi] * 100
	ci_ood[0,r] = results_nsd_ood['ci_lower'][roi] * 100
	ci_ood[1,r] = results_nsd_ood['ci_upper'][roi] * 100
# Difference bewteen prediction accuracy and confidence intervals
ci[0] = np.mean(acc, 0) - ci[0]
ci[1] = ci[1] - np.mean(acc, 0)
ci_ood[0] = np.mean(acc_ood, 0) - ci_ood[0]
ci_ood[1] = ci_ood[1] - np.mean(acc_ood, 0)

# Plot parameters
rois = ['V1', 'V2', 'V3', 'V4', 'EBA', 'FFA', 'PPA', 'RSC']
width = 0.4

# Plot
fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,
	figsize=(5, 3))
axs = np.reshape(axs, (-1))

for r, roi in enumerate(rois):

	# Plot the single-subject encoding accuracies
	x = np.arange(len(acc))
	axs[r].bar(x-width/2, acc[:,r], width=width, color=colors_1[0],
		label='_nolegend_')
	axs[r].bar(x+width/2, acc_ood[:,r], width=width, color=colors_1[1],
		label='_nolegend_')

	# Plot the subject-average encoding accuracies
	x = len(acc)
	label = 'In-distribution generalization'
	axs[r].bar(x-width/2, np.mean(acc[:,r]), width=width, color=colors_1[0],
		label=label)
	label = 'Out-of-distribution generalization'
	axs[r].bar(x+width/2, np.mean(acc_ood[:,r]), width=width,
		color=colors_1[1], label=label)
	# Plot the subject-average CIs
	axs[r].errorbar(x-width/2, np.mean(acc[:,r]),
		yerr=np.reshape(ci[:,r], (-1,1)), fmt="none", ecolor='k',
		elinewidth=3, capsize=0, label='_nolegend_')
	axs[r].errorbar(x+width/2, np.mean(acc_ood[:,r]),
		yerr=np.reshape(ci_ood[:,r], (-1,1)), fmt="none", ecolor='k',
		elinewidth=3, capsize=0, label='_nolegend_')

	# y-axis parameters
	if r in [0, 4]:
		axs[r].set_ylabel('Noise-ceiling-normalized\nexplained variance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)
		axs[r].set_yticks(ticks=yticks, labels=ylabels)
	axs[r].set_ylim(bottom=0, top=100)

	# x-axis parameters
	if r in [4, 5, 6, 7]:
		axs[r].set_xlabel('Subjects', fontsize=fontsize)
		xticks = np.arange(len(acc) + 1)
		xlabels = ['1', '2', '3', '4', '5', '6', '7', '8', 'Mean']
		axs[r].set_xticks(ticks=xticks, labels=xlabels, fontsize=fontsize,
			rotation=45)

	# Title
	axs[r].set_title(roi, fontsize=fontsize)

	# Legend
	if r == 7:
		axs[r].legend(loc=1, ncol=1, fontsize=fontsize)

# Save the figure
fig_name = 'encoding_accuracy_nsd_single_subjects.svg'
fig.savefig(fig_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the single subjects encoding accuracy results (results_VisualIllusionRecon)
# =============================================================================
# Format the results for plotting
acc = np.zeros((len(args.VisualIllusionRecon_subjects),
	len(args.VisualIllusionRecon_rois)))
ci = np.zeros((2, len(args.VisualIllusionRecon_rois)))
acc_ood = np.zeros((len(args.VisualIllusionRecon_subjects),
	len(args.VisualIllusionRecon_rois)))
ci_ood = np.zeros((2, len(args.VisualIllusionRecon_rois)))
for r, roi in enumerate(args.VisualIllusionRecon_rois):
	# Prediction accuracy
	for s, sub in enumerate(args.VisualIllusionRecon_subjects):
		acc[s,r] = np.mean(results_VisualIllusionRecon['explained_variance']\
			['id']['s'+str(sub)+'_'+roi]) * 100
		acc_ood[s,r] = np.mean(results_VisualIllusionRecon\
			['explained_variance']['ood']['s'+str(sub)+'_'+roi]) * 100
	# Aggregate the confidence intervals
	ci[0,r] = results_VisualIllusionRecon['ci_lower']['id'][roi] * 100
	ci[1,r] = results_VisualIllusionRecon['ci_upper']['id'][roi] * 100
	ci_ood[0,r] = results_VisualIllusionRecon['ci_lower']['ood'][roi] * 100
	ci_ood[1,r] = results_VisualIllusionRecon['ci_upper']['ood'][roi] * 100
# Difference bewteen prediction accuracy and confidence intervals
ci[0] = np.mean(acc, 0) - ci[0]
ci[1] = ci[1] - np.mean(acc, 0)
ci_ood[0] = np.mean(acc_ood, 0) - ci_ood[0]
ci_ood[1] = ci_ood[1] - np.mean(acc_ood, 0)

# Plot parameters
rois = ['V1', 'V2', 'V3', 'V4']
width = 0.4

# Plot
fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,
	figsize=(5, 3))
axs = np.reshape(axs, (-1))

for r, roi in enumerate(rois):
	r_plot = r + len(rois)

	# Plot the single-subject encoding accuracies
	x = np.arange(len(acc))
	axs[r_plot].bar(x-width/2, acc[:,r], width=width, color=colors_1[0],
		label='_nolegend_')
	axs[r_plot].bar(x+width/2, acc_ood[:,r], width=width, color=colors_1[1],
		label='_nolegend_')

	# Plot the subject-average encoding accuracies
	x = len(acc)
	label = 'In-distribution generalization'
	axs[r_plot].bar(x-width/2, np.mean(acc[:,r]), width=width,
		color=colors_1[0], label=label)
	label = 'Out-of-distribution generalization'
	axs[r_plot].bar(x+width/2, np.mean(acc_ood[:,r]), width=width,
		color=colors_1[1], label=label)
	# Plot the subject-average CIs
	axs[r_plot].errorbar(x-width/2, np.mean(acc[:,r]),
		yerr=np.reshape(ci[:,r], (-1,1)), fmt="none", ecolor='k',
		elinewidth=3, capsize=0, label='_nolegend_')
	axs[r_plot].errorbar(x+width/2, np.mean(acc_ood[:,r]),
		yerr=np.reshape(ci_ood[:,r], (-1,1)), fmt="none", ecolor='k',
		elinewidth=3, capsize=0, label='_nolegend_')

	# y-axis parameters
	if r_plot in [0, 4]:
		axs[r_plot].set_ylabel(
			'Noise-ceiling-normalized\nexplained variance (%)',
			fontsize=fontsize)
		yticks = np.arange(0, 101, 20)
		ylabels = np.arange(0, 101, 20)
		axs[r_plot].set_yticks(ticks=yticks, labels=ylabels)
	axs[r_plot].set_ylim(bottom=0, top=100)

	# x-axis parameters
	if r_plot in [4, 5, 6, 7]:
		axs[r_plot].set_xlabel('Subjects', fontsize=fontsize)
		xticks = np.arange(len(acc) + 1)
		xlabels = ['1', '2', '3', '4', '5', '6', '7', 'Mean']
		axs[r_plot].set_xticks(ticks=xticks, labels=xlabels, fontsize=fontsize,
			rotation=45)

	# Title
	axs[r_plot].set_title(roi, fontsize=fontsize)

	# Legend
	if r == 3:
		axs[r_plot].legend(loc=1, ncol=1, fontsize=fontsize)

# Save the figure
fig_name = 'encoding_accuracy_visualIllusionRecon_single_subjects.svg'
fig.savefig(fig_name, bbox_inches='tight', transparent=True, format='svg')


# =============================================================================
# Plot the noise analysis results
# =============================================================================
# Only plot early visual cortex ROIs
evc_rois = ['V1', 'V2', 'V3', 'hV4']

# Format the results for plotting
acc_gt1tr_synt = np.zeros((len(evc_rois), len(args.nsd_subjects)))
acc_gt1tr_gt1tr = np.zeros((len(evc_rois), len(args.nsd_subjects)))
acc_gt1tr_gt2tr = np.zeros((len(evc_rois), len(args.nsd_subjects)))
ci_gt1tr_synt = np.zeros((len(evc_rois), 2))
ci_gt1tr_gt1tr = np.zeros((len(evc_rois), 2))
ci_gt1tr_gt2tr = np.zeros((len(evc_rois), 2))
sig_gt1tr_gt1tr_vs_gt1tr_gt2tr = np.zeros((len(evc_rois)))
sig_gt1tr_gt2tr_vs_gt1tr_synt = np.zeros((len(evc_rois)))
for r, roi in enumerate(evc_rois):
	# Prediction accuracy
	for s, sub in enumerate(args.nsd_subjects):
		acc_gt1tr_synt[r,s] = results_nsd\
			['explained_variance_gt1tr_synt']['s'+str(sub)+'_'+roi] * 100
		acc_gt1tr_gt1tr[r,s] = results_nsd\
			['explained_variance_gt1tr_gt1tr']['s'+str(sub)+'_'+roi] * 100
		acc_gt1tr_gt2tr[r,s] = results_nsd\
			['explained_variance_gt1tr_gt2tr']['s'+str(sub)+'_'+roi] * 100
	# Aggregate the confidence intervals
	ci_gt1tr_synt[r,0] = results_nsd['ci_lower_gt1tr_synt'][roi] * 100
	ci_gt1tr_synt[r,1] = results_nsd['ci_upper_gt1tr_synt'][roi] * 100
	ci_gt1tr_gt1tr[r,0] = results_nsd['ci_lower_gt1tr_gt1tr'][roi] * 100
	ci_gt1tr_gt1tr[r,1] = results_nsd['ci_upper_gt1tr_gt1tr'][roi] * 100
	ci_gt1tr_gt2tr[r,0] = results_nsd['ci_lower_gt1tr_gt2tr'][roi] * 100
	ci_gt1tr_gt2tr[r,1] = results_nsd['ci_upper_gt1tr_gt2tr'][roi] * 100
	# Aggregate the significance
	sig_gt1tr_gt1tr_vs_gt1tr_gt2tr[r] = results_nsd\
		['gt1tr_gt2tr_minus_gt1tr_gt1tr_between_subject_pval'][roi]
	sig_gt1tr_gt2tr_vs_gt1tr_synt[r] = results_nsd\
		['gt1tr_synt_minus_gt1tr_gt2tr_between_subject_pval'][roi]
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
sig_offset = 7
sig_bar_length = 3
linewidth_sig_bar = 1
sig_star_offset_top = 2

# Plot
fig = plt.figure(figsize=(10,9))

for i, xc in enumerate(x_coord):
	for r, roi in enumerate(evc_rois):

		# Encoding accuracy scores
		x = np.repeat(xc+x_dist[r], len(args.nsd_subjects))
		if i == 0:
			plt.scatter(x, acc_gt1tr_gt1tr[r], s=s, color=colors_2[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_gt1tr[r]), s=s,
				color=colors_2[r], label=rois[r])
		elif i == 1:
			plt.scatter(x, acc_gt1tr_gt2tr[r], s=s, color=colors_2[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_gt2tr[r]), s=s,
				color=colors_2[r], label='_nolegend_')
		elif i == 2:
			plt.scatter(x, acc_gt1tr_synt[r], s=s, color=colors_2[r],
				alpha=alpha, label='_nolegend_')
			plt.scatter(x[0], np.mean(acc_gt1tr_synt[r]), s=s, color=colors_2[r],
				label='_nolegend_')

		# Confidence intervals
		if i == 0:
			plt.errorbar(x[0], np.mean(acc_gt1tr_gt1tr[r]),
				yerr=np.reshape(ci_gt1tr_gt1tr[r], (-1,1)), fmt="none",
				ecolor=colors_2[r], elinewidth=5, capsize=0)
		elif i == 1:
			plt.errorbar(x[0], np.mean(acc_gt1tr_gt2tr[r]),
				yerr=np.reshape(ci_gt1tr_gt2tr[r], (-1,1)), fmt="none",
				ecolor=colors_2[r], elinewidth=5, capsize=0)
		elif i == 2:
			plt.errorbar(x[0], np.mean(acc_gt1tr_synt[r]),
				yerr=np.reshape(ci_gt1tr_synt[r], (-1,1)), fmt="none",
				ecolor=colors_2[r], elinewidth=5, capsize=0)

# Significance 1
if all(sig_gt1tr_gt1tr_vs_gt1tr_gt2tr < 0.05):
	res = np.append(acc_gt1tr_gt1tr, acc_gt1tr_gt2tr)
	y_max = max(res) + sig_offset
	plt.plot([x_coord[0], x_coord[0]], [y_max, y_max+sig_bar_length],
		'k-', [x_coord[0], x_coord[1]],
		[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
		[x_coord[1], x_coord[1]], [y_max+sig_bar_length, y_max], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x_coord[0], x_coord[1])))
	y = y_max + sig_bar_length + sig_star_offset_top
	for r, roi in enumerate(evc_rois):
		plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
			color=colors_2[r], fontweight='bold', ha='center', va='center')

# Significance 2
if all(sig_gt1tr_gt2tr_vs_gt1tr_synt < 0.05):
	res = np.append(acc_gt1tr_gt2tr, acc_gt1tr_synt)
	y_max = max(res) + sig_offset
	plt.plot([x_coord[1], x_coord[1]], [y_max, y_max+sig_bar_length],
		'k-', [x_coord[1], x_coord[2]],
		[y_max+sig_bar_length, y_max+sig_bar_length], 'k-',
		[x_coord[2], x_coord[2]], [y_max+sig_bar_length, y_max], 'k-',
		linewidth=linewidth_sig_bar)
	x_mean = np.mean(np.asarray((x_coord[1], x_coord[2])))
	y = y_max + sig_bar_length + sig_star_offset_top
for r, roi in enumerate(evc_rois):
	plt.text(x_mean+x_dist_sig[r], y, s='*', fontsize=fontsize_sig,
		color=colors_2[r], fontweight='bold', ha='center', va='center')

# x-axis parameters
xticks = x_coord
xlabels = ['NSD single\ntrials', 'NSD trials\naverage', ' In silico\nresponses']
plt.xticks(ticks=xticks, labels=xlabels, rotation=0)
xlabel = 'Predictors'
#	plt.xlabel(xlabel, fontsize=fontsize)
plt.xlim(left=0.5, right=3.5)

# y-axis parameters
yticks = [0, 20, 40, 60, 80, 100]
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
ylabel = 'Noise-ceiling-normalized\nexplained variance (%)'
plt.ylabel(ylabel, fontsize=fontsize)
plt.ylim(bottom=0, top=100)

# Legend
plt.legend(loc=2, ncol=2, fontsize=fontsize, frameon=False)

# Save the figure
fig_name = 'noise_analysis_nsd_single_subjects.svg'
fig.savefig(fig_name, bbox_inches='tight', transparent=True, format='svg')
