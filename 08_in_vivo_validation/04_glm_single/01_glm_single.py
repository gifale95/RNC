"""Apply GLM-single to the preprocessed fMRI data.

https://github.com/cvnlab/GLMsingle/tree/main
https://glmsingle.readthedocs.io/en/latest/python.html
https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example1.html

This code is available at:
https://github.com/gifale95/RNC/blob/main/06_in_vivo_validation/04_glm_single/01_glm_single.py

Parameters
----------
subject : int
	Used subject, out of all 6 collected.
roi : str
	Whether to apply GLM-single to voxels from 'V1' or 'V4'.
exp_name : str
	Experiment name. Available choices are 'univariate_rnc' and
	'multivariate_rnc'.
stimdur : str
	Stimulus duration, in seconds. Here the images were presented for 2 second.
tr : int
	fMRI repetition time. Here the TR was 1 second.
trs_per_trial : int
	Number of TRs for each trial. Here there were 4 TRs per trial.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import time
from pprint import pprint
import pandas as pd

from glmsingle.glmsingle import GLM_single


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--exp_name', default='univariate_rnc', type=str)
parser.add_argument('--stimdur', default=2, type=int)
parser.add_argument('--tr', default=1, type=int)
parser.add_argument('--trs_per_trial', default=4, type=int)
parser.add_argument('--project_dir', default='../relational_neural_control', type=str)
args = parser.parse_args()

print('>>> GLM single <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Get ROI voxel mask
# =============================================================================
# The V1 and V4 manual delineations in subject-native volume space found in:
# "https://github.com/gifale95/RNC/tree/main/06_in_vivo_validation/03_prf_mapping/02_delineate_rois/v1_v4_rois_volume/derived/sub-0*/alldata/roi/*h.V*.func.nii.gz"
# need to be added to the directory:
# "../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/derived/sub0*/alldata/roi/"

hemispheres = ['lh', 'rh']

for h, hemi in enumerate(hemispheres):
	label_dir = os.path.join(args.project_dir, 'in_vivo_validation',
		'in_vivo_fmri_dataset', 'derived', 'sub'+format(args.subject, '02'),
		'alldata', 'roi', hemi+'.'+args.roi+'.func.nii.gz')
	mask = nib.load(label_dir).get_fdata()
	if h == 0:
		roi_mask = mask
	else:
		roi_mask += mask
	del mask

# Convert the mask to boolean
roi_mask = roi_mask != 0


# =============================================================================
# Load the preprocessed fMRI data
# =============================================================================
# data -> consists of several runs of 4D volume files (x,y,z,t) where
# (t)ime is the 4th dimention

if args.exp_name == 'univariate_rnc':
	runs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	trs_per_run = 436
	tot_conditions = 151 # 150 image conditions plus the catch images condition
elif args.exp_name == 'multivariate_rnc':
	runs = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
	trs_per_run = 484
	tot_conditions = 151 # 150 image conditions plus the catch images condition

data = []
trs_per_run_check = []

# Iterate through each run of data
for r, run in tqdm(enumerate(runs)):

	# Load the data
	data_dir = os.path.join(args.project_dir, 'in_vivo_validation',
		'in_vivo_fmri_dataset', 'derived', 'sub'+format(args.subject, '02'),
		'alldata', 'run'+format(run, '02'), 'uarf'+format(args.subject, '02')
		+'-'+format(run, '02')+'.nii')
	data_run = nib.load(data_dir)
	trs_per_run_check.append(data_run.shape[3])

	data_run_masked = []

	# Iterate through TRs
	for t in range(trs_per_run_check[r]):

		# Only keep voxels of the ROI of interest
		data_tr = data_run.dataobj
		data_tr = data_tr[:,:,:,t]
		data_tr = data_tr[roi_mask]
		data_run_masked.append(data_tr)
		del data_tr
	del data_run

	# Append each run's timeseries data to list
	data_run_masked = np.asarray(data_run_masked)
	data_run_masked = np.swapaxes(data_run_masked, 0, 1)
	data.append(data_run_masked)
	del data_run_masked

# Raise error if TRs don't match
trs_per_run_check = np.asarray(trs_per_run_check)
if not(all(trs_per_run_check == trs_per_run)):
	raise Exception('TRs do not match!')


# =============================================================================
# Create the design matrices
# =============================================================================
# design -> each run has a corresponding design matrix where each column
# describes a single condition (conditions are repeated across runs). Each
# design matrix is binary with 1 specfing the time (TR) when the stimulus
# is presented on the screen.

# Create the design matrices for each run
design = []

for r in range(len(runs)):

	# Empty run design matrix
	design_run = np.zeros((trs_per_run, tot_conditions), dtype=int)

	# Load the functional data events
	if args.exp_name == 'univariate_rnc':
		events_file = os.path.join(args.project_dir, 'in_vivo_validation',
			'in_vivo_fmri_dataset', 'raw', 'sub-0'+str(args.subject), 'ses-01',
			'func', 'sub-0'+str(args.subject)+'_ses-01_task-univariate_run-'+
			format(r+1, '02')+'_events.tsv') # Download the fMRI events from: https://openneuro.org/datasets/ds005503
	elif args.exp_name == 'multivariate_rnc':
		events_file = os.path.join(args.project_dir, 'in_vivo_validation',
			'in_vivo_fmri_dataset', 'raw', 'sub-0'+str(args.subject), 'ses-02',
			'func', 'sub-0'+str(args.subject)+'_ses-02_task-multivariate_run-'+
			format(r+1, '02')+'_events.tsv') # Download the fMRI events from: https://openneuro.org/datasets/ds005503
	events = np.asarray(pd.read_csv(events_file, sep='\t')['stim_id'])

	for c, cond in enumerate(events):

		# Blank trials
		if np.isnan(cond):
			pass

		# Stimulus trials
		else:
			design_run[(c*args.trs_per_trial),int(cond)-1] = 1

	design.append(design_run)


# =============================================================================
# Run GLMsingle with default parameters to estimate single-trial betas
# =============================================================================
# outputs and figures will be stored in a folder (you can specify its name
# as the 5th output to GLMsingle). model estimates can be also
# saved to the 'results' variable which is the only output of
# GLMsingle.

# optional parameters below can be assigned to a structure, i.e., opt =
# dict('wantlibrary':1, 'wantglmdenoise':1); options are the 6th input to
# GLMsingle.

# there are many options that can be specified; here, we comment on the
# main options that one might want to modify/set. defaults for the options
# are indicated below.

# wantlibrary = 1 -> fit HRF to each voxel 
# wantglmdenoise = 1 -> use GLMdenoise 
# wantfracridge = 1 -> use ridge regression to improve beta estimates 
# chunklen = 50000 -> is the number of voxels that we will
#    process at the same time. for setups with lower memory, you may need to 
#    decrease this number.

# wantmemoryoutputs is a logical vector [A B C D] indicating which of the
#     four model types to return in the output <results>. the user must be
#     careful with this, as large datasets can require a lot of RAM. if you
#     do not request the various model types, they will be cleared from
#     memory (but still potentially saved to disk). default: [0 0 0 1]
#     which means return only the final type-D model.

# wantfileoutputs is a logical vector [A B C D] indicating which of the
#     four model types to save to disk (assuming that they are computed). A
#     = 0/1 for saving the results of the ONOFF model, B = 0/1 for saving
#     the results of the FITHRF model, C = 0/1 for saving the results of the
#     FITHRF_GLMdenoise model, D = 0/1 for saving the results of the
#     FITHRF_GLMdenoise_RR model. default: [1 1 1 1] which means save all
#     computed results to disk.

# numpcstotry (optional) is a non-negative integer indicating the maximum
#     number of GLMdenoise PCs to enter into the model. default: 10.

# fracs (optional) is a vector of fractions that are greater than 0
#     and less than or equal to 1. we automatically sort in descending
#     order and ensure the fractions are unique. these fractions indicate
#     the regularization levels to evaluate using fractional ridge
#     regression (fracridge) and cross-validation. default:
#     fliplr(.05:.05:1). a special case is when <fracs> is specified as a
#     single scalar value. in this case, cross-validation is NOT performed
#     for the type-D model, and we instead blindly use the supplied
#     fractional value for the type-D model.

# create a directory for saving GLMsingle outputs
outputdir_glmsingle = os.path.join(args.project_dir, 'in_vivo_validation',
	'in_vivo_fmri_dataset', 'GLMsingle', args.exp_name+'_experiment', 'sub-'+
	format(args.subject, '02'), 'roi-'+args.roi)
if os.path.isdir(outputdir_glmsingle) == False:
	os.makedirs(outputdir_glmsingle)

opt = dict()

# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [0,0,0,1]
opt['wantmemoryoutputs'] = [0,0,0,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
pprint(glmsingle_obj.params)

start_time = time.time()

print('running GLMsingle...')

# run GLMsingle
results_glmsingle = glmsingle_obj.fit(
	design,
	data,
	args.stimdur,
	args.tr,
	outputdir=outputdir_glmsingle)

# we assign outputs of GLMsingle to the "results_glmsingle" variable.
# note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model,
# where all images are treated as the same condition. these estimates
# could be potentially used to find cortical areas that respond to
# visual stimuli. we want to compare beta weights between conditions
# therefore we are not going to include the ONOFF betas in any analyses of 
# voxel reliability

elapsed_time = time.time() - start_time

print(
	'\telapsed time: ',
	f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)


# =============================================================================
# Summary of important outputs
# =============================================================================
# the outputs of GLMsingle are formally documented in its
# header. here, we highlight a few of the more important outputs:

# R2 -> is model accuracy expressed in terms of R^2 (percentage).

# betasmd -> is the full set of single-trial beta weights (X x Y x Z x
# TRIALS). beta weights are arranged in chronological order.

# HRFindex -> is the 1-index of the best fit HRF. HRFs can be recovered
# with getcanonicalHRFlibrary(stimdur,tr)

# FRACvalue -> is the fractional ridge regression regularization level
# chosen for each voxel. values closer to 1 mean less regularization.
