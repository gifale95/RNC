"""Convert the V4v/V4d labels delineated with freeview to functional space.

The V4 ventral and dorsal delineations can be found here:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/delineate_nsd_v4_dorsal_ventral/nsd_v4_delineations

This code is available at:
https://github.com/gifale95/RNC/blob/main/05_multivariate_rnc_retinotopy/delineate_nsd_v4_dorsal_ventral/v4_surface_to_vol.py

Parameters
----------
all_subjects : list of int
	List of all subjects. These are the 8 (NSD) subjects for which there are
	synthetic fMRI responses.
nsd_dir : str
	Directory of the NSD.
	https://naturalscenesdataset.org/

"""

import argparse
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn.plotting import plot_glass_brain
from nsdcode.nsd_mapdata import NSDmapdata

parser = argparse.ArgumentParser()
parser.add_argument('--all_subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
args = parser.parse_args()


# =============================================================================
# Subjects loop
# =============================================================================
for sub in tqdm(args.all_subjects):


# =============================================================================
# Load brain surface templates
# =============================================================================
	# LH
	lh_surface_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'lh.prf-visualrois.mgz')
	lh_surface = nib.load(lh_surface_dir)
	lh_surface_data = lh_surface.get_fdata()
	lh_surface_affine = lh_surface.affine
	lh_surface_header = lh_surface.header

	# RH
	rh_surface_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'rh.prf-visualrois.mgz')
	rh_surface = nib.load(rh_surface_dir)
	rh_surface_data = rh_surface.get_fdata()
	rh_surface_affine = rh_surface.affine
	rh_surface_header = rh_surface.header


# =============================================================================
# Load the label files
# =============================================================================
	# LH V4v
	lh_v_label_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'lh.V4v.label')
	lh_v_label = nib.freesurfer.read_label(lh_v_label_dir)
	# LH V4d
	lh_d_label_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'lh.V4d.label')
	lh_d_label = nib.freesurfer.read_label(lh_d_label_dir)

	# RH V4v
	rh_v_label_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'rh.V4v.label')
	rh_v_label = nib.freesurfer.read_label(rh_v_label_dir)
	# RH V4d
	rh_d_label_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'rh.V4d.label')
	rh_d_label = nib.freesurfer.read_label(rh_d_label_dir)


# =============================================================================
# Convert surface (.label) files to subject-native surface space
# =============================================================================
	# LH V4v
	lh_v_label_surface = np.zeros((lh_surface_data.shape))
	lh_v_label_surface[lh_v_label] = 1
	lh_v_label_surface_mgh = nib.MGHImage(lh_v_label_surface, lh_surface_affine,
		lh_surface_header)
	# LH V4d
	lh_d_label_surface = np.zeros((lh_surface_data.shape))
	lh_d_label_surface[lh_d_label] = 1
	lh_d_label_surface_mgh = nib.MGHImage(lh_d_label_surface, lh_surface_affine,
		lh_surface_header)

	# RH V4v
	rh_v_label_surface = np.zeros((rh_surface_data.shape))
	rh_v_label_surface[rh_v_label] = 1
	rh_v_label_surface_mgh = nib.MGHImage(rh_v_label_surface, rh_surface_affine,
		rh_surface_header)
	# RH V4d
	rh_d_label_surface = np.zeros((rh_surface_data.shape))
	rh_d_label_surface[rh_d_label] = 1
	rh_d_label_surface_mgh = nib.MGHImage(rh_d_label_surface, rh_surface_affine,
		rh_surface_header)


# =============================================================================
# Save the surfaces
# =============================================================================
	# LH V4v
	lh_v_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'lh.V4v.mgz')
	nib.save(lh_v_label_surface_mgh, lh_v_save_dir)
	# LH V4d
	lh_d_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'lh.V4d.mgz')
	nib.save(lh_d_label_surface_mgh, lh_d_save_dir)

	# RH V4v
	rh_v_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'rh.V4v.mgz')
	nib.save(rh_v_label_surface_mgh, rh_v_save_dir)
	# RH V4d
	rh_d_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'freesurfer', 'subj'+
		format(sub, '02'), 'label', 'rh.V4d.mgz')
	nib.save(rh_d_label_surface_mgh, rh_d_save_dir)


# =============================================================================
# Map data from surface space to functional space (V4v)
# =============================================================================
	nsd = NSDmapdata(args.nsd_dir)

	sourcedata = np.r_[
		np.tile(
			f'{args.nsd_dir}/nsddata/freesurfer/subj{sub:02d}/label/lh.V4v.mgz', 3),
		np.tile(
			f'{args.nsd_dir}/nsddata/freesurfer/subj{sub:02d}/label/rh.V4v.mgz', 3)
			].tolist()

	sourcespace = [
		'lh.layerB1',
		'lh.layerB2',
		'lh.layerB3',
		'rh.layerB1',
		'rh.layerB2',
		'rh.layerB3'
		]

	targetspace = 'anat0pt8'

	surface_to_anat = nsd.fit(
		sub,
		sourcespace,
		targetspace,
		sourcedata,
		interptype='surfacewta',
		badval=-1,
		outputfile=None)

	sourcespace = 'anat0pt8'
	targetspace = 'func1pt8'

	v4v_func = nsd.fit(
		sub,
		sourcespace,
		targetspace,
		sourcedata=surface_to_anat,
		interptype='nearest', # wta
		badval=-1,
		outputfile=None)


# =============================================================================
# Map data from surface space to functional space (V4d)
# =============================================================================
	nsd = NSDmapdata(args.nsd_dir)

	sourcedata = np.r_[
		np.tile(
			f'{args.nsd_dir}/nsddata/freesurfer/subj{sub:02d}/label/lh.V4d.mgz', 3),
		np.tile(
			f'{args.nsd_dir}/nsddata/freesurfer/subj{sub:02d}/label/rh.V4d.mgz', 3)
			].tolist()

	sourcespace = [
		'lh.layerB1',
		'lh.layerB2',
		'lh.layerB3',
		'rh.layerB1',
		'rh.layerB2',
		'rh.layerB3'
		]

	targetspace = 'anat0pt8'

	surface_to_anat = nsd.fit(
		sub,
		sourcespace,
		targetspace,
		sourcedata,
		interptype='surfacewta',
		badval=-1,
		outputfile=None)

	sourcespace = 'anat0pt8'
	targetspace = 'func1pt8'

	v4d_func = nsd.fit(
		sub,
		sourcespace,
		targetspace,
		sourcedata=surface_to_anat,
		interptype='nearest', # wta
		badval=-1,
		outputfile=None)


# =============================================================================
# Get nifti images metadata
# =============================================================================
	roi_masks_dir = os.path.join(args.nsd_dir, 'nsddata', 'ppdata', 'subj'+
		format(sub, '02'), 'func1pt8mm', 'roi', 'prf-visualrois.nii.gz')
	roi_mask_volume = nib.load(roi_masks_dir)

	volume_data = roi_mask_volume.get_fdata()
	volume_shape = volume_data.shape
	affine = roi_mask_volume.affine
	header = roi_mask_volume.header


# =============================================================================
# Convert the functional data to nifti images
# =============================================================================
	# V4v
	volume_v4v = v4v_func == 1
	volume_v4v = nib.Nifti1Image(volume_v4v, affine, header)

	# V4d
	volume_v4d = v4d_func == 1
	volume_v4d = nib.Nifti1Image(volume_v4d, affine, header)


# =============================================================================
# Save the nifti images
# =============================================================================
	# V4v
	v4v_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'ppdata', 'subj'+
		format(sub, '02'), 'func1pt8mm', 'roi', 'V4v.nii.gz')
	nib.save(volume_v4v, v4v_save_dir)

	# V4d
	vdv_save_dir = os.path.join(args.nsd_dir, 'nsddata', 'ppdata', 'subj'+
		format(sub, '02'), 'func1pt8mm', 'roi', 'V4d.nii.gz')
	nib.save(volume_v4d, vdv_save_dir)


# =============================================================================
# Plot the nifti images
# =============================================================================
	# Original V4 definition
	# volume_data_v4 = volume_data == 7
	# roi_mask_volume = nib.Nifti1Image(volume_data_v4, affine, header)
	# plot_glass_brain(roi_mask_volume)

	# New V4v/V4d definition
	# plot_glass_brain(volume_v4v)
	# plot_glass_brain(volume_v4d)

