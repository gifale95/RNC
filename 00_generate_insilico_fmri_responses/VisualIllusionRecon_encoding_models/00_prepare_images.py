"""Prepare the Visual Illusion Reconstruction dataset stimulus images, for
encoding model training and testing.

The images are available upon request from the Visual Illusion Reconstruction
dataset authors:
https://doi.org/10.1126/sciadv.adj3906

This image '.csv' files are available at:
https://github.com/gifale95/RNC/blob/main/00_generate_insilico_fmri_responses/VisualIllusionRecon_encoding_models

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Prepare the Visual Illusion Reconstruction stimulus images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Reformat the stimulus images
# =============================================================================
# Images parent directory
image_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'images')
# Image groups
imagesets = ['ImageNetTraining', 'ImageNetTraining_S6', 'FMD', 'MSCOCO',
	'Illusion', 'Illusion_S4']


# Images reshape size
resize_px_fwrf = 227
resize_px_linear = 224

for imageset in tqdm(imagesets):

	# Get the imageset folder
	if imageset in ['ImageNetTraining', 'ImageNetTraining_S6']:
		imageset_folder = 'ImageNetTraining'
	elif imageset in ['FMD']:
		imageset_folder = 'FMD'
	elif imageset in ['MSCOCO']:
		imageset_folder = 'MSCOCO'
	elif imageset in ['Illusion', 'Illusion_S4']:
		imageset_folder = 'Illusion'

	# Get the image file names
	image_list = pd.read_csv(os.path.join(image_dir, imageset+'.csv'),
		sep="\t", header=None, usecols=[0])[0].tolist()

	# Get the image file formats
	if imageset in ['ImageNetTraining', 'ImageNetTraining_S6']:
		form = '.JPEG'
	elif imageset in ['FMD', 'MSCOCO']:
		form = '.jpg'
	elif imageset in ['Illusion', 'Illusion_S4']:
		form = '.tif'

	# Empty image lists
	image_data_fwrf = []
	image_data_linear = []

	# Load the images
	for img_file in image_list:
		img_dir = os.path.join(image_dir, imageset_folder, img_file+form)
		img_fwrf = np.asarray(
			Image.open(img_dir).convert('RGB').resize((
			resize_px_fwrf,resize_px_fwrf), resample=Image.BILINEAR))
		img_linear = np.asarray(
			Image.open(img_dir).convert('RGB').resize((
			resize_px_linear,resize_px_linear), resample=Image.BILINEAR))
		image_data_fwrf.append(img_fwrf.transpose(2,0,1))
		image_data_linear.append(img_linear.transpose(2,0,1))
		del img_fwrf, img_linear

	# Reformat to int32
	image_data_fwrf = np.asarray(image_data_fwrf).astype(np.int32)
	image_data_linear = np.asarray(image_data_linear).astype(np.int32)


# =============================================================================
# Save the prepared stimulus images
# =============================================================================
	save_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
		'prepared_data', 'images')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	save_file_fwrf = 'images_dataset-' + imageset + '_encoding-fwrf'
	save_file_linear = 'images_dataset-' + imageset + '_encoding-linear'

	np.save(os.path.join(save_dir, save_file_fwrf), image_data_fwrf)
	np.save(os.path.join(save_dir, save_file_linear), image_data_linear)
	del image_data_fwrf, image_data_linear
