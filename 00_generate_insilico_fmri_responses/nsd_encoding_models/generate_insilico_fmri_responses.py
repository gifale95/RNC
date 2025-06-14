"""Generate the in silico fMRI responses to images using the Brain Encoding
Response Generator (BERG): https://github.com/gifale95/BERG.

This code additionally stores the noise ceiling signal-to-noise ratio of the
fMRI responses.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
subject : int
	Subject for which the in silico fMRI responses are generated, out of all 8
	(NSD) subjects.
roi : str
	ROI for which the in silico fMRI responses are generated.
imageset : str
	Image set for which the in silico fMRI responses are generated. Possible
	choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.
berg_dir : str
	Directory of the Brain Encoding Response Generator.
	https://github.com/gifale95/BERG
nsd_dir : str
	Directory of the Natural Scenes Dataset.
	https://naturalscenesdataset.org/
imagenet_dir : str
	Directory of the ImageNet image set.
	https://www.image-net.org/challenges/LSVRC/2012/index.php
things_dir : str
	Directory of the THINGS database.
	https://osf.io/jum2f/

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from berg import BERG
import pandas as pd
import h5py
import torchvision
from torchvision import transforms as trn
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--roi', default='FFA', type=str)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--berg_dir', default='../brain-encoding-reponse-generator/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../things_database/', type=str)
args = parser.parse_args()

print('>>> Generate in silico fMRI responses - NSD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize BERG
# =============================================================================
# https://github.com/gifale95/BERG
berg_object = BERG(args.berg_dir)


# =============================================================================
# Read the images
# =============================================================================
if args.imageset == 'nsd':
	imageset_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5')
	images = h5py.File(imageset_dir, 'r').get('imgBrick')

elif args.imageset == 'imagenet_val':
	imageset_dir = os.path.join(args.imagenet_dir)
	images = torchvision.datasets.ImageNet(root=imageset_dir, split='val')

elif args.imageset == 'things':
	imageset_dir = os.path.join(args.things_dir, '01_image-level',
		'image-paths.csv')
	images = pd.read_csv(imageset_dir, header=None)
	images = images[0].tolist()
	for i, img in enumerate(images):
		new_img = os.path.join(args.things_dir, 'image-database_things',
			img[7:])
		images[i] = new_img


# =============================================================================
# Load the trained encoding model weights
# =============================================================================
# ROIs 'FFA' and 'VWFA' are divided into two parts (anterior and posterior):
# https://cvnlab.slite.page/p/X_7BBMgghj/ROIs
if args.roi in ['FFA', 'VWFA']:
	# Encoding model for ROI split 1
	encoding_model_1 = berg_object.get_encoding_model(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		selection={'roi': args.roi+'-1'},
		device='auto'
		)
	# Encoding model for ROI split 2
	encoding_model_2 = berg_object.get_encoding_model(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		selection={'roi': args.roi+'-2'},
		device='auto'
		)
else:
	encoding_model = berg_object.get_encoding_model(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		selection={'roi': args.roi},
		device='auto'
		)


# =============================================================================
# Generate the in silico fMRI responses to images
# =============================================================================
insilico_fmri = []

for i in tqdm(range(len(images))):

	if args.imageset == 'nsd':
		img = images[i]

	elif args.imageset == 'imagenet_val':
		img, _ = images.__getitem__(i)
		transform = trn.Compose([trn.CenterCrop(min(img.size))])
		img = np.asarray(transform(img))

	elif args.imageset == 'things':
		img = np.asarray(Image.open(images[i]).convert('RGB'))

	# Set the images to the correct format for encoding:
	# Must be a 4-D numpy array of shape
	# (Batch size x 3 RGB Channels x Width x Height) consisting of integer
	# values in the range [0, 255]. Furthermore, the images must be of square
	# size (i.e., equal width and height).
	img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
	img = np.expand_dims(img, 0)

	# Generate the in silico fMRI image responses
	if args.roi in ['FFA']:
		insilico_fmri_1 = np.squeeze(berg_object.encode(
			encoding_model_1,
			img,
			return_metadata=False
			))
		insilico_fmri_2 = np.squeeze(berg_object.encode(
			encoding_model_2,
			img,
			return_metadata=False
			))
		insilico_fmri.append(np.append(insilico_fmri_1, insilico_fmri_2))
		del insilico_fmri_1, insilico_fmri_2
	else:
		insilico_fmri.append(np.squeeze(berg_object.encode(
			encoding_model,
			img,
			return_metadata=False
			)))

insilico_fmri = np.asarray(insilico_fmri)
insilico_fmri = insilico_fmri.astype(np.float32)


# =============================================================================
# Save the in silico fMRI responses
# =============================================================================
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'nsd_encoding_models', 'insilico_fmri', 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'insilico_fmri_responses_sub-' + format(args.subject, '02') + \
	'_roi-' + args.roi + '.h5'

# Save the h5py file
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
	f.create_dataset('insilico_fmri_responses', data=insilico_fmri,
		dtype=np.float32)


# =============================================================================
# Load and save the ncsnr
# =============================================================================
# ROIs 'FFA' and 'VWFA' are divided into two parts (anterior and posterior):
# https://cvnlab.slite.page/p/X_7BBMgghj/ROIs
if args.roi in ['FFA', 'VWFA']:
	# Load the metadata for ROI split 1
	metadata_1 = berg_object.get_model_metadata(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		roi=args.roi+'-1'
		)
	# Load the metadata for ROI split 2
	metadata_2 = berg_object.get_model_metadata(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		roi=args.roi+'-2'
		)
	# Extract the ncsnr
	ncsnr = np.append(metadata_1['fmri']['ncsnr'], metadata_2['fmri']['ncsnr'])
else:
	# Load the metadata
	metadata = berg_object.get_model_metadata(
		model_id='fmri-nsd-fwrf',
		subject=args.subject,
		roi=args.roi
		)
	# Extract the ncsnr
	ncsnr = metadata['fmri']['ncsnr']

# Save the ncsnr
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'nsd_encoding_models', 'insilico_fmri')
file_name = file_name = 'ncsnr_sub-' + format(args.subject, '02') + \
	'_roi-' + args.roi + '.npy'
np.save(os.path.join(save_dir, file_name), ncsnr)
