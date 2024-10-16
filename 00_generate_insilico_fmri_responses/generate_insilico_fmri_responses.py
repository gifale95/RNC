"""Generate the in silico fMRI responses to images using the Neural Encoding
Dataset (NED): https://github.com/gifale95/NED.

This code is available at:
https://github.com/gifale95/RNC/blob/main/00_generate_insilico_fmri_responses/generate_insilico_fmri_responses.py

Parameters
----------
subject : int
	Subject for which the in silico fMRI responses are generated, out of all 8
	(NSD) subjects.
roi : str
	ROI for which the in silico fMRI responses are generated. Possible choices
	are 'V1', 'V3', 'V2', 'hV4'.
imageset : str
	Image set for which the in silico fMRI responses are generated. Possible
	choices are 'nsd', 'imagenet_val', 'things'.
project_dir : str
	Directory of the project folder.
ned_dir : str
	Directory of the Neural Encoding Dataset.
	https://github.com/gifale95/NED
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
from ned.ned import NED
import h5py
import torchvision
from torchvision import transforms as trn
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../THINGS/', type=str)
args = parser.parse_args()

print('>>> Generate in silico fMRI responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Initialize NED
# =============================================================================
# https://github.com/gifale95/NED
ned_object = NED(args.ned_dir)


# =============================================================================
# Read the images
# =============================================================================
if args.imageset == 'nsd':
	img_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli', 'nsd',
		'nsd_stimuli.hdf5')
	images = h5py.File(img_dir, 'r').get('imgBrick')

elif args.imageset == 'imagenet_val':
	imageset_dir = os.path.join(args.imagenet_dir)
	images = torchvision.datasets.ImageNet(root=imageset_dir, split='val')

elif args.imageset == 'things':
	_, metadata = ned_object.load_insilico_neural_responses(
		modality='fmri',
		train_dataset='nsd',
		model='fwrf',
		imageset=args.imageset,
		subject=1,
		roi='V1',
		return_metadata=True
		)
	images = metadata['things_labels']['image_paths']


# =============================================================================
# Load the trained encoding model weights
# =============================================================================
encoding_model = ned_object.get_encoding_model(
	modality='fmri',
	train_dataset='nsd',
	model='fwrf',
	subject=args.subject,
	roi=args.roi,
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
		img_dir = os.path.join(args.things_dir, 'images', images[i])
		img = np.asarray(Image.open(img_dir).convert('RGB'))

	# Set the images to the correct format for encoding:
	# Must be a 4-D numpy array of shape
	# (Batch size x 3 RGB Channels x Width x Height) consisting of integer
	# values in the range [0, 255]. Furthermore, the images must be of square
	# size (i.e., equal width and height).
	img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
	img = np.expand_dims(img, 0)

	# Generate the in silico fMRI image responses
	insilico_fmri.append(np.squeeze(ned_object.encode(
		encoding_model,
		img,
		return_metadata=False,
		device='auto'
		)))

insilico_fmri = np.asarray(insilico_fmri)
insilico_fmri = insilico_fmri.astype(np.float32)


# =============================================================================
# Save the in silico fMRI responses
# =============================================================================
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'insilico_fmri_responses_sub-' + format(args.subject, '02') + \
	'_roi-' + args.roi + '.h5'

# Save the h5py file
with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
	f.create_dataset('insilico_fmri_responses', data=insilico_fmri,
		dtype=np.float32)
