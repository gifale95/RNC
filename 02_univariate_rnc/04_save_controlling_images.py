"""Save the controlling images selected by applying univariate RNC on the
in-silico univariate fMRI responses averaged over all subjects (i.e., with no
subject cross-validation).

This code is available at:
https://github.com/gifale95/RNC/blob/main/02_univariate_rnc/04_save_controlling_images.py

Parameters
----------
rois : list of str
	List of used ROIs.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_saved_images : int
	Number saved controlling images.
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
import h5py
from PIL import Image
import torchvision
from torchvision import transforms as trn
from ned.ned import NED

parser = argparse.ArgumentParser()
parser.add_argument('--rois', type=list, default=['V1', 'V2', 'V3', 'V4'])
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_saved_images', type=int, default=25)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../THINGS/', type=str)
args = parser.parse_args()

print('>>> Save the univariate RNC controlling images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Pairwise ROI comparisons
# =============================================================================
# 0: V1
# 1: V2
# 2: V3
# 3: hV4
r1 = [0, 0, 0, 1, 1, 2]
r2 = [1, 2, 3, 2, 3, 3]


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc', 'stats', 'cv-0',
	'imageset-'+args.imageset, 'stats.npy')

data_dict = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Access the image datasets
# =============================================================================
if args.imageset == 'nsd':
	stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5')
	sf = h5py.File(stimuli_dir, 'r')
	sdataset = sf.get('imgBrick')

elif args.imageset == 'imagenet_val':
	dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val')

elif args.imageset == 'things':
	# https://github.com/gifale95/NED
	ned_object = NED(args.ned_dir)
	_, metadata = ned_object.load_synthetic_neural_responses(
		modality='fmri',
		train_dataset='nsd',
		model='fwrf',
		imageset=args.imageset,
		subject=1,
		roi='V1',
		return_metadata=True
		)
	images = metadata['things_labels']['image_paths']
	transform = trn.Compose([
		trn.Resize((425,425))
		])


# =============================================================================
# Save the univariate RNC controlling images
# =============================================================================
for r in range(len(r1)):

	# Save directory
	save_dir = os.path.join(args.project_dir, 'univariate_rnc',
		'controlling_images', 'cv-0', args.rois[r1[r]]+'-'+args.rois[r2[r]],
		'imageset-'+args.imageset)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	# High ROI 1, High ROI 2
	control_images = data_dict['high_1_high_2'][r]
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[control_images[i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(control_images[i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_dir, 'images',
				images[control_images[i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'high_' + args.rois[r1[r]] + '_high_' + args.rois[r2[r]] + \
			'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
			format(data_dict['high_1_high_2'][r,i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

	# High ROI 1, Low ROI 2
	control_images = data_dict['high_1_low_2'][r]
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[control_images[i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(control_images[i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_dir, 'images',
				images[control_images[i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'high_' + args.rois[r1[r]] + '_low_' + args.rois[r2[r]] + \
			'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
			format(data_dict['high_1_low_2'][r,i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

	# Low ROI 1, High ROI 2
	control_images = data_dict['low_1_high_2'][r]
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[control_images[i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(control_images[i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_dir, 'images',
				images[control_images[i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'low_' + args.rois[r1[r]] + '_high_' + args.rois[r2[r]] + \
			'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
			format(data_dict['low_1_high_2'][r,i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

	# Low ROI 1, Low ROI 2
	control_images = data_dict['low_1_low_2'][r]
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[control_images[i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(control_images[i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_dir, 'images',
				images[control_images[i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'low_' + args.rois[r1[r]] + '_low_' + args.rois[r2[r]] + \
			'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
			format(data_dict['low_1_low_2'][r,i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

	# Baseline ROI 1
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[data_dict['baseline_images'][r1[r],i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(data_dict['baseline_images'][r1[r],i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_dir, 'images',
				images[data_dict['baseline_images'][r1[r],i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'rnc_baseline_'+args.rois[r1[r]]+'_img-' + format(i+1, '03') + \
			'_' + args.imageset + '-' + \
			format(data_dict['baseline_images'][r1[r],i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

	# Baseline ROI 2
	for i in tqdm(range(args.n_saved_images)):
		if args.imageset == 'nsd':
			img = sdataset[data_dict['baseline_images'][r2[r],i]]
			img = Image.fromarray(img).convert('RGB')
		elif args.imageset == 'imagenet_val':
			img, _ = dataset.__getitem__(data_dict['baseline_images'][r2[r],i])
			min_size = min(img.size)
			transform = trn.Compose([
				trn.CenterCrop(min_size),
				trn.Resize((425,425))
				])
			img = transform(img)
		elif args.imageset == 'things':
			img_dir = os.path.join(args.things_di, 'images',,
				images[data_dict['baseline_images'][r2[r],i]])
			img = transform(Image.open(img_dir).convert('RGB'))
		img_name = 'rnc_baseline_'+args.rois[r2[r]]+'_img-' + format(i+1, '03') + \
			'_' + args.imageset + '-' + \
			format(data_dict['baseline_images'][r2[r],i], '06') + '.png'
		img.save(os.path.join(save_dir, img_name))

