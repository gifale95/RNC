"""Save the controlling images selected by applying univariate RNC on the
in-silico univariate fMRI responses averaged over all subjects (i.e., with no
subject cross-validation).

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
encoding_models_train_dataset : str
	Dataset on which the encoding models were trained. Possible options are
	'nsd' and 'VisualIllusionRecon'.
roi_pair : str
	Used pairwise ROI combination.
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
n_saved_images : int
	Number saved controlling images.
project_dir : str
	Directory of the project folder.
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
import h5py
import pandas as pd
from PIL import Image
import torchvision
from torchvision import transforms as trn

parser = argparse.ArgumentParser()
parser.add_argument('--encoding_models_train_dataset', type=str, default='nsd')
parser.add_argument('--roi_pair', type=str, default='V1-V2')
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--n_saved_images', type=int, default=25)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../things_database/', type=str)
args = parser.parse_args()

print('>>> Save the univariate RNC controlling images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# ROI names
# =============================================================================
idx = args.roi_pair.find('-')
roi_1 = args.roi_pair[:idx]
roi_2 = args.roi_pair[idx+1:]
rois = [roi_1, roi_2]

if roi_1 == 'hV4':
	roi_1 = 'V4'
if roi_2 == 'hV4':
	roi_2 = 'V4'


# =============================================================================
# Load the univariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models', 'stats', 'cv-0',
	'imageset-'+args.imageset, args.roi_pair, 'stats.npy')

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
	imageset_dir = os.path.join(args.things_dir, '01_image-level',
		'image-paths.csv')
	images = pd.read_csv(imageset_dir, header=None)
	images = images[0].tolist()
	for i, img in enumerate(images):
		new_img = os.path.join(args.things_dir, 'image-database_things', img[7:])
		images[i] = new_img
	transform = trn.Compose([
		trn.Resize((425,425))
		])


# =============================================================================
# Save the univariate RNC controlling images
# =============================================================================
# Save directory
save_dir = os.path.join(args.project_dir, 'univariate_rnc',
	args.encoding_models_train_dataset+'_encoding_models',
	'controlling_images', 'cv-0', 'imageset-'+args.imageset, roi_1+'-'+roi_2)
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

# High ROI 1, High ROI 2
control_images = data_dict['high_1_high_2']
for i, img_num in enumerate(control_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'high_' + roi_1 + '_high_' + roi_2 + \
		'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['high_1_high_2'][i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

# High ROI 1, Low ROI 2
control_images = data_dict['high_1_low_2']
for i, img_num in enumerate(control_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'high_' + roi_1 + '_low_' + roi_2 + \
		'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['high_1_low_2'][i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

# Low ROI 1, High ROI 2
control_images = data_dict['low_1_high_2']
for i, img_num in enumerate(control_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'low_' + roi_1 + '_high_' + roi_2 + \
		'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['low_1_high_2'][i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

# Low ROI 1, Low ROI 2
control_images = data_dict['low_1_low_2']
for i, img_num in enumerate(control_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'low_' + roi_1 + '_low_' + roi_2 + \
		'_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['low_1_low_2'][i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

# Baseline ROI 1
baseline_images = data_dict['baseline_images'][0]
for i, img_num in enumerate(baseline_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'rnc_baseline_'+roi_1+'_img-' + \
		format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['baseline_images'][0,i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

# Baseline ROI 2
baseline_images = data_dict['baseline_images'][1]
for i, img_num in enumerate(baseline_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet_val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img = transform(Image.open(images[img_num]).convert('RGB'))
	img_name = 'rnc_baseline_'+roi_2+'_img-' + \
		format(i+1, '03') + '_' + args.imageset + '-' + \
		format(data_dict['baseline_images'][1,i], '06') + '.png'
	img.save(os.path.join(save_dir, img_name))
