"""Save the controlling images selected by applying multivariate RNC on the
in-silico univariate fMRI responses averaged over all subjects (i.e., with no
subject cross-validation).

This code is available at:
https://github.com/gifale95/RNC/blob/main/04_multivariate_rnc/07_save_controlling_images.py

Parameters
----------
roi_pair : int
	Integer indicating the chosen pairwise ROI combination for which to compute
	the baseline RSA score. Possible values are '0' (V1-V2), '1' (V1-V3), '2'
	(V1-hV4), '3' (V2-V3), '4' (V2-hV4), '5' (V3-hV4).
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
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
import h5py
from PIL import Image
import torchvision
from torchvision import transforms as trn
from ned.ned import NED

parser = argparse.ArgumentParser()
parser.add_argument('--roi_pair', type=int, default=2)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--ned_dir', default='../neural_encoding_dataset/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../THINGS/', type=str)
args = parser.parse_args()

print('>>> Save the multivariate RNC controlling images <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Other parameters
# =============================================================================
# ROI pair combinations
# 0 --> V1 - V2
# 1 --> V1 - V3
# 2 --> V1 - V4
# 3 --> V2 - V3
# 4 --> V2 - V4
# 5 --> V3 - V4
roi_comb_names = [['V1', 'V2'], ['V1', 'V3'], ['V1', 'hV4'], ['V2', 'V3'],
	['V2', 'hV4'], ['V3', 'hV4']]
roi_1 = roi_comb_names[args.roi_pair][0]
roi_2 = roi_comb_names[args.roi_pair][1]


# =============================================================================
# Load the multivariate RNC stats
# =============================================================================
data_dir = os.path.join(args.project_dir, 'multivariate_rnc', 'stats', 'cv-0',
	'imageset-'+args.imageset, roi_1+'-'+roi_2, 'stats.npy')

data_dict = np.load(data_dir, allow_pickle=True).item()

alignment_images = data_dict['best_generation_image_batches']['align'][-1]
disentanglement_images = data_dict['best_generation_image_batches']\
	['disentangle'][-1]
baseline_images = data_dict['baseline_images']


# =============================================================================
# Access the image datasets
# =============================================================================
if args.imageset == 'nsd':
	stimuli_dir = os.path.join(args.nsd_dir, 'nsddata_stimuli', 'stimuli',
		'nsd', 'nsd_stimuli.hdf5')
	sf = h5py.File(stimuli_dir, 'r')
	sdataset = sf.get('imgBrick')

elif args.imageset == 'imagenet-val':
	dataset = torchvision.datasets.ImageNet(root=args.imagenet_dir,
		split='val')

elif args.imageset == 'things':
	ned_object = NED(args.ned_dir)
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
	transform = trn.Compose([
		trn.Resize((425,425))
		])


# =============================================================================
# Save the aligning images
# =============================================================================
if roi_2 == 'hV4':
	roi_2 = 'V4'
save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	'controlling_images', 'cv-0', roi_1+'-'+roi_2, 'imageset-'+args.imageset)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

for i, img_num in enumerate(alignment_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet-val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img_dir = os.path.join(args.things_dir, 'images', images[img_num])
		img = transform(Image.open(img_dir).convert('RGB'))
	img_name = 'align_img-' + format(i+1, '03') + '_' + args.imageset + '-' + \
		format(img_num, '06') + '.png'
	img.save(os.path.join(save_dir, img_name))


# =============================================================================
# Save the disentangling images
# =============================================================================
for i, img_num in enumerate(disentanglement_images):
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
		img_dir = os.path.join(args.things_dir, 'images', images[img_num])
		img = transform(Image.open(img_dir).convert('RGB'))
	img_name = 'disentangle_img-' + format(i+1, '03') + '_' + args.imageset + \
		'-' + format(img_num, '06') + '.png'
	img.save(os.path.join(save_dir, img_name))


# =============================================================================
# Save the baseline images
# =============================================================================
for i, img_num in enumerate(baseline_images):
	if args.imageset == 'nsd':
		img = sdataset[img_num]
		img = Image.fromarray(img).convert('RGB')
	elif args.imageset == 'imagenet-val':
		img, _ = dataset.__getitem__(img_num)
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((425,425))
			])
		img = transform(img)
	elif args.imageset == 'things':
		img_dir = os.path.join(args.things_dir, 'images', images[img_num])
		img = transform(Image.open(img_dir).convert('RGB'))
	img_name = 'rnc_baseline_img-' + format(i+1, '03') + '_' + args.imageset + \
		'-' + format(img_num, '06') + '.png'
	img.save(os.path.join(save_dir, img_name))

