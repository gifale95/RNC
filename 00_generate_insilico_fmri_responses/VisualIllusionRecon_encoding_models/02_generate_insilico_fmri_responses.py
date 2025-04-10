"""Generate the in silico fMRI responses to images using the encoding models
trained on the Visual Illusion Reconstruction dataset.

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
subject : int
	Used subject, out of the 7 Visual Illusion Reconstruction dataset subjects.
roi : str
	ROI for which the in silico fMRI responses are generated.
imageset : str
	Image set for which the in silico fMRI responses are generated. Possible
	choices are 'nsd', 'imagenet_val', 'things'.
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
from tqdm import tqdm
import h5py
import pandas as pd
import torchvision
from torchvision import transforms as trn
from PIL import Image
import torch

from src_new.load_nsd import image_feature_fn
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF

parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
parser.add_argument('--nsd_dir', default='../natural-scenes-dataset/', type=str)
parser.add_argument('--imagenet_dir', default='../ILSVRC2012/', type=str)
parser.add_argument('--things_dir', default='../things_database/', type=str)
args = parser.parse_args()

print('>>> Generate in silico fMRI responses <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
	imageset_dir = os.path.join(args.things_dir, '01_image-level',
		'image-paths.csv')
	images = pd.read_csv(imageset_dir, header=None)
	images = images[0].tolist()
	for i, img in enumerate(images):
		new_img = os.path.join(args.things_dir, 'image-database_things',
			img[7:])
		images[i] = new_img


# =============================================================================
# Get the voxel number for the selected subject and ROI
# =============================================================================
data_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'prepared_data', 'fmri', 'fmri_sub-0'+str(args.subject)+'_roi-'+args.roi+
	'.npy')
data = np.load(data_dir, allow_pickle=True).item()
voxel_num = data['fmri']['ImageNetTraining'].shape[1]
del data


# =============================================================================
# Load the trained encoding models
# =============================================================================
# Load the trained encoding model weights
model_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'VisualIllusionRecon_encoding_models', 'trained_encoding_models',
	'weights_sub-0'+str(args.subject)+'_roi-'+args.roi+'.pt')
trained_model = torch.load(model_dir, weights_only=False,
	map_location=torch.device('cpu'))
stim_mean = trained_model['stim_mean']

# Model instantiation
# Model functions
_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)
def _model_fn(_ext, _con, _x):
	'''model consists of an extractor (_ext) and a connection model (_con)'''
	_y, _fm, _h = _ext(_x)
	return _con(_fm)
def _pred_fn(_ext, _con, xb):
	return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))
# Shared encoder model
img_chan = 3
resize_px = 227
stim_data = np.zeros((20, img_chan, resize_px, resize_px), dtype=np.float32)
for i in range(20):
	if args.imageset == 'nsd':
		img = images[i]
		img = Image.fromarray(np.uint8(img))
	elif args.imageset == 'imagenet_val':
		img, _ = images.__getitem__(i)
	elif args.imageset == 'things':
		img = Image.open(os.path.join(args.things_dir,
			images[i])).convert('RGB')
	min_size = min(img.size)
	transform = trn.Compose([
		trn.CenterCrop(min_size),
		trn.Resize((resize_px,resize_px))
	])
	img = transform(img)
	img = np.asarray(img)
	img = img.transpose(2,0,1)
	img = image_feature_fn(img)
	stim_data[i] = img
backbone_model = Encoder(mu=stim_mean, trunk_width=64,
	use_prefilter=1).to(device)
rec, fmaps, h = backbone_model(torch.from_numpy(stim_data).to(device))
# Subject specific FWRF models
fwrf_model = Torch_LayerwiseFWRF(fmaps, nv=voxel_num,
	pre_nl=_log_act_fn, post_nl=_log_act_fn,
	dtype=np.float32).to(device)

# Load the pretrained weights into the model
backbone_model.load_state_dict(trained_model['best_params']['enc'])
fwrf_model.load_state_dict(trained_model['best_params']['fwrf'])
backbone_model.eval()
fwrf_model.eval()


# =============================================================================
# Generate the in silico fMRI responses to images
# =============================================================================
insilico_fmri = []

with torch.no_grad():
	for i in tqdm(range(len(images))):
		if args.imageset == 'nsd':
			img = images[i]
			img = Image.fromarray(np.uint8(img))
		elif args.imageset == 'imagenet_val':
			img, _ = images.__getitem__(i)
		elif args.imageset == 'things':
			img = Image.open(images[i]).convert('RGB')
		min_size = min(img.size)
		transform = trn.Compose([
			trn.CenterCrop(min_size),
			trn.Resize((resize_px,resize_px))
		])
		img = transform(img)
		img = np.asarray(img)
		img = img.transpose(2,0,1)
		img = image_feature_fn(img)
		img = np.expand_dims(img, 0)
		insilico_fmri.append(subject_pred_pass(_pred_fn, backbone_model,
			fwrf_model, img, batch_size=1))

insilico_fmri = np.squeeze(np.asarray(insilico_fmri)).astype(np.float32)


# =============================================================================
# Save the in silico fMRI responses
# =============================================================================
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'VisualIllusionRecon_encoding_models', 'insilico_fmri', 'imageset-'+
	args.imageset)

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
# Load the ncsnr
data_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'prepared_data', 'fmri', 'fmri_sub-0'+str(args.subject)+'_roi-'+
	args.roi+'.npy')
data = np.load(data_dir, allow_pickle=True).item()
ncsnr = data['ncsnr_all']

# Save the ncsnr
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'VisualIllusionRecon_encoding_models', 'insilico_fmri')
file_name = file_name = 'ncsnr_sub-' + format(args.subject, '02') + \
	'_roi-' + args.roi + '.npy'
np.save(os.path.join(save_dir, file_name), ncsnr)
