"""Train fMRI encoding models on the Visual Illusion Reconstruction datset
using the feature-weighted receptive field (fwrf) encoding model (St-Yves &
Naselaris, 2018).

This code is an adapted version of the code from the paper:
Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle,
	L.T., Nau, M., Caron, B., Pestilli, F., Charest, I. and Hutchinson,
	J.B., 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience
	and artificial intelligence. Nature neuroscience, 25(1), pp.116-126.

The original code, written by Ghislain St-Yves, can be found here:
https://github.com/styvesg/nsd_gnet8x

This code is available at:
https://github.com/gifale95/RNC

Parameters
----------
subject : int
	Used subject, out of the 7 Visual Illusion Reconstruction dataset subjects.
roi : str
	Used ROI.
trained : int
	If [1] the entire model is trained, if [0] only the fwrf projection heads
	(and not the backbone) are trained.
random_prefilter : int
	If 1 import weights of pre-filter from a pre-trained AlexNet, if 0 use
	randomly intialized weights.
train_prefilter : int
	If 0 the prefilter weights are frozen during training phase 1. If 1 the
	prefilter weights are trained during training phase 1.
	from a pre-trained AlexNet.
use_prefilter : int
	If 0 the prefilter features are not used to encode the brain data, if 1 the
	prefilter features are used to encode the brain data.
train_phases : int
	Whether to run 1, 2 or 3 training phases.
epochs_phase_1 : int
	Number of epochs for the first training phase.
epochs_phase_2 : int
	Number of epochs for the second training phase.
epochs_phase_3 : int
	Number of epochs for the third training phase.
lr : float
	Learning rate.
weight_decay : float
	Weight decay coefficient.
batch_size : int
	Batch size for weight update.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import random
import numpy as np
import torch
from src_new.load_nsd import image_feature_fn
from src_new.torch_joint_training_unpacked_sequences import *
from src_new.torch_gnet import Encoder
from src_new.torch_mpf import Torch_LayerwiseFWRF
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--roi', type=str, default='V1')
parser.add_argument('--trained', default=1, type=int)
parser.add_argument('--random_prefilter', type=int, default=1)
parser.add_argument('--train_prefilter', type=int, default=1)
parser.add_argument('--use_prefilter', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Train encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)
print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())


# =============================================================================
# Set random seeds to make results reproducible
# =============================================================================
# Random seeds
seed = (args.subject * 100) + (np.sum([ord(c) for c in args.roi]))
seed = int(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generator object for DataLoader random batching
g_cpu = torch.Generator()
g_cpu.manual_seed(seed)


# =============================================================================
# Computing resources
# =============================================================================
# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda:1") #cuda

if device == 'cuda':
	torch.backends.cudnn.enabled=True
	print ('#device:', torch.cuda.device_count())
	print ('device#:', torch.cuda.current_device())
	print ('device name:', torch.cuda.get_device_name(
		torch.cuda.current_device()))


# =============================================================================
# Output directories
# =============================================================================
save_dir = os.path.join(args.project_dir, 'insilico_fmri_responses',
	'VisualIllusionRecon_encoding_models')
subj_roi_dir = 'sub-0' + str(args.subject) + '_roi-' + args.roi

# Create models output directory
model_output_dir = os.path.join(save_dir, 'trained_encoding_models')
if not os.path.exists(model_output_dir):
	os.makedirs(model_output_dir)

# TensorBoard output directory
tensorboard_parent = os.path.join(save_dir, 'tensorboard', subj_roi_dir)
if not os.path.exists(tensorboard_parent):
	os.makedirs(tensorboard_parent)


# =============================================================================
# Load the stimulus images
# =============================================================================
# Images parent directory
img_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'prepared_data', 'images')

# Load the training images
trn_images = []
if args.subject == 6:
	imagesets = ['ImageNetTraining_S6', 'FMD', 'MSCOCO']
else:
	imagesets = ['ImageNetTraining', 'FMD', 'MSCOCO']
for imageset in imagesets:
	trn_images.append(image_feature_fn(np.load(os.path.join(img_dir,
	'images_dataset-'+imageset+'_encoding-fwrf.npy'))))
trn_images = np.concatenate(trn_images, 0)

trn_images_mean = np.mean(trn_images, axis=(0,2,3), keepdims=True)


# =============================================================================
# Load the fMRI data
# =============================================================================
# Load the fMRI data
fmri_dir = os.path.join(args.project_dir, 'VisualIllusionRecon_dataset',
	'prepared_data', 'fmri', 'fmri_sub-0'+str(args.subject)+'_roi-'+args.roi+
	'.npy')
fmri_data = np.load(fmri_dir, allow_pickle=True).item()

# Get the fMRI responses
fmri = []
imagesets = ['ImageNetTraining', 'FMD', 'MSCOCO']
for imageset in imagesets:
	fmri.append(fmri_data['fmri'][imageset])
fmri = np.concatenate(fmri, 0)
voxel_num = fmri.shape[1]

# Get the fMRI responses metadata
fmri_metadata = fmri_data

# Append the image indices of all images
image_index = fmri_metadata['image_index']
image_index_all = np.append(image_index['ImageNetTraining'],
	image_index['FMD']+len(np.unique(image_index['ImageNetTraining'])))
image_index_all = np.append(image_index_all, image_index['MSCOCO']+
	len(np.unique(image_index['ImageNetTraining']))+
	len(np.unique(image_index['FMD'])))

# Delete unused variables
del fmri_metadata['fmri'], fmri_data, image_index


# =============================================================================
# Split the dataset into training, validation and test partitions
# =============================================================================
# The training split consists of the first 6 (out of 8) images for each of the
# 150 'NaturalImageTraining' categories (for a total of 900 images); plus the
# first 80 (out of 100) images from each of the 10 'FMD' categories (for a
# total of 800 images); plus the first 800 (out of 1,000) 'MSCOCO' images.
# Thus, the training split includes a total of 2,500 images.

# The validation split consists of the 7th (out of 8) image for each of the
# 150 'NaturalImageTraining' categories (for a total of 150 images); plus
# images 80 to 90 (out of 100) from each of the 10 'FMD' categories (for a
# total of 100 images); plus images 800 to 900 (out of 1,000) from 'MSCOCO'.
# Thus, the validation split includes a total of 350 images.

# The testing split consists of the 8th (out of 8) image for each of the
# 150 'NaturalImageTraining' categories (for a total of 150 images); plus
# images 90 to 100 (out of 100) from each of the 10 'FMD' categories (for a
# total of 100 images); plus images 900 to 1,000 (out of 1,000) from 'MSCOCO'.
# Thus, the testing split includes a total of 350 images.

# Unique image conditions
unique_images = np.unique(image_index_all)

# Train data partition
trn_voxel_data = np.empty(0)
# Validation data partition (called 'holdout' in the code)
hld_voxel_data = np.empty(0)

# Select the training image conditions
trn_img = np.empty(0, dtype=np.int32)
# 'NaturalImageTraining'
categories = 150
exemplars = 8
trn_exemplars = np.arange(6)
for c in range(categories):
	trn_img = np.append(trn_img, trn_exemplars+(c*exemplars))
# 'FMD'
categories = 10
exemplars = 100
trn_exemplars = np.arange(80)
offset = 1200
for c in range(categories):
	trn_img = np.append(trn_img, trn_exemplars+(c*exemplars)+offset)
# 'MSCOCO'
trn_exemplars = np.arange(800)
offset = 2200
trn_img = np.append(trn_img, trn_exemplars+offset)


# Select the validation image conditions
hld_img = np.empty(0, dtype=np.int32)
# 'NaturalImageTraining'
categories = 150
exemplars = 8
trn_exemplars = np.arange(6, 7)
for c in range(categories):
	hld_img = np.append(hld_img, trn_exemplars+(c*exemplars))
# 'FMD'
categories = 10
exemplars = 100
trn_exemplars = np.arange(80, 90)
offset = 1200
for c in range(categories):
	hld_img = np.append(hld_img, trn_exemplars+(c*exemplars)+offset)
# 'MSCOCO'
trn_exemplars = np.arange(800, 900)
offset = 2200
hld_img = np.append(hld_img, trn_exemplars+offset)

# Divide the data into training and validation
for img in unique_images:
	idx = np.where(image_index_all == img)[0]
	if img in hld_img:
		if len(hld_voxel_data) == 0:
			hld_voxel_data = fmri[idx]
			hld_stim_ordering = image_index_all[idx]
		else:
			hld_voxel_data = np.append(hld_voxel_data, fmri[idx], 0)
			hld_stim_ordering = np.append(hld_stim_ordering,
				image_index_all[idx], 0)
	elif img in trn_img:
		if len(trn_voxel_data) == 0:
			trn_voxel_data = fmri[idx]
			trn_stim_ordering = image_index_all[idx]
		else:
			trn_voxel_data = np.append(trn_voxel_data, fmri[idx], 0)
			trn_stim_ordering = np.append(trn_stim_ordering,
				image_index_all[idx], 0)

del fmri, image_index_all


# =============================================================================
# Model instantiation
# =============================================================================
_log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)

# Backbone encoder model
backbone_model = Encoder(mu=trn_images_mean, trunk_width=64,
	use_prefilter=args.use_prefilter).to(device)
rec, fmaps, h = backbone_model(torch.from_numpy(trn_images[:20]).to(device))

# FWRF model
fwrf_model = Torch_LayerwiseFWRF(fmaps, nv=voxel_num, pre_nl=_log_act_fn, \
	post_nl=_log_act_fn, dtype=np.float32).to(device)


# =============================================================================
# Load prefilter weights from a trained AlexNet
# =============================================================================
if args.random_prefilter == 0:
	try:
		from torch.hub import load_state_dict_from_url
	except ImportError:
		from torch.utils.model_zoo import load_url as load_state_dict_from_url

	# Load the AlexNet weights
	state_dict = load_state_dict_from_url(
		'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
		progress=True)

	# Rename dictionary keys to match new breakdown
	pre_state_dict = {}
	pre_state_dict['conv1.0.weight'] = state_dict.pop('features.0.weight')
	pre_state_dict['conv1.0.bias'] = state_dict.pop('features.0.bias')
	pre_state_dict['conv2.0.weight'] = state_dict.pop('features.3.weight')
	pre_state_dict['conv2.0.bias'] = state_dict.pop('features.3.bias')

	# Add the AlexNet weights to the prefilter network
	backbone_model.pre.load_state_dict(pre_state_dict)

# If "args.trained == 0" do NOT train the shared model layers
if args.trained == 0:
	for param in backbone_model.parameters():
		param.requires_grad = False


# =============================================================================
# Loss functions, etc.
# =============================================================================
fpX = np.float32

def _model_fn(_ext, _con, _x):
	'''model consists of an extractor (_ext) and a connection model (_con)'''
	_y, _fm, _h = _ext(_x)
	return _con(_fm)

def _smoothness_loss_fn(_rf, n):
	delta_x = torch.sum(torch.pow(torch.abs(_rf[:,1:] - _rf[:,:-1]), n))
	delta_y = torch.sum(torch.pow(torch.abs(_rf[:,:,1:] - _rf[:,:,:-1]), n))
	return delta_x + delta_y

def vox_loss_fn(r, v, nu=0.5, delta=1.):
	#err = torch.sum(huber(r, v, delta), dim=0)
	err = torch.sum((r - v)**2, dim=0)
	# Squared correlation coefficient with 'leak'
	cr = r - torch.mean(r, dim=0, keepdim=True)
	cv = v - torch.mean(v, dim=0, keepdim=True)
	wgt = torch.clamp(torch.pow(torch.mean(cr*cv, dim=0), 2) / \
		((torch.mean(cr**2, dim=0)) * (torch.mean(cv**2, dim=0)) + 1e-6), \
		min=nu, max=1).detach()
	weighted_err = wgt * err # error per voxel
	loss = torch.sum(weighted_err) / torch.mean(wgt)
	return err, loss

def _loss_fn(_ext, _con, _x, _v):
	_r = _model_fn(_ext, _con, _x)
	#_err = T.sum((_r - _v)**2, dim=0)
	#_loss = T.sum(_err)
	_err, _loss = vox_loss_fn(_r, _v, nu=0.1, delta=.5)
	_loss += fpX(1e-1) * torch.sum(torch.abs(_con.w))
	return _err, _loss

def _training_fn(_ext, _con, _opts, xb, yb):
	for _opt in _opts:
		_opt.zero_grad()
		_err, _loss = _loss_fn(_ext, _con, torch.from_numpy(xb).to(device),
			torch.from_numpy(yb).to(device))
		_loss.backward()
		_opt.step()
	return _err

def _holdout_fn(_ext, _con, xb, yb):
	# print (xb.shape, yb.shape)
	_err,_ = _loss_fn(_ext, _con, torch.from_numpy(xb).to(device),
		torch.from_numpy(yb).to(device))
	return _err

def _pred_fn(_ext, _con, xb):
	return _model_fn(_ext, _con, torch.from_numpy(xb).to(device))


# =============================================================================
# Model optimizers
# =============================================================================
# Backbone model optimizer
if args.train_prefilter == 0:
	optimizer_backbone = torch.optim.Adam([
		{'params': backbone_model.enc.parameters()},
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay)
elif args.train_prefilter == 1:
	optimizer_backbone = torch.optim.Adam([
		{'params': backbone_model.pre.parameters()},
		{'params': backbone_model.enc.parameters()},
		], lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
		weight_decay=args.weight_decay)

# FWRF model optimizer
optimizer_fwrf = torch.optim.Adam([{'params': fwrf_model.parameters()}],
	lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

# Combine optimizers
if args.trained == 0:
	# If "args.trained == 0" do NOT train the backbone model layers
	optimizer = [optimizer_fwrf]
elif args.trained == 1:
	# If "args.trained == 1" train all model layers
	optimizer = [optimizer_backbone, optimizer_fwrf]


# =============================================================================
# Model training
# =============================================================================
# TensorBoard
writer = SummaryWriter(tensorboard_parent)

# Model training
best_params, final_params, hold_hist, trn_hist, best_epoch, \
	best_joint_cc_score = learn_params_(
	writer,
	_training_fn,
	_holdout_fn,
	_pred_fn,
	backbone_model,
	fwrf_model,
	optimizer,
	trn_images,
	trn_voxel_data,
	trn_stim_ordering,
	hld_voxel_data,
	hld_stim_ordering,
	num_epochs=args.epochs,
	batch_size=args.batch_size)


# =============================================================================
# Save the trained model
# =============================================================================
out_dir = os.path.join(model_output_dir, 'weights_'+subj_roi_dir+'.pt')
torch.save({
	'args.': args,
	'best_params': best_params,
	'final_params': final_params,
	'trn_hist': trn_hist,
	'hold_hist': hold_hist,
	'best_epoch': best_epoch,
	'best_joint_cc_score': best_joint_cc_score,
	'stim_mean': trn_images_mean,
	'fmri_metadata': fmri_metadata
	}, out_dir)
