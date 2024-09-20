"""For each pairwise area combination (V1 vs. V2, V1 vs. V3, V1 vs. V4,
V2 vs. V3, V2 vs. V4, V3 vs. V4) multivariate RNC uses genetic optimization and
representational similarity analysis (RSA) to search, across stimuli from the
chosen imageset, for a batch of images that aligns (i.e., images leading to a
high RSA correlation score) and disentangles (i.e., images leading to a low
absolute RSA correlation score) the synthetic multivariate fMRI responses
for the two ROIs being compared, thus highlighting shared and unique
representational content, respectively.

This code is available at:
https://github.com/gifale95/RNC/blob/main/04_multivariate_rnc/05_multivariate_rnc.py

Parameters
----------
cv : int
	If '1' univariate RNC leaves the data of one subject out for
	cross-validation, if '0' univariate RNC uses the data of all subjects.
cv_subject : int
	If 'cv==0' the left-out subject during cross-validation, out of all 8 (NSD)
	subjects.
roi_pair : int
	Integer indicating the chosen pairwise ROI combination on which to perform
	multivariate RNC. Possible values are '0' (V1-V2), '1' (V1-V3), '2'
	(V1-hV4), '3' (V2-V3), '4' (V2-hV4), '5' (V3-hV4).
imageset : str
	Used image set. Possible choices are 'nsd', 'imagenet_val', 'things'.
control_condition : str
	Whether to 'align' or 'disentangle' the multivariate fMRI responses for the
	two ROIs being compared.
generations : int
	Number of genetic optimization generations.
n_batches : int
	Initial amount of image batches at each genetic optimization generation.
n_images_per_batch : int
	Amount of images per image batch, that is, of the controlling images.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm

from utils import load_rsms
from utils import create_batches
from utils import mutate
from utils import evaluate
from utils import select

parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--cv_subject', type=int, default=1)
parser.add_argument('--roi_pair', type=int, default=0)
parser.add_argument('--imageset', type=str, default='nsd')
parser.add_argument('--control_condition', type=str, default='disentangle')
parser.add_argument('--generations', type=int, default=2000)
parser.add_argument('--n_batches', type=int, default=200)
parser.add_argument('--n_images_per_batch', type=int, default=50)
parser.add_argument('--project_dir', default='../relational_neural_control/', type=str)
args = parser.parse_args()

print('>>> Multivariate RNC <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# ROI pair combinations
# =============================================================================
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
# Load the pre-computed synthetic fMRI RSMs
# =============================================================================
if args.cv == 0:
	# If not cross-validating, load and use the RSMs averaged across all
	# subjects.
	rsm_1, rsm_2 = load_rsms(args)

elif args.cv == 1:
	# If cross-validating, load and use the RSMs of the N-1 subjects (i.e., the
	# 7 remaining subjects beyond the 'cv_subject').
	rsm_1, rsm_2 = load_rsms(args, args.cv_subject, 'train')


# =============================================================================
# Use genetic optimization to find aligning and disentangling images
# =============================================================================
# Randomly create the first generation of image batches
image_batches = create_batches(args.n_batches, args.n_images_per_batch,
	len(rsm_1))
image_batches_scores = np.zeros((args.generations))
best_generation_image_batches = np.zeros((args.generations,
	args.n_images_per_batch), dtype=int)

for g in tqdm(range(args.generations)):
	# At the beginning of each genetic optimization generation the image batches
	# are augmented following exploitation and exploration. Exploitation
	# involves creating five mutated versions for each of the image batches,
	# where in each version a different amount of batch images is randomly
	# replaced with other images from the ROIs RSMs (while ensuring that no
	# image is repeated within the same batch). Exploration involves creating
	# new random batches.
	# Augment the image batches via mutations (exploitation)
	mutated_image_batches = mutate(image_batches, len(rsm_1))
	image_batches = np.append(image_batches, mutated_image_batches, 0)
	# Augment the image batches with new random batches (exploration)
	new_image_batches = create_batches(len(image_batches),
		args.n_images_per_batch, len(rsm_1))
	image_batches = np.append(image_batches, new_image_batches, 0)
	image_batches.sort(1)

	# Perform RSA between the two ROIs (i.e., correlate the RSMs of the two
	# ROIs) using only the RSM entries corresponding to the images from the
	# used image batches, resulting in one RSA correlation score per batch.
	scores = evaluate(image_batches, rsm_1, rsm_2)

	# To align the two ROIs, keep the N image batches (where N is defined by the
	# 'n_batches' variable) with highest correlation scores (i.e., containing
	# images most similarly represented by the two ROIs), whereas to disentangle
	# them keep the N image batches with lowest absolute correlation scores
	# (i.e., containing images most differently represented by the two ROIs).
	# These image batches are then passed to the next genetic otpimization
	# generation, where the same steps are repeated.
	image_batches, scores = select(args.control_condition, image_batches,
		scores, args.n_batches)
	# Store the best image batch of each generation, along with its score
	image_batches_scores[g] = scores[0]
	best_generation_image_batches[g] = image_batches[0]


# =============================================================================
# Save the results
# =============================================================================
results = {
	'roi_1': roi_1,
	'roi_2': roi_2,
	'image_batches_scores': image_batches_scores,
	'best_generation_image_batches': best_generation_image_batches
	}

save_dir = os.path.join(args.project_dir, 'multivariate_rnc',
	'best_image_batches', 'cv-'+format(args.cv), 'imageset-'+args.imageset,
	roi_1+'-'+roi_2, 'control_condition-'+args.control_condition)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if args.cv == 0:
	file_name = 'best_image_batches'
elif args.cv == 1:
	file_name = 'best_image_batches_subject-' + format(args.cv_subject, '02')

np.save(os.path.join(save_dir, file_name), results)

