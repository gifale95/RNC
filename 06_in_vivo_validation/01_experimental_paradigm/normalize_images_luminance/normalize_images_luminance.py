"""Normalize each image’s mean luminance (i.e., its luminance across all pixels)
to the luminance of the stimuli presentation screen background (a uniform gray
screen with an RGB value of [127 127 127]).

https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color

This code is available at:
https://github.com/gifale95/RNC/05_multivariate_rnc_retinotopy/06_in_vivo_validation/01_experimental_paradigm/normalize_images_luminance/normalize_images_luminance.py

"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance


# =============================================================================
# I/O directories
# =============================================================================
input_img_dir = '../path_to_input_image_folder'
output_img_dir = '../path_to_output_image_folder'
reference_img_dir = '../background.png' # https://github.com/gifale95/RNC/05_multivariate_rnc_retinotopy/06_in_vivo_validation/01_experimental_paradigm/normalize_images_luminance/background.png


# =============================================================================
# Get the reference image luminance
# =============================================================================
# Import the image
reference_img = Image.open(reference_img_dir)
reference_img_np = np.asarray(reference_img)
luminance_ref = np.zeros((reference_img_np.shape[0], reference_img_np.shape[1]))

# Convert all sRGB 8 bit integer values to decimal 0.0-1.0
reference_img_np = reference_img_np / 255

for h in range(reference_img_np.shape[0]):
	for w in range(reference_img_np.shape[1]):
		chan_stats = np.zeros(reference_img_np.shape[2])
		for c in range(reference_img_np.shape[2]):

			# Convert a gamma encoded RGB to a linear value
			if reference_img_np[h,w,c] <= 0.04045:
				chan_stats[c] = reference_img_np[h,w,c] / 12.92
			else:
				chan_stats[c] = np.power(((
					reference_img_np[h,w,c] + 0.055) / 1.055), 2.4)

		# To find luminance apply the standard coefficients for sRGB
		luminance_ref[h,w] = (0.2126 * chan_stats[0] + \
			0.7152 * chan_stats[1] + 0.0722 * chan_stats[2])

# Average the reference luminance across voxels
luminance_ref = np.mean(luminance_ref)


# =============================================================================
# Normalize the images luminance to the reference image luminance
# =============================================================================
image_files = os.listdir(input_img_dir)
image_files.sort()

images_luminance_old = []
images_luminance_new = []

for img in tqdm(image_files, leave=False):

	# Compute the (voxel-average) luminance of the original image
	# Import the image
	image = np.asarray(Image.open(os.path.join(input_img_dir, img)))
	# Convert all sRGB 8 bit integer values to decimal 0.0-1.0
	image = image / 255
	# Compute the luminance
	luminance = np.zeros((image.shape[0], image.shape[1]))
	for h in range(image.shape[0]):
		for w in range(image.shape[1]):
			chan_stats = np.zeros(image.shape[2])
			for c in range(image.shape[2]):
				# Convert a gamma encoded RGB to a linear value
				if image[h,w,c] <= 0.04045:
					chan_stats[c] = image[h,w,c] / 12.92
				else:
					chan_stats[c] = np.power(((
						image[h,w,c] + 0.055) / 1.055), 2.4)
			# To find luminance apply the standard coefficients for sRGB
			luminance[h,w] = (0.2126 * chan_stats[0] + \
				0.7152 * chan_stats[1] + 0.0722 * chan_stats[2])
	mean_luminance = np.mean(luminance)
	images_luminance_old.append(mean_luminance)

	# Change the image luminance until convergence to the reference image
	# luminance
	luminance_log = []
	step = 0.2
	factor = 1
	error_threshold = 0.01 # Error < 0.1%
	error_previous = 0
	while abs(luminance_ref - mean_luminance) >= error_threshold:
		# Add a step to the brightness scaling factor in case the algorithm
		# didn't converge to the luminance of the reference image
		error = luminance_ref - mean_luminance
		# Decrease the step if the new image luminance overshoots the reference
		# luminance value
		if (error * error_previous) < 0:
			step /= 2
		error_previous = error
		# Decide whether to increase or decrease the brightness factor
		if error < 0:
			factor -= step
		else:
			factor += step
		# Normalize the image luminance by the scaling factor
		# Load the image and change the brightness
		image = Image.open(os.path.join(input_img_dir, img)).convert("RGB")
		img_enhancer = ImageEnhance.Brightness(image)
		image = img_enhancer.enhance(factor)
		# Convert the image to numpy format
		image = np.asarray(image)
		# Convert all sRGB 8 bit integer values to decimal 0.0-1.0
		image = image / 255
		# Reshape the image
		image_shape = image.shape
		image = np.reshape(image, (-1,image.shape[2]))
		# Compute the luminance
		luminance = np.zeros(image.shape[0])
		for p in range(image.shape[0]):
			chan_stats = np.zeros(image.shape[1])
			for c in range(image.shape[1]):
				# Convert a gamma encoded RGB to a linear value
				if image[p,c] <= 0.04045:
					chan_stats[c] = image[p,c] / 12.92
				else:
					chan_stats[c] = np.power(((
						image[p,c] + 0.055) / 1.055), 2.4)
			# To find luminance apply the standard coefficients for sRGB
			luminance[p] = (0.2126 * chan_stats[0] + \
				0.7152 * chan_stats[1] + 0.0722 * chan_stats[2])
		mean_luminance = np.mean(luminance)
		luminance_log.append(mean_luminance)
	# Save the new image
	image = np.reshape(image, image_shape)
	image = Image.fromarray(np.uint8(image*255))
	image.save(os.path.join(output_img_dir, img))
	images_luminance_new.append(mean_luminance)

