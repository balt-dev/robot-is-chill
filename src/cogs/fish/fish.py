# import imageio
import numpy as np
from math import sqrt
import sys
# import argparse
import os


def get_fish_xn_yn(source_x, source_y, radius, distortion):
	"""
	Get normalized x, y pixel coordinates from the original image and return normalized 
	x, y pixel coordinates in the destination fished image.
	:param distortion: Amount in which to move pixels from/to center.
	As distortion grows, pixels will be moved further from the center, and vice versa.
	"""
	try:
		return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))
	except ZeroDivisionError:
		return np.Infinity,np.Infinity


def fish(img, distortion_coefficient):
	"""
	:type img: numpy.ndarray
	:param distortion_coefficient: The amount of distortion to apply.
	:return: numpy.ndarray - the image with applied effect.
	"""

	# If input image is only BW or RGB convert it to RGBA
	# So that output 'frame' can be transparent.
	w, h = img.shape[0], img.shape[1]
	if len(img.shape) == 2:
		# Duplicate the one BW channel twice to create Black and White
		# RGB image (For each pixel, the 3 channels have the same value)
		bw_channel = np.copy(img)
		img = np.dstack((img, bw_channel))
		img = np.dstack((img, bw_channel))
	if len(img.shape) == 3 and img.shape[2] == 3:
		print("RGB to RGBA")
		img = np.dstack((img, np.full((w, h), 255)))

	# prepare array for dst image
	dstimg = np.zeros_like(img)

	# floats for calcultions
	w, h = float(w), float(h)

	# easier calcultion if we traverse x, y in dst image
	for x in range(len(dstimg)):
		for y in range(len(dstimg[x])):

			# normalize x and y to be in interval of [-1, 1]
			xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

			# get xn and yn distance from normalized center
			rd = sqrt(xnd**2 + ynd**2)

			# new normalized pixel coordinates
			xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

			# convert the normalized distorted xdn and ydn back to image pixels
			try:
				xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

				# if new pixel is in bounds copy from source pixel to destination pixel
				if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
					dstimg[x][y] = img[xu][yu]
			except OverflowError:
				pass

	return dstimg.astype(np.uint8)
