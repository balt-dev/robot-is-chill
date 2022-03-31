import numpy as np
import math
from random import random

def carve_once(img):
	def get_diff(coord1: tuple, coord2: tuple):
		return math.dist(img[coord1[0],coord1[1]],img[coord2[0],coord2[1]])
	lscore = None
	lpts = None
	for x in range(img.shape[1]):
		score = 0
		searchx = x
		pixels_to_remove = np.zeros(img.shape[:2],dtype=bool)
		for y in range(img.shape[0]):
			deltal = deltah = deltar = 2147483648
			pixels_to_remove[y,searchx] == True
			if y != img.shape[0]-1:
				deltah = get_diff((y,searchx),(y+1,searchx))
				deltal = get_diff((y,searchx),(y+1,max(searchx-1,0)))
				deltar = get_diff((y,searchx),(y+1,min(searchx+1,img.shape[1]-1)))
			if deltal < deltah and deltal < deltar:
				searchx -= 1
				score += deltal
			elif deltar < deltah and deltar < deltal:
				searchx += 1
				score += deltar
			else:
				score += deltah
			if lscore != None and score > lscore:
				break
		if lscore == None or score < lscore:
			lscore = score
			lpts = pixels_to_remove
	img = np.delete(img,lpts,1)
		
def seam_carve(img: np.ndarray, size: tuple):
	'''Content-aware scales an image to a given box.'''
	assert size[0] <= img.shape[0] and size[1] <= img.shape[1], "Bounding box too big!"
	for _ in range(int(img.shape[1]-size[1])):
		img = carve_once(img)
	img = np.swapaxes(img,0,1)
	for _ in range(int(img.shape[1]-size[0])):
		img = carve_once(img)
	img = np.swapaxes(img,0,1)
	return img
