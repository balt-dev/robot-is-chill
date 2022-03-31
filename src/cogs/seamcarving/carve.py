import numpy as np
import math
from random import random

def do_carving(img):
	def get_diff(coord1: tuple, coord2: tuple):
		r1,g1,b1,a1=img[coord1[0],coord1[1]]
		r2,g2,b2,a2=img[coord2[0],coord2[1]]
		return math.sqrt(((a1-a2)**2)+((r1-r2)**2)+((g1-g2)**2)+((b1-b2)**2))
	lscore = None
	lpts = []
	for x in range(img.shape[1]):
		score = 0
		searchx = x
		pixels_to_remove = []
		for y in range(img.shape[0]):
			deltal = deltah = deltar = 2147483648
			pixels_to_remove.append([y,searchx])
			if y != img.shape[0]-1:
				deltah = get_diff((y,searchx),(y+1,searchx))
				if searchx != 0:
					deltal = get_diff((y,searchx),(y+1,searchx-1))
				if searchx != img.shape[1]-1:
					deltar = get_diff((y,searchx),(y+1,searchx+1))
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
	img = np.ndarray.tolist(img)
	for p in lpts:
		img[p[0]].pop(p[1])
	return np.array(img,dtype=np.uint8)
		
def seam_carve(img: np.ndarray, size: tuple):
	'''Content-aware scales an image to a given box.'''
	assert size[0] <= img.shape[0] and size[1] <= img.shape[1], "Bounding box too big!"
	for _ in range(int(img.shape[1]-size[1])):
		img = do_carving(img)
	img = np.swapaxes(img,0,1)
	for _ in range(int(img.shape[1]-size[0])):
		img = do_carving(img)
	img = np.swapaxes(img,0,1)
	return img
