# Mom, can I have fluid simulation?
# No, we have fluid simulation at home
# Fluid simulation at home:

import cv2
import numpy as np
from PIL import Image, ImageDraw

def sorter(x):
	# print(x.shape)
	out = np.array(sorted(x,key=lambda y: int(y[3]!=0)))#lambda y: 0 if (y[3] == 0) else 1)
	# print(out.shape)
	return out

def flatten_to_color_array(x):
	return x.reshape(-1,4)

def get_colors(x):
	f=flatten_to_color_array(x)
	return np.unique(f[f[:,3]!=0],axis=0)

def get_colors_unsorted(x):
	f=flatten_to_color_array(x)
	arr = f[f[:,3]!=0]
	indexes = np.unique(arr, return_index=True, axis=0)[1]
	return np.array([arr[index] for index in sorted(indexes)])

def count_instances_of_color(x, color):
	f=flatten_to_color_array(x)
	return np.count_nonzero((f[:]==color).all(1))

def remove_instances_of_color(x, color):
	f=flatten_to_color_array(x)
	f[(f[:]==color).all(1)]=[0,0,0,0]
	return f.reshape(x.shape)

def colorflood(x, color, times):
	f=flatten_to_color_array(x)
	f[np.argwhere(f[:,3]==0).flatten()[-times:]]=color
	return f.reshape(x.shape)

def makecircle(size, radius, color):
	img = Image.new("RGBA", size)
	center = (size[0]/2,size[1]/2)
	bbox = (center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius)
	draw = ImageDraw.Draw(img)
	draw.ellipse(bbox, fill=tuple(color))
	del draw
	return img

def liquify(img):
	#Count colors
	most_used_color = [0,0,0,0]
	most_used_color_count = 0
	total_color_count = 0
	for color in get_colors(img):
		instances = count_instances_of_color(img, color)
		if instances>most_used_color_count:
			most_used_color_count = instances
			most_used_color = color
		total_color_count+=instances

	#Remove most used color
	img = remove_instances_of_color(img,most_used_color)

	#Collapse
	img = img.swapaxes(0,1)
	for i in range(len(img)):
		img[i] = sorter(img[i])
	img = img.swapaxes(0,1)

	#Flood - where the magic happens
	img = colorflood(img, most_used_color, most_used_color_count)

	return img

def planet(img):
	#Count colors
	most_used_color = [0,0,0,0]
	most_used_color_count = 0
	total_color_count = 0
	colors = get_colors(img)
	if len(colors)>1:
		for color in colors:
			instances = count_instances_of_color(img, color)
			if instances>most_used_color_count:
				most_used_color_count = instances
				most_used_color = color
			total_color_count+=instances

		#Remove most used color
		img = remove_instances_of_color(img,most_used_color)
	else:
		most_used_color = colors[0]
		total_color_count = most_used_color_count = count_instances_of_color(img, most_used_color)
		radius = pow(most_used_color_count/np.pi,0.5) #sqrt(area/π) = radius
		return makecircle((img.shape[1], img.shape[0]), radius, most_used_color)

	#Center
	for axis in range(2):
		nonempty = np.nonzero(np.any(img, axis=1-axis))[0]
		first, last = nonempty.min(), nonempty.max()
		shift = (img.shape[axis] - first - last)//2
		img = np.roll(img, shift, axis=axis)

	#Create circle of volume most_used_color_count with color most_used_color
	radius = pow(most_used_color_count/np.pi,0.5) #sqrt(area/π) = radius
	circle = makecircle((img.shape[1], img.shape[0]),radius, most_used_color)
	
	#Blend
	pimg = Image.fromarray(img)
	pimg = Image.alpha_composite(circle, pimg)

	return pimg
