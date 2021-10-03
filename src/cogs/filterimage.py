import PIL.Image as Image
import numpy as np

####Format:####
#R relativeX+128
#G relativeY+128
#B Brightness
#A Alpha

def apply_filterimage(img,fil):
	npimg = np.array(img)
	npfil = np.array(fil,dtype=int)

	out = np.zeros((npfil.shape[0],npfil.shape[1],npimg.shape[2]))

	for y in range(npfil.shape[0]):
		for x in range(npfil.shape[1]):
			samplecoordinates = npfil[y,x,:2]-128
			brightness = npfil[y,x,2]
			alpha = npfil[y,x,3]
			if alpha>0:
				color = npimg[((samplecoordinates[1]+y)%npimg.shape[1]),((samplecoordinates[0]+x)%npimg.shape[0])]
				color[:3]=(color[:3]*(brightness/255)).astype(int)
				color[3]=int(color[3]*(alpha/255))
			else:
				color=(0,0,0,0)
			out[y,x]=color

	img = Image.fromarray(out.astype("uint8"))

	return img
