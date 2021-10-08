import PIL.Image as Image
import numpy as np

####Format:####
#R relativeX+128
#G relativeY+128
#B Brightness
#A Alpha

def apply_filterimage(img: Image.Image,fil: Image.Image,absolute: bool):
	npimg = np.array(img)
	npfil = np.array(fil,dtype=int)

	out = np.zeros((npfil.shape[0],npfil.shape[1],npimg.shape[2]))

	samplecoordinates = (npfil[:,:,:2]-128)
	if not absolute:
		samplecoordinates[:,:,0]+=np.arange(npfil.shape[0])
		samplecoordinates[:,:,1]=(samplecoordinates[:,:,1].T+np.arange(npfil.shape[1])).T
	samplecoordinates%=npimg.shape[:2]
	brightness = npfil[:,:,2]
	alpha = npfil[:,:,3]
	out[alpha>0] = npimg[samplecoordinates[:,:,1],samplecoordinates[:,:,0]][alpha>0]
	out[:,:,:3]*=(brightness/255).reshape(brightness.shape+(1,)).repeat(3,2)
	out[:,:,3]*=(alpha/255)

	img = Image.fromarray(out.astype("uint8"))

	return img
