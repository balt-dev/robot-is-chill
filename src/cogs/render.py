from __future__ import annotations

from discord.ext import commands
import copy
import math
import os
import itertools
import random
import zipfile
import collections
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO
import cv2
import time
import config
import asyncio
from functools import reduce

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageSequence
from src.tile import FullTile, ReadyTile

from .. import constants, errors
from ..utils import cached_open

if TYPE_CHECKING:
	from ...ROBOT import Bot

import src.cogs.fish as fish
import src.cogs.filterimage as filterimage
import src.cogs.liquify as liquify
import requests

def rgb_to_hsv(rgb):
	# Translated from source of colorsys.rgb_to_hsv
	# r,g,b should be a numpy arrays with values between 0 and 255
	# rgb_to_hsv returns an array of floats between 0.0 and 1.0.
	rgb = rgb.astype('float')
	hsv = np.zeros_like(rgb)
	# in case an RGBA array was passed, just copy the A channel
	hsv[..., 3:] = rgb[..., 3:]
	r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
	maxc = np.max(rgb[..., :3], axis=-1)
	minc = np.min(rgb[..., :3], axis=-1)
	hsv[..., 2] = maxc
	mask = maxc != minc
	hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
	rc = np.zeros_like(r)
	gc = np.zeros_like(g)
	bc = np.zeros_like(b)
	rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
	gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
	bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
	hsv[..., 0] = np.select(
		[r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
	hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
	return hsv

def hsv_to_rgb(hsv):
	# Translated from source of colorsys.hsv_to_rgb
	# h,s should be a numpy arrays with values between 0.0 and 1.0
	# v should be a numpy array with values between 0.0 and 255.0
	# hsv_to_rgb returns an array of uints between 0 and 255.
	rgb = np.empty_like(hsv)
	rgb[..., 3:] = hsv[..., 3:]
	h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
	i = (h * 6.0).astype('uint8')
	f = (h * 6.0) - i
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))
	i = i % 6
	conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
	rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
	rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
	rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
	return rgb.astype('uint8')

def shift_hue(arr,hueshift):
	hsv=rgb_to_hsv(arr)
	hsv[...,0]= (hsv[...,0] + (hueshift/360)) % 1
	rgb=hsv_to_rgb(hsv)
	return rgb

def lock(t,arr,lock):
	hsv=rgb_to_hsv(arr)
	hsv[...,t]= lock
	rgb=hsv_to_rgb(hsv)
	return rgb

def grayscale(arr,influence):
	arr = arr.astype(np.float64)
	arr[:,:,0:3] = (((arr[:,:,0:3].sum(2)/3).repeat(3).reshape(arr.shape[:2]+(3,))) * influence) + (arr[:,:,0:3]*(1-influence))
	return arr.astype(np.uint8)

def alpha_paste(img1,img2,coords,format):
	imgtemp = Image.new('RGBA',img1.size,(0,0,0,0))
	imgtemp.paste(
		img2,
		coords
	)
	return Image.alpha_composite(img1, imgtemp)

class Renderer:
	'''This class exposes various image rendering methods.
	Some of them require metadata from the bot to function properly.
	'''
	def __init__(self, bot: Bot) -> None:
		self.bot = bot

	def recolor(self, sprite: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
		'''Apply rgb color multiplication (0-255)'''
		r,g,b = rgb
		rc,gc,bc,ac = sprite.split()
		rc = rc.point(lambda i: int(i * (r/255)))
		gc = gc.point(lambda i: int(i * (g/255)))
		bc = bc.point(lambda i: int(i * (b/255)))
		
		return Image.merge('RGBA', (rc,gc,bc,ac))
				
	async def render(
		self,
		grid: list[list[list[ReadyTile]]],
		*,
		before_image: Image = None,
		palette: str = "default",
		images: list[str] | None = None,
		image_source: str = constants.BABA_WORLD,
		out: str | BinaryIO = "target/renders/render.gif",
		background: tuple[int, int] | str | None = None,
		upscale: int = 1,
		extra_out: str | BinaryIO | None = None,
		extra_name: str | None = None,
		frames: list[int] = [1,2,3],
		speed: int = 200,
		gridol: tuple[int] = None,
		scaleddef: float = 1,
		crop: tuple[int,int,int,int] | None = None,
		pad: tuple[int,int,int,int] | None = (0,0,0,0),
		format: str = 'gif'
	):
		'''Takes a list of tile objects and generates a gif with the associated sprites.

		`out` is a file path or buffer. Renders will be saved there, otherwise to `target/renders/render.gif`.

		`palette` is the name of the color palette to refer to when rendering.

		`images` is a list of background image filenames. Each image is retrieved from `data/images/{image_source}/image`.

		`background` is a palette index. If given, the image background color is set to that color, otherwise transparent. Background images overwrite this.
		'''
		palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")
		durations = []
		dur_check = []
		sprite_cache: dict[str, Image.Image] = {}
		imgs = []
		times = []
		width = len(grid[0][0])
		height = len(grid[0])
		img_width_raw = int(width * (constants.DEFAULT_SPRITE_SIZE*scaleddef))
		img_height_raw = int(height * (constants.DEFAULT_SPRITE_SIZE*scaleddef))
		new_frames = []
		for f in frames:
			if len(durations) == 0 or dur_check[-1] != f: #if the frame before is the same as the current frame
				durations.append(speed)
				new_frames.append(f)
			else:
				durations[-1] += speed
			dur_check.append(f)
		frames = new_frames
		def wl(a):
			try:
				return int(a.frames[0].width * scaleddef)
			except:
				return 0
		def hl(a):
			try:
				return int(a.frames[0].height * scaleddef)
			except:
				return 0
		padding = max(np.amax(np.array([[[hl(c) for c in b] for b in a] for a in grid],dtype=object)),np.amax(np.array([[[wl(c) for c in b] for b in a] for a in grid],dtype=object)))
		padding = padding[0] if type(padding) == list else padding
		img_width = int(img_width_raw + (2 * padding) + pad[0] + pad[2])
		img_height = int(img_height_raw + (2 * padding) + pad[1] + pad[3])
		i = 0
		for l, frame in enumerate(frames):
			if images and image_source is not None:
				img = Image.new("RGBA", (img_width, img_height))
				# for loop in case multiple background images are used (i.e. baba's world map)
				for image in images:
					overlap = Image.open(f"data/images/{image_source}/{image}_{frame}.png")
					img.paste(overlap, (padding, padding), mask=overlap)
			# bg color
			elif type(background) == tuple:
				palette_color = palette_img.getpixel(background)
				img = Image.new("RGBA", (img_width, img_height), color=palette_color)
			elif type(background) == str:
				img = Image.new("RGBA", (img_width, img_height), color=tuple([int(a+b,16) for a,b in np.reshape(list(background),(3,2))]))
			# neither
			else:
				img = Image.new("RGBA", (img_width, img_height), color=(0,0,0,0))
			imgs.append(img)
		# keeping track of the amount of padding we can slice off
		pad_r=pad_u=pad_l=pad_d=0
		for layer in grid:
			for y, row in enumerate(layer):
				for x, tile in enumerate(row):
					t = time.perf_counter()
					if tile.frames is None:
						continue
					tframes = []
					for f in frames:
						tframes.append(tile.frames[f-1])
					for frame, sprite in enumerate(tframes[:len(frames)]):
						x_offset = int((sprite.width - (constants.DEFAULT_SPRITE_SIZE*scaleddef)) / 2 )
						y_offset = int((sprite.height - (constants.DEFAULT_SPRITE_SIZE*scaleddef)) / 2 )
						x_offset_disp = int(((sprite.width - (constants.DEFAULT_SPRITE_SIZE*scaleddef)) / 2 ) + tile.displace[0]*scaleddef)
						y_offset_disp = int(((sprite.height - (constants.DEFAULT_SPRITE_SIZE*scaleddef)) / 2 ) + tile.displace[1]*scaleddef)
						if x == 0:
							pad_l = max(pad_l, x_offset)
						if x == width - 1:
							pad_r = max(pad_r, x_offset)
						if y == 0:
							pad_u = max(pad_u, y_offset)
						if y == height - 1:
							pad_d = max(pad_d, y_offset)
						alpha = sprite.getchannel("A")
						x_offset_disp -= pad[0]
						y_offset_disp -= pad[1]
						if tile.mask_alpha:
							alpha = ImageChops.invert(alpha)
							if background is not None:
								palette_color = palette_img.getpixel(background)
							else:
								palette_color = (0,0,0,0)
							sprite = Image.new("RGBA", (sprite.width, sprite.height), color=palette_color)
							imgs[frame].paste(
								sprite,
								(
									int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
									int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
								),
								alpha
							)
						elif tile.cut_alpha:
							if background is not None:
								imgs[frame].paste(
									Image.new("RGBA", (sprite.width, sprite.height), palette_img.getpixel(background)),
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									alpha
								)
							else:
								imgs[frame].paste(
									Image.new("RGBA", (sprite.width, sprite.height)),
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									alpha
								)
						else:
							if tile.blending == 'add':
								imgtemp=Image.new('RGBA',imgs[frame].size,(0,0,0,0))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								imgs[frame] = Image.fromarray(cv2.add(np.asarray(imgs[frame]),np.asarray(imgtemp)))
							elif tile.blending == 'subtract':
								imgtemp=Image.new('RGBA',imgs[frame].size,(0,0,0,0))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								inmp = np.asarray(imgtemp)
								inmp[:,:,3] = 0
								imgs[frame] = Image.fromarray(cv2.subtract(np.asarray(imgs[frame]),inmp))
							elif tile.blending == 'maximum':
								imgtemp=Image.new('RGBA',imgs[frame].size,(0,0,0,0))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								imgs[frame] = Image.fromarray(cv2.max(np.asarray(imgs[frame]),np.asarray(imgtemp)))
							elif tile.blending and tile.blending.startswith('xor'):
								imgtemp=Image.new('RGBA',imgs[frame].size,(0,0,0,0))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								i1 = np.asarray(imgs[frame])
								i2 = np.asarray(imgtemp)
								rgb = (i1^i2)
								if tile.blending == 'xora':
									rgb[:,:,3] = cv2.max(i1[:,:,3],i2[:,:,3])
								imgs[frame] = Image.fromarray(rgb)
							elif tile.blending == 'minimum':
								imgtemp=Image.new('RGBA',imgs[frame].size,(0,0,0,0))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								imgtempar = np.asarray(imgtemp)
								imgtempar[:,:,3] = np.asarray(imgs[frame])[:,:,3]
								imgs[frame] = Image.fromarray(cv2.min(np.asarray(imgs[frame]),imgtempar))
							elif tile.blending == 'multiply':
								imgtemp = Image.new('RGBA',imgs[frame].size,(255,255,255,255))
								imgtemp.paste(
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									mask=sprite
								)
								imgtempar = np.array(imgtemp,dtype=np.uint8)/255
								imgtempar[:,:,3] = 1.0
								imgs[frame] = Image.fromarray((np.array(imgs[frame])*np.array(imgtempar)).astype(np.uint8))
							else:
								imgs[frame] = alpha_paste(
									imgs[frame],
									sprite,
									(
										int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
										int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
									),
									format
								)
					times.append(tile.delta + (time.perf_counter() - t))
		if before_image:
			bfr=0
			before_durations = []
			for frame in ImageSequence.Iterator(before_image):
				try:
					before_durations.append(frame.info['duration'])
					im = frame.convert('RGBA').resize((frame.width//2,frame.height//2),Image.NEAREST)
					newImage = Image.new('RGBA', (img_width,img_height), (0, 0, 0, 0))
					newImage.paste(im, (padding-pad_l,padding-pad_u),mask=im)
					imgs.insert(bfr,newImage)
					bfr += 1 #i dont wanna use an enumerate on an iterator
				except KeyError:
					pass
			durations = before_durations + durations
		outs = []
		for img in imgs:
			img = np.array(img,dtype=np.uint8)
			if type(gridol) != type(None):
				for col in range(img.shape[0]//(gridol[0]*2)):
					img[col*gridol[0]*2,:,:] = ~img[col*gridol[0]*2,:,:]
					img[col*gridol[0]*2,:,3] = 255
				for row in range(img.shape[1]//(gridol[1]*2)):
					img[:,row*gridol[1]*2,:] = ~img[:,row*gridol[1]*2,:]
					img[:,row*gridol[1]*2,3] = 255
			if format == 'gif':
				img[:, :, :3] = (img[:, :, :3] * (img[:, :, 3] / 255).repeat(3).reshape(img.shape[:2]+(3,))).astype("uint8")
			img = Image.fromarray(img)
			img = img.crop((padding - pad_l, padding - pad_u, img.width - padding + pad_r, img.height - padding + pad_d))
			if crop != None:
				img = img.crop((crop[0],crop[1],img.width-crop[2],img.height-crop[3]))
			if upscale != 1:
				img = img.resize((int(upscale * img.width), int(upscale * img.height)), resample=Image.NEAREST)
			outs.append(img)
		
		self.save_frames(
			outs,
			out,
			durations=durations,
			extra_out=extra_out,
			extra_name=extra_name,
			format=format
		)
		if len(times)==0:
			return 0, 0, 0
		else:
			return sum(times)/len(times), max(times), sum(times)

	async def render_full_tile(self,
		tile: FullTile,
		*,
		position: tuple[int, int],
		palette_img: Image.Image,
		random_animations: bool = False,
		sprite_cache: dict[str, Image.Image],
		gscale: float = 1
	) -> ReadyTile:
		'''woohoo'''

		t = time.perf_counter()
		if tile.empty:
			return ReadyTile(None)
		out = []
		x, y = position
		for frame in range(3):
			wobble = (11 * x + 13 * y + frame) % 3 if random_animations else frame
			for a in tile.filters:
				if a[0] == 'freeze':
					wobble = a[1]-1
			if tile.custom:
				sprite = await self.generate_sprite(
					tile.name,
					style=tile.custom_style or ("noun" if len(tile.name) < 1 else "letter"),
					direction=tile.custom_direction,
					meta_level=tile.meta_level,
					wobble=wobble,
					filters=tile.filters,
					position=(x,y),
					gscale=gscale
				)
			elif isinstance(tile.sprite,np.ndarray):
				sprite = Image.fromarray(tile.sprite[(tile.variant_number*3)+wobble])
				Image.open(sprite)
			else:
				path_fallback = None
				if tile.name == "icon":
					path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}.png"
				elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
					path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}_1.png"
				elif tile.name == "default":
					path = f"data/sprites/{constants.BABA_WORLD}/default_{wobble + 1}.png"
				elif tile.variant_number == -1:
					source, sprite_name = tile.sprite
					path = f"data/sprites/vanilla/error_0_{wobble + 1}.png"
				else:
					source, sprite_name = tile.sprite
					path = f"data/sprites/{source}/{sprite_name}_{tile.variant_number}_{wobble + 1}.png"
				try:
					path_fallback = f"data/sprites/{source}/{sprite_name}_{tile.variant_fallback}_{wobble + 1}.png"
				except:
					path_fallback = None
				try:
					sprite = cached_open(path, cache=sprite_cache, fn=Image.open).convert("RGBA")
				except FileNotFoundError:
					if path_fallback is not None:
						sprite = cached_open(path_fallback, cache=sprite_cache, fn=Image.open).convert("RGBA")
					else:
						assert 0, f'The tile `{tile.name}` was found, but the files don\'t exist for it.'
				sprite = sprite.resize((int(sprite.width*gscale),int(sprite.height*gscale)),Image.NEAREST)
				sprite = await self.apply_options_name(
					tile.name,
					sprite,
					style=tile.custom_style,
					direction=tile.custom_direction,
					meta_level=tile.meta_level,
					wobble=wobble,
					filters=tile.filters,
					gscale=gscale
				)
			# Color augmentation
			if tile.overlay == "":
				if tile.palette=="":
					rgb = tile.color_rgb if tile.color_rgb is not None else palette_img.getpixel(tile.color_index)
				else:
					rgb = tile.color_rgb if tile.color_rgb is not None else Image.open(f"data/palettes/{tile.palette}.png").convert("RGB").getpixel(tile.color_index)
				sprite = self.recolor(sprite, rgb)
			else:
				try:
					overlay = Image.open(f"data/overlays/{tile.overlay}.png").convert("RGBA")
					if overlay.width < sprite.width or overlay.height < sprite.height:
						width = math.ceil(sprite.width/overlay.width)
						height = math.ceil(sprite.height/overlay.height)
						rgb = np.tile(np.array(overlay),(height,width,1))/255
					else:
						rgb = np.array(overlay)/255
				except FileNotFoundError:
					raise errors.OverlayNotFound(tile.overlay)
				ovsprite = np.array(sprite).astype("float64")
				ovsprite*=rgb[:ovsprite.shape[0],:ovsprite.shape[1]]
				ovsprite=(ovsprite).astype("uint8")
				sprite = Image.fromarray(ovsprite)
			if tile.negative:
				sprite = Image.fromarray(np.dstack((~np.array(sprite)[:,:,:3],np.array(sprite)[:,:,3])))
			if tile.hueshift != 0.0:
				sprite = Image.fromarray(shift_hue(np.array(sprite,dtype="uint8"),tile.hueshift))
			if tile.brightness != 1:
				bsprite = np.array(sprite,dtype="float64")
				bsprite*=(tile.brightness,tile.brightness,tile.brightness,1)
				bsprite[bsprite>255]=255
				bsprite[bsprite<0]=0
				sprite = Image.fromarray(bsprite.astype("uint8"))
			if tile.grayscale != 0:
				sprite = Image.fromarray(grayscale(np.array(sprite),tile.grayscale))
			if tile.filterimage != "":
				url=tile.filterimage
				absolute = False
				if url.startswith("abs"):
					url=url[3:]
					absolute = True
				if url.startswith("db!"):
					url=url[3:]
					command="SELECT url FROM filterimages WHERE name == ?;"
					args=(url,)
					async with self.bot.db.conn.cursor() as cursor:
						await cursor.execute(command,args)
						results=await cursor.fetchone()
						if results==None:
							raise requests.exceptions.ConnectionError
							return
						url=results[0]
				p = requests.get(url, stream=True).raw.read()
				#try:
				ifilterimage = Image.open(BytesIO(p)).convert("RGBA")
				sprite = filterimage.apply_filterimage(sprite,ifilterimage.resize((int(ifilterimage.width*gscale),int(ifilterimage.height*gscale)),Image.NEAREST),absolute)
				#except OSError:
				#    raise AssertionError('Image wasn\'t able to be accessed, or is invalid!')
			if not np.array_equal(tile.channelswap,np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])):
				im_np = np.array(sprite,dtype=float)/255 #making it a float for convenience
				out_np = np.zeros(im_np.shape,dtype=float)
				for i, channel_dst in enumerate(tile.channelswap):
					for j, channel_src in enumerate(channel_dst):
						out_np[:,:,i] += im_np[:,:,j]*channel_src
				out_np = np.array(np.vectorize(lambda n: int(min(max(n,0),1)*255))(out_np),dtype=np.uint8)
				sprite = Image.fromarray(out_np)
			numpysprite = np.array(sprite)
			numpysprite[np.all(numpysprite[:,:,:3]<=(0,0,0),axis=2)&(numpysprite[:,:,3]>1),:3]=8
			sprite = Image.fromarray(numpysprite)
			out.append(sprite)
		if 'land' in [a[0] for a in tile.filters]:
			lowestlist = []
			for f in out:
				h=f.height-int(constants.DEFAULT_SPRITE_SIZE*gscale)
				framelowest = 0
				nf=np.array(f)
				for i,row in enumerate(nf):
					if any(row[:,3]):
						framelowest=i+1+math.ceil(h/2)
				lowestlist.append(framelowest)
			lowestlist.sort()
			tile.displace = (tile.displace[0],(tile.displace[1]+int(lowestlist[0]*(gscale**-1)))-int(out[0].height//gscale))
		f0, f1, f2 = out
		return ReadyTile((f0, f1, f2), tile.cut_alpha, tile.mask_alpha, tile.displace, tile.blending, time.perf_counter()-t)

	async def render_full_tiles(
		self,
		grid: list[list[list[FullTile]]],
		*,
		palette: str = "default",
		random_animations: bool = False,
		gscale: float = 1
	) -> list[list[list[ReadyTile]]]:
		'''Final individual tile processing step'''
		sprite_cache = {}
		palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")

		a = []
		for layer in grid:
			b = []
			for y, row in enumerate(layer):
				c = []
				for x, tile in enumerate(row):
					c.append(
						await self.render_full_tile(
							tile,
							position=(x, y),
							palette_img=palette_img,
							random_animations=random_animations,
							sprite_cache=sprite_cache,
							gscale=gscale
						)
					)
				b.append(c)
			a.append(b)
		return a

	async def generate_sprite(
		self,
		text: str,
		*,
		style: str,
		direction: int | None,
		meta_level: int,
		wobble: int,
		seed: int | None = None,
		filters: list,
		gscale: float,
		position: tuple[int,int]
	) -> Image.Image:
		'''Generates a custom text sprite'''
		text = text[5:]
		raw = text.replace("/", "")
		newline_count = text.count("/")
		assert len(text) <= 64, 'Text has a maximum length of `64` characters.'
		if seed is None:
			seed = int((7+position[0])/(3+position[1])*100000000)
		seed_digits = [(seed >> 8 * i ) | 0b11111111 for i in range(len(raw))]
		# Get mode and split status
		if newline_count > 1:
			raise errors.TooManyLines(text, newline_count)
		elif newline_count >= 1:
			fixed = True
			mode = "small"
			index = text.index("/")
		else:
			fixed = False
			mode = "big"
			index = -1
			if len(raw) >= 4:
				mode = "small"
				index = len(raw) - math.ceil(len(raw) / 2)
	
		if style == "letter":
			if mode == "big":
				mode = "letter"
			else:
				raise errors.BadLetterStyle(text)
		
		if index == 0 or index == len(raw):
			raise errors.LeadingTrailingLineBreaks(text)
		
		width_cache: dict[str, list[int]] = {}
		for c in raw:
			rows = await self.bot.db.conn.fetchall(
				'''
				SELECT char, width FROM letters
				WHERE char == ? AND mode == ?;
				''',
				c, mode
			)

			for row in rows:
				char, width = row
				width_cache.setdefault(char, []).append(width)

		def width_greater_than(c: str, w: int = 0) -> int:
			try:
				return min(width for width in width_cache[c] if width > w)
			except ValueError:
				raise KeyError

		# fetch the minimum possible widths first
		try:
			widths: list[int] = [width_greater_than(c) for c in raw]
		except KeyError as e:
			raise errors.BadCharacter(text, mode, e.args[0])

		max_width = max(sum(widths[:index]),constants.DEFAULT_SPRITE_SIZE)
		def check_or_adjust(widths: list[int], index: int) -> int:
			'''Is the arrangement valid?'''
			if mode == "small":
				if not fixed:
					while sum(widths[:index]) > max_width:
						index -= 1
					while sum(widths[index:]) > max_width:
						index += 1
					return index
			return index
		
		def too_squished(widths: list[int], index: int) -> bool:
			'''Is the arrangement too squished? (bad letter spacing)'''
			if mode == "small":
				top = widths[:index]
				top_gaps = max_width - sum(top)
				bottom = widths[index:]
				bottom_gaps = max_width - sum(bottom)
				return top_gaps < len(top) - 1 or bottom_gaps < len(bottom) - 1
			else:
				gaps = max_width - sum(widths)
				return gaps < len(widths) - 1
		
		# Check if the arrangement is valid with minimum sizes
		# If allowed, shift the index to make the arrangement valid
		index = check_or_adjust(widths, index)

		# Expand widths where possible
		stable = [False for _ in range(len(widths))]
		while not all(stable):
			old_width, i = min((w, i) for i, w in enumerate(widths) if not stable[i])
			try:
				new_width = width_greater_than(raw[i], old_width)
			except KeyError:
				stable[i] = True
				continue
			widths[i] = new_width
			try:
				index = check_or_adjust(widths, index)
			except errors.CustomTextTooLong:
				widths[i] = old_width
				stable[i] = True
			else:
				if too_squished(widths, index):
					# We've shown that a "perfect" width already exists below this
					# So stick to the "perfect" one
					widths[i] = old_width
					stable[i] = True

		# Arrangement is now the widest it can be
		# Kerning: try for 1 pixel between sprites, and rest to the edges
		gaps: list[int] = []
		if mode == "small":
			rows = [widths[:index], widths[index:]]
		else:
			rows = [widths[:]]
		for row in rows:
			space = max_width - sum(row)
			# Extra -1 is here to not give kerning space outside the left/rightmost char
			chars = len(row) - 1
			if space >= chars:
				# left edge
				gaps.append((space - chars) // 2)
				# char gap
				gaps.extend([1] * chars)
				# right edge gap is implied
			else:
				# left edge
				gaps.append(0)
				# as many char gaps as possible, starting from the left
				gaps.extend([1] * space)
				gaps.extend([0] * (chars - space))

		letters: list[Image.Image] = []
		for c, seed_digit, width in zip(raw, seed_digits, widths):
			l_rows = await self.bot.db.conn.fetchall(
				# fstring use safe
				f'''
				SELECT char, width, sprite_{int(wobble)} FROM letters
				WHERE char == ? AND mode == ? AND width == ?
				''',
				c, mode, width
			)
			options = list(l_rows)
			char, width, letter_sprite = options[seed_digit % len(options)]
			buf = BytesIO(letter_sprite)
			letters.append(Image.open(buf))

		sprite = Image.new("L", (max(sum(widths[:index]),constants.DEFAULT_SPRITE_SIZE), constants.DEFAULT_SPRITE_SIZE))
		if mode == "small":
			x = gaps[0]
			y_center = 6
			for i in range(index):
				letter = letters[i]
				y_top = y_center - letter.height // 2
				sprite.paste(letter, (x, y_top), mask=letter)
				x += widths[i]
				if i != index - 1:
					x += gaps[i + 1]
			x = gaps[index]
			y_center = 18
			for i in range(index, len(raw)):
				letter = letters[i]
				y_top = y_center - letter.height // 2
				sprite.paste(letter, (x, y_top), mask=letter)
				x += widths[i]
				if i != len(raw) - 1:
					x += gaps[i + 1]
		else:
			x = gaps[0]
			y_center = 12
			for i in range(len(raw)):
				letter = letters[i]
				y_top = y_center - letter.height // 2
				sprite.paste(letter, (x, y_top), mask=letter)
				x += widths[i]
				if i != len(raw) - 1:
					x += gaps[i + 1]
			
		sprite = Image.merge("RGBA", (sprite, sprite, sprite, sprite))
		sprite = sprite.resize((int(sprite.width*gscale),int(sprite.height*gscale)),Image.NEAREST)
		return self.apply_options(
			sprite,
			original_style="noun",
			original_direction=None,
			direction=direction,
			wobble=wobble,
			filters=filters,
			name="text"+raw,
			meta_level=meta_level,
			style=style
		)

	async def apply_options_name(
		self,
		name: str,
		sprite: Image.Image,
		*,
		direction: int | None,
		meta_level: int,
		wobble: int,
		filters: list,
		gscale: float,
		style: str
	) -> Image.Image:
		'''Takes an image, taking tile data from its name, and applies the given options to it.'''
		tile_data = await self.bot.db.tile(name)
		assert tile_data is not None
		original_style = constants.TEXT_TYPES[tile_data.text_type]
		original_direction = tile_data.text_direction
		try:
			return self.apply_options(
				sprite,
				original_style=original_style,
				original_direction=original_direction,
				direction=direction,
				meta_level=meta_level,
				wobble=wobble,
				filters=filters,
				name=name,
				style=style
			)
		except ValueError as e:
			size = e.args[0]
			raise errors.BadTileProperty(name, size)

	def apply_options(
		self,
		sprite: Image.Image,
		*,
		original_style: str,
		original_direction: int | None,
		direction: int | None,
		meta_level: int,
		wobble: int,
		name: str,
		style: str | None = None,
		filters: list, #using list of tuples now
	):
		'''Takes an image, with or without a plate, and applies the given options to it.'''
		#rocket's handlers are such a better system but i have no idea how they work since i don't really know OOP
		def rotate(li, x):
			return li[-x % len(li):] + li[:-x % len(li)]
		def gradient(head:int,start:float,end:float,startvalue:float,endvalue:float):
			v=(head-start)/(end-start)
			if v<0:
				return startvalue
			elif v>1:
				return endvalue
			else:
				return startvalue+((endvalue-startvalue)*v)
		def scan(spritenumpyscan, props):
			vis,inv,off = props
			for i in range(len(spritenumpyscan)):
				if ((i+off)%(inv+vis))>=vis:
					spritenumpyscan[i]=0
			return spritenumpyscan
		def avg(*args):
			return sum(args)/len(args)
		if (
			(
				original_style != style and
				style != None
			) or (
				style == "property" and
				original_direction != direction
			)
		):
			if original_style == "property":
				# box: position of upper-left coordinate of "inner text" in the larger text tile
				plate, box = self.bot.db.plate(original_direction, wobble)
				plate_alpha = ImageChops.invert(plate.getchannel("A"))
				sprite_alpha = ImageChops.invert(sprite.getchannel("A"))
				alpha = ImageChops.subtract(sprite_alpha, plate_alpha)
				sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
				sprite = sprite.crop((box[0], box[1], constants.DEFAULT_SPRITE_SIZE + box[0], constants.DEFAULT_SPRITE_SIZE + box[1]))
			if style == "property":
				assert (sprite.height <= constants.DEFAULT_SPRITE_SIZE and sprite.width <= constants.DEFAULT_SPRITE_SIZE), f'Properties can\'t be larger than {constants.DEFAULT_SPRITE_SIZE}x{constants.DEFAULT_SPRITE_SIZE}.'
				plate, box = self.bot.db.plate(direction, wobble)
				plate_alpha = plate.getchannel("A")
				sprite_alpha = sprite.getchannel("A")
				sprite_alpha = sprite_alpha.crop(
					(-box[0], -box[0], sprite_alpha.width + box[0], sprite_alpha.height + box[1])
				)
				alpha = ImageChops.subtract(plate_alpha, sprite_alpha)
				sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
		for name, value in filters:
			if name == 'meta_level' and value != 0:
					sprite = self.make_meta(sprite, value)
			elif name == 'threeoo' and value != None:
				img = np.array(sprite,dtype=np.uint8)
				h,w,_ = img.shape
				assert h <= 24 and w <= 24, 'Image too large for 3oo filter!'
				def carve_once(img):
					def get_diff(coord1: tuple, coord2: tuple):
						return math.dist(img[coord1[0],coord1[1]],img[coord2[0],coord2[1]])
					lscore = None
					lpts = None
					for x in range(img.shape[1]):
						score = 0
						searchx = x
						pixels_to_remove = ~np.zeros(img.shape,dtype=bool)
						for y in range(img.shape[0]):
							deltal = deltah = deltar = 2147483648
							pixels_to_remove[y,searchx,:] = False
							if y != img.shape[0]-1:
								deltah = math.dist(img[y,searchx],img[y+1,searchx])
								deltal = math.dist(img[y,searchx],img[y+1,max(searchx-1,0)])
								deltar = math.dist(img[y,searchx],img[y+1,min(searchx+1,img.shape[1]-1)])
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
					return img[lpts].reshape(img.shape[0],img.shape[1]-1,-1)
				
				assert img.shape[0]//value <= img.shape[0] and img.shape[1]//value <= img.shape[1], "Bounding box too big!"
				for _ in range(int(sprite.size[1]-(sprite.size[1]//value))):
					img = carve_once(img)
				img = np.swapaxes(img,0,1)
				for _ in range(int(sprite.size[0]-(sprite.size[0]//value))):
					img = carve_once(img)
				img = np.swapaxes(img,0,1)
				sprite = Image.fromarray(img).resize((w,h),Image.NEAREST)
			elif name == 'normalize':
				sprite = np.array(sprite,dtype=np.uint8)
				minx, miny = sprite.shape[:2]
				maxx, maxy = (0,0)
				for y, row in enumerate(sprite):
					for x, cell in enumerate(row):
						if cell[3] != 0:
							minx = min(x,minx)
							miny = min(y,miny)
							maxx = max(x,maxx)
							maxy = max(y,maxy)
				center = (int(avg(miny,maxy)),int(avg(minx,maxx)))
				absolute_center = [n//2 for n in sprite.shape[:2]]
				displacement = [(a-b)-1 for a,b in zip(absolute_center,center)]
				displacement = [displacement[0] if value[0] else 0, displacement[1] if value[1] else 0]
				sprite = Image.fromarray(np.roll(sprite,displacement[::-1],(1,0)))
			elif name == 'pad' and any(value):
				sprite = Image.fromarray(np.pad(np.array(sprite),((value[1],value[3]),(value[0],value[2]),(0,0))))
			elif name == 'floodfill' and type(value) == float:
				im = np.array(sprite)
				
				im[im[:,:,3]==0]=0 #Optimal
				
				ima = im[:,:,3] #Stores the alpha channel separately
				ima[ima>0]=69 #Sets all nonzero numbers to a number that's neither 0 or 255. In this case, a funny number.
				ima = np.pad(ima,((1,1),(1,1))) #Pads the alpha channel by 1 on each side to prevent using the sprite borders for filling or something probably

				#Balt I think you might be overusing the numpy array constructor
				imf = cv2.floodFill(
					image=ima,
					mask=None,
					seedPoint=(0,0),
					newVal=255
					)[1]
				imf[imf!=255]=0
				imf=255-imf
				
				im[:,:,3] = imf[1:-1,1:-1].astype(np.uint8) #Crops the alpha channel back to the original size and positioning

				brightnessvalue=round(value*255) #This does not need to be in a loop

				im[(im[:,:]==[0,0,0,255]).all(2)] = [brightnessvalue,brightnessvalue,brightnessvalue,255] #Optimal, "somehow this doesn't fuck up anywhere"

				sprite = Image.fromarray(im)
			elif name == 'surround' and type(value) == float:
				#Guess what, it's mostly a big copy-paste of the previous section.
				im = np.array(sprite)
				
				im[im[:,:,3]==0]=0 #Optimal
				
				ima = im[:,:,3] #Stores the alpha channel separately
				ima[ima>0]=69 #Sets all nonzero numbers to a number that's neither 0 or 255. In this case, a funny number.
				ima = np.pad(ima,((1,1),(1,1))) #Pads the alpha channel by 1 on each side to prevent using the sprite borders for filling or something probably

				#Balt I think you might be overusing the numpy array constructor
				imf = cv2.floodFill(
					image=ima,
					mask=None,
					seedPoint=(0,0),
					newVal=255
					)[1]
				imf[imf!=0]=255
				
				im[:,:,3] = imf[1:-1,1:-1].astype(np.uint8) #Crops the alpha channel back to the original size and positioning

				brightnessvalue=round(value*255) #This does not need to be in a loop

				im[(im[:,:]==[0,0,0,255]).all(2)] = [brightnessvalue,brightnessvalue,brightnessvalue,255] #Optimal, "somehow this doesn't fuck up anywhere"

				sprite = Image.fromarray(im)
			elif name == 'colselect' and value != None:
				img = np.array(sprite)

				colors = liquify.get_colors_unsorted(img)
				if len(colors)>1:
					colors = list(sorted(
							colors,
							key=lambda color: liquify.count_instances_of_color(img, color),
							reverse=True
						))
					try:
						selection = np.arange(len(colors))[value]
					except IndexError:
						assert 0, f'The color slice `{value}` is invalid.'
					if type(selection) == np.ndarray:
						selection = selection.flatten().tolist()
					else:
						selection = [selection]
					#Modulo the value field
					positivevalue = [(color%len(colors)) for color in selection]
					#Remove most used color
					for color_index, color in enumerate(colors):
						if color_index not in positivevalue:
							img = liquify.remove_instances_of_color(img,color)

					sprite = Image.fromarray(img) # This is indented because we don't need to convert back if nothing changed
			elif name == 'crop' and any(value):
				cropped = sprite.crop((value[0],value[1],value[0]+value[2],value[1]+value[3]))
				im = Image.new('RGBA',(sprite.width,sprite.height),(0,0,0,0))
				im.paste(cropped,(value[0],value[1]))
				sprite = im
			elif name == 'snip' and any(value):
				im = np.array(sprite,dtype=np.uint8)
				h,w,_ = im.shape
				im = np.pad(im,((0,1),(0,1),(0,0)))
				im[max(0,value[1]):min(value[1]+value[3],h),max(value[0],0):min(value[0]+value[2],w)] = [0,0,0,0]
				sprite = Image.fromarray(im[:-1,:-1])
			elif name == 'mirror':
				value = list(value)
				value[1] = 1 if value[1] else -1
				np_img = np.array(sprite,dtype=np.uint8)
				midpoint = [n//2 for n in np_img.shape[:2]]
				offset = [n%2 for n in np_img.shape[:2]]
				if value[0]:
					try:
						np_img[:,:midpoint[1]+offset[1]:value[1]] = np_img[:,midpoint[1]::value[1]][:,::-1]
					except:
						np_img[:,:midpoint[1]-1+offset[1]:value[1]] = np_img[:,midpoint[1]-1::value[1]][:,::-1]
				else:
					try:
						np_img[:midpoint[0]+offset[0]:value[1],:] = np_img[midpoint[0]::value[1]][::-1]
					except:
						np_img[:midpoint[0]-1+offset[0]:value[1],:] = np_img[midpoint[0]-1::value[1]][::-1]
				sprite = Image.fromarray(np_img)
			elif name == 'scale' and any([x!=1 for x in value]):
				sprite = sprite.resize((math.floor(sprite.width*value[0]),math.floor(sprite.height*value[1])), resample=Image.NEAREST)
			elif name == 'wrap' and any([x!=0 for x in value]):
				sprite = Image.fromarray(np.roll(np.array(sprite),value,(1,0)))
			elif name == 'pixelate':
				wid,hgt = sprite.size
				mx,my = value if len(value) == 2 else (value[0],value[0])
				sprite = sprite.resize((math.floor(sprite.width/value[0]),math.floor(sprite.height/value[1])), resample=Image.NEAREST)
				sprite = sprite.resize((wid,hgt), resample=Image.NEAREST)
			elif name == 'glitch' and all([x!=0 for x in value]):
				fil = np.random.random_integers(0x80-value[0],0x80+value[0],(sprite.size[1],sprite.size[0],2))
				fil[np.random.choice([False,True],(sprite.size[1],sprite.size[0]),p=(value[1],1-value[1]))] = 0x80
				fil=np.pad(fil,((0,0),(0,0),(0,2)),constant_values=(255,))
				sprite = filterimage.apply_filterimage(sprite,fil,absolute=False)
			elif name == 'wavex' and value[1]!=0:
				numpysprite = np.array(sprite)
				for l in range(len(numpysprite)):
					off = np.sin(((l/numpysprite.shape[0])*value[2]*np.pi*2)+(value[0]/numpysprite.shape[0]*np.pi*2))*value[1]
					numpysprite[l]=rotate(numpysprite[l].tolist(),int(off+0.5))
				sprite = Image.fromarray(numpysprite)
			elif name == 'wavey' and value[1]!=0:
				numpysprite = np.array(sprite).swapaxes(0,1)
				for l in range(len(numpysprite)):
					off = np.sin(((l/numpysprite.shape[0])*value[2]*np.pi*2)+(value[0]/numpysprite.shape[0]*np.pi*2))*-value[1]
					numpysprite[l]=rotate(numpysprite[l].tolist(),int(off+0.5))
				sprite = Image.fromarray(numpysprite.swapaxes(0,1))
			elif name == 'lockhue' and value != None:
				sprite = Image.fromarray(lock(0,np.array(sprite,dtype="uint8"),value))
			elif name == 'locksat' and value != None:
				sprite = Image.fromarray(lock(1,np.array(sprite,dtype="uint8"),value))
			elif name == 'gradientx' and value!=(1,1,1,1):
				numpysprite = np.array(sprite).swapaxes(0,1)
				for l in range(len(numpysprite)):
					v=gradient(l,*(value*np.array([24,24,1,1])))
					numpysprite[l]=numpysprite[l]*(v,v,v,1)
				sprite = Image.fromarray(numpysprite.swapaxes(0,1))
			elif name == 'gradienty' and value!=(1,1,1,1):
				numpysprite = np.array(sprite)
				for l in range(len(numpysprite)):
					v=gradient(l,*(value*np.array([24,24,1,1])))
					numpysprite[l]=numpysprite[l]*(v,v,v,1)
				sprite = Image.fromarray(numpysprite)
			elif name == 'flipx':
				sprite = ImageOps.mirror(sprite)
			elif name == 'flipy':
				sprite = ImageOps.flip(sprite)
			elif name == 'blank':
				sprite = Image.composite(Image.new("RGBA", (sprite.width, sprite.height), (255,255,255,255)),sprite,sprite)
			elif name == 'scanx':
				sprite = Image.fromarray(scan(np.array(sprite),value))
			elif name == 'scany':
				sprite = Image.fromarray(scan(np.array(sprite).swapaxes(0,1),value).swapaxes(0,1))
			elif name == 'invert':
				sprite = Image.fromarray(np.dstack((~np.array(sprite)[:,:,:3],np.array(sprite)[:,:,3])))
			elif name == 'melt':
				sprite_arr = list(np.array(sprite,dtype=np.uint8).swapaxes(0,1))
				for i, col in enumerate(sprite_arr):
					l = col
					col_removed = list(filter(lambda a: a[3] != 0, col))
					while len(col_removed) < len(col):
						col_removed.insert(0,(0,0,0,0))
					sprite_arr[i]=col_removed
				sprite = Image.fromarray(np.array(sprite_arr,dtype=np.uint8).swapaxes(0,1))
			elif name == 'liquify':
				sprite = Image.fromarray(liquify.liquify(np.array(sprite)))
			elif name == 'planet':
				sprite = liquify.planet(np.array(sprite))
			elif name == 'reverse':
				im = np.array(sprite.convert('RGBA'),dtype=np.uint8)
				def colortoint(a):
					return int.from_bytes(bytearray(a),byteorder='big')
				def inttocolor(a):
					return np.array(tuple(a.to_bytes(byteorder='big',length=4)))
				colors = []
				for y, row in enumerate(im):
					for x, pixel in enumerate(row):
						if pixel[3] != 0:
							colors.append(colortoint(pixel))
				colors = [a for a,_ in collections.Counter(colors).most_common()]
				colors_inverted = colors[::-1]
				im_inverted = np.zeros(im.shape,dtype=np.uint8)
				for y, row in enumerate(im):
					for x, pixel in enumerate(row):
						if pixel[3] != 0:
							im_inverted[y,x] = inttocolor(colors[colors_inverted.index(colortoint(pixel))])
				sprite = Image.fromarray(im_inverted)
			elif name == 'fisheye' and value != 0:
				spritefish = fish.fish(np.array(sprite),value)
				sprite = Image.fromarray(spritefish)
			elif name == 'aberrate' and value != (0, 0):
				arr = np.array(sprite)
				arr = np.pad(arr,((abs(value[1]),abs(value[1])),(abs(value[0]),abs(value[0])),(0,0)))
				arr[:,:,0]=np.roll(arr[:,:,0],-value[0],1)
				arr[:,:,2]=np.roll(arr[:,:,2],value[0],1)
				arr[:,:,0]=np.roll(arr[:,:,0],-value[1],0)
				arr[:,:,2]=np.roll(arr[:,:,2],value[1],0)
				arr=arr.astype(np.uint16)
				arr[:,:,3]+=np.roll(np.roll(arr[:,:,3],-value[0],1),-value[1],0)
				arr[:,:,3]+=np.roll(np.roll(arr[:,:,3],value[0],1),value[1],0)
				arr[arr>255]=255
				sprite = Image.fromarray(arr.astype(np.uint8))
			elif name == 'opacity' and value < 1:
				arr=np.array(sprite,dtype=float)
				arr[:,:,3]*=value
				sprite = Image.fromarray(arr.astype(np.uint8))
			elif name == 'neon':
				#"i REALLY need to rewrite this"
				#Guess what.
				hv_directions = (
									 ( 0,-1),
							(-1, 0),          ( 1, 0),
									 ( 0, 1)
						)
				diag_directions = (
							(-1,-1),          ( 1,-1),

							(-1, 1),          ( 1, 1)
						)
				img = np.array(sprite, dtype=float)
				img = np.pad(img,((1,),(1,),(0,))) #Pad only the first 2 axes.
				colors = liquify.get_colors(img)
				for color in colors:
					neighbormap = np.zeros((img.shape[0],img.shape[1]),dtype=int)
					neighbormask = (img==color).all(2) #This array contains booleans, but this will automatically get typecasted to an int when performing math operations. (Because neighbotmap is an int, otherwise it'd be a float.)
					for direction in hv_directions:
						directionmask = neighbormask.copy()
						if direction[0]!=0:
							directionmask = np.roll(directionmask, direction[0], 1) #Roll directionmask on the X axis, numpy axis 1.
						if direction[1]!=0:
							directionmask = np.roll(directionmask, direction[1], 0) #Roll directionmask on the Y axis, numpy axis 0.
						neighbormap+=directionmask
					# if neighbors >= 4:
					img[:,:,3][neighbormask & (neighbormap[:,:]>=4)] //= abs(value)
					for direction in diag_directions:
						directionmask = neighbormask.copy()
						if direction[0]!=0:
							directionmask = np.roll(directionmask, direction[0], 1) #Roll directionmask on the X axis, numpy axis 1.
						if direction[1]!=0:
							directionmask = np.roll(directionmask, direction[1], 0) #Roll directionmask on the Y axis, numpy axis 0.
						neighbormap+=directionmask
					# if neighbors >= 8:
					img[:,:,3][neighbormask & (neighbormap[:,:]>=8)] //= abs(value)
				img = img[1:-1,1:-1].astype(np.uint8)
				if value < 0:
					notzero = img[:,:,3]!=0
					img[:,:,3][notzero]=255-img[:,:,3][notzero]
				sprite = Image.fromarray(img)
			elif name == 'warp' and value != ((0,0),(0,0),(0,0),(0,0)):
				widwarp = [-1*min(value[0][0],value[3][0],0),max((value[2][0]),(value[1][0]),0)]
				hgtwarp = [-1*min(value[0][1],value[1][1],0),max((value[2][1]),(value[3][1]),0)]
				paddedwidth = int(max(math.floor(widwarp[0]),math.floor(widwarp[1])))
				paddedheight = int(max(math.floor(hgtwarp[0]),math.floor(hgtwarp[1])))
				spritenumpywarp = np.array(sprite)
				srcpoints = np.array(
					[
						[paddedwidth,paddedheight],
						[sprite.width+paddedwidth,paddedheight],
						[sprite.width+paddedwidth,sprite.height+paddedheight],
						[paddedwidth,sprite.height+paddedheight]
					]
				)
				dstpoints = np.array(
					[
						[value[0][0]+paddedwidth,value[0][1]+paddedheight],
						[sprite.width+value[1][0]+paddedwidth,value[1][1]+paddedheight],
						[sprite.width+value[2][0]+paddedwidth,sprite.height+value[2][1]+paddedheight],
						[value[3][0]+paddedwidth,sprite.height+value[3][1]+paddedheight]
					]
				)
				srcpoints = np.float32(srcpoints.tolist())
				dstpoints = np.float32(dstpoints.tolist())
				Mwarp = cv2.getPerspectiveTransform(srcpoints, dstpoints)
				spritenumpywarp = np.pad(spritenumpywarp,((paddedheight,paddedheight),(paddedwidth,paddedwidth),(0,0)), 'constant', constant_values=0)
				warped = cv2.warpPerspective(spritenumpywarp, Mwarp, dsize=(int((paddedwidth*2)+sprite.width), int((paddedheight*2)+sprite.height)), flags = cv2.INTER_NEAREST)
				sprite = Image.fromarray(warped)
			elif name == 'angle' and value != 0:
				sprite = sprite.rotate(-value,expand=True)
			elif name == 'blur_radius' and value != 0:
				sprite = sprite.filter(ImageFilter.GaussianBlur(radius = value))
		return sprite

	def make_meta(self, img: Image.Image, level: int) -> Image.Image:
		'''Applies a meta filter to an image.'''
		if abs(level) > constants.MAX_META_DEPTH:
			raise ValueError(level)
		
		orig = img.copy()
		base = img.getchannel("A")
		if level<0:
			level=abs(level)
			base=ImageOps.invert(base)
		for _ in range(level):
			temp = base.crop((-2, -2, base.width + 2, base.height + 2))
			filtered = ImageChops.invert(temp).filter(ImageFilter.FIND_EDGES)
			base = filtered.crop((1, 1, filtered.width - 1, filtered.height - 1))
		
		base = Image.merge("RGBA", (base, base, base, base))
		if level % 2 == 0 and level != 0:
			base.paste(orig, (level, level), mask=orig)
		elif level % 2 == 1 and level != 1:
			blank = Image.new("RGBA", orig.size)
			base.paste(blank, (level, level), mask=orig)
		return base

	def save_frames(
		self,
		imgs: list[Image.Image],
		out: str | BinaryIO,
		durations: list[int],
		extra_out: str | BinaryIO | None = None,
		extra_name: str = 'render',
		format: str = 'gif'
	) -> None:
		'''Saves the images as a gif to the given file or buffer.
		
		If a buffer, this also conveniently seeks to the start of the buffer.
		If extra_out is provided, the frames are also saved as a zip file there.
		'''
		if format == 'gif':
			imgs[0].save(
				out,
				format="GIF",
				interlace=True,
				save_all=True,
				append_images=imgs[1:],
				loop=0,
				duration=durations,
				disposal=2, # Frames don't overlap
				transparency=255,
				background=255,
				optimize=False
			)
		elif format == 'png':
			imgs[0].save(
				out,
				format="PNG",
				save_all=True,
				append_images=imgs,
				default_image=True,
				loop=0,
				duration=durations
			)
		if not isinstance(out, str):
			out.seek(0)
		if extra_name == None: extra_name = 'render'
		if extra_out is not None:
			file = zipfile.PyZipFile(extra_out, "x")
			for i, img in enumerate(imgs):
				buffer = BytesIO()
				img.save(buffer, "PNG")
				file.writestr(f"{extra_name}_{i//3}_{(i%3)+1}.png", buffer.getvalue())
			file.close()

def setup(bot: Bot):
	bot.renderer = Renderer(bot)

