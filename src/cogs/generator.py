from __future__ import annotations

from typing import Literal

import discord
import numpy as np
import json
import random
import time
import re
import zlib
import zipfile

from io import BytesIO
from discord.ext import commands
from PIL import Image
from os import urandom as _urandom

from ..types import Bot, Context
from .. import constants


class GeneratorCog(commands.Cog, name="Generation Commands"):
	def __init__(self, bot: Bot):
		self.bot = bot

	# Rewriting Random so I can get the seed
	# https://stackoverflow.com/a/34699351/13290530
	class Random(random.Random):
		def seed(self, a=None):
			if a is None:
				a = int(time.time() * 256) % (2**64)  # use fractional seconds
			self._current_seed = a
			super().seed(a)
		def get_seed(self):
			return self._current_seed

	def recolor(self, sprite: Image.Image, color: str, palette: np.ndarray) -> Image.Image:
		"""Apply rgb color"""
		r, g, b = palette[constants.COLOR_NAMES[color][::-1]]
		arr = np.asarray(sprite, dtype="float64")
		arr[..., 0] *= r / 256
		arr[..., 1] *= g / 256
		arr[..., 2] *= b / 256
		return Image.fromarray(arr.astype("uint8"))

	# New code for character generation
	@commands.command(aliases=["char"])
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def character(self, ctx: Context):
		"""Randomly generate a character using prefabs."""
		await ctx.trigger_typing()
		if not await ctx.bot.is_owner(ctx.author): #keep this off limits for now
			return await ctx.error('The bot owner\'s currently remaking =character. Please use =oldcharacter or =oldchar for now.')


	# Old code for character generation

	def old_blacken(self, sprite: Image.Image, palette) -> np.ndarray:
		"""Apply black (convenience)"""
		return self.recolor(np.array(sprite.convert("RGBA")), "black", palette)

	def old_paste(self, src: Image.Image, dst: Image.Image, loc: tuple(int, int), snap: int = 0):
		src.paste(dst, tuple([int(x - (s / 2)) for x, s in zip(loc, dst.size)]) if snap == 0 else (int(loc[0] - (dst.width / 2)), min(loc[1], 24 - dst.height)) if snap == 1 else (int(loc[0] - (dst.width / 2)), loc[1] - dst.height), dst.convert("RGBA"))
		return src

	def old_generate_image(self, ears, legs, eyes, mouth, color, variant, type, rand):
		with Image.open(f"data/generator/legacy/sprites/{type}_{variant}.png") as im:
			palette = np.array(Image.open(f"data/palettes/default.png").convert("RGB"))
			with open("data/generator/legacy/spritedata.json") as f:
				spritedata = json.loads(f.read())

			if legs != 0:
				positions = spritedata[type][variant][("1leg" if legs == 1 else f"{legs}legs")]
				for leg in positions:
					with Image.open(f"data/generator/legacy/sprites/parts/legs/{rand.randint(1,5)}.png") as i:
						im = self.old_paste(im, i, leg, 1)
			if ears != 0:
				positions = spritedata[type][variant][("1ear" if ears == 1 else "2ears")]
				for ear in positions:
					with Image.open(f"data/generator/legacy/sprites/parts/ears/{rand.randint(1,4)}.png") as i:
						im = self.old_paste(im, i, ear, 2)
			if eyes != 0:
				with Image.open(f"data/generator/legacy/sprites/parts/eyes/{eyes}.png") as i:
					im = self.old_paste(im, self.old_blacken(i, palette), spritedata[type][variant]["eyes"][0])
			if mouth:
				try:
					with Image.open(f"data/generator/legacy/sprites/parts/mouth.png") as i:
						im = self.old_paste(im, self.old_blacken(i, palette), spritedata[type][variant]["mouth"][0])
				except:
					pass

			# Recolor after generation
			im = self.recolor(np.array(im), color, palette)

			# Send generated sprite
			btio = BytesIO()
			im.resize((192, 192), Image.NEAREST).save(btio, "png")
			btio.seek(0)
			return btio

	@commands.command(aliases=["oldchar"])
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def oldcharacter(self, ctx: Context, *, seed: str = None):
		"""Old code for =char, kept for legacy purposes"""
		rand = self.Random()
		try:
			seed = int(seed)
		except ValueError:
			seed = int.from_bytes(bytes(seed, "utf-8"), "big") % (2**64)
		except TypeError:
			seed = None
		rand.seed(seed)
		ears = rand.choice([0, 0, 0, 1, 2, 2, 2, 2])
		legs = rand.choice([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])
		eyes = rand.choice([0, 0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6])
		mouth = rand.random() > 0.75
		color = rand.choice(["pink", "red", "maroon", "yellow", "orange", "gold", "brown", "lime", "green", "cyan", "blue", "purple", "white", "silver", "grey"])
		variant = rand.choice(["smooth", "fuzzy", "fluffy", "polygonal", "skinny", "belt"])
		typ = rand.choice(["long", "tall", "curved", "round"])
		a = rand.choice(["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z", "sh", "ch", "th", "ph", "cr", "gr", "tr", "br", "dr", "pr", "bl", "sl", "pl", "cl", "gl", "fl", "sk", "sp", "st", "sn", "sm", "sw"])
		b = rand.choice(["a", "e", "i", "o", "u", "ei", "oi", "ea", "ou", "ai", "au", "bu"])
		c = rand.choice(["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z", "sh", "ch", "ck", "th", "ph", "sk", "sp", "st"])
		name = rand.choice([a + b + a + b, a + b, a + b + c, b + c, a + c + b, a + c + b + a + c + b, b + c + b + c, a + b + c + a + b + c, b + a]).title()
		embed = discord.Embed(color=self.bot.embed_color, title=name, description=f"{name} is a __**{color}**__, __**{variant}**__, __**{typ}**__ creature with __**{eyes}**__ eye{'s' if eyes != 1 else ''}, __**{ears}**__ ear{'s' if ears != 1 else ''}{', __**a mouth**__' if mouth else ''}{f',and __**{legs}'}**__ leg{'s' if legs != 1 else ''}.")
		embed.set_footer(text=f"Seed: {rand.get_seed()}")
		file = discord.File(self.old_generate_image(ears, legs, eyes, mouth, color, variant, typ, rand), filename=f"{name}-{rand.get_seed()}.png")
		embed.set_image(url=f"attachment://{name}-{rand.get_seed()}.png")
		# note to self: it's literally this easy what are you doing
		await ctx.send(embed=embed, file=file)

	# Level generation

	@commands.command()
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def genlevel(self, ctx: Context, width: int, height: int):
		"""Generates a blank level given a width and height."""
		assert width >= 1 and height >= 1, "Too small!"
		t = time.time()
		width += 2
		height += 2
		if width * height > 4194304 and not await ctx.bot.is_owner(ctx.author):
			return await ctx.error("Level size too big! Levels are capped at an area of 2048 by 2048, including borders.")
		b = bytearray(b"\x41\x43\x48\x54\x55\x4e\x47\x21\x05\x01\x4d\x41\x50\x20\x02\x00\x00\x00\x00\x00\x4c\x41\x59\x52\x1d\x01\x00\x00\x03\x00")
		blankrow_bordered = b"\x00\x00" + (b"\xFF\xFF" * (width - 2)) + b"\x00\x00"
		blankrow_borderless = b"\xFF\xFF" * (width)
		for n in range(3):
			b.extend(int.to_bytes(width, length=4, byteorder="little"))
			b.extend(int.to_bytes(height, length=4, byteorder="little"))
			b.extend(b"\x0c\x00\x0c\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x80\x3f\x00\x00\x01\x00\x00\x80\x3f\xff\xff\xff\x02\x4d\x41\x49\x4e")
			l = bytearray()
			for m in range(height):
				if n == 0:
					if m == 0 or m == height - 1:
						l.extend(b"\x00\x00" * width)
					else:
						l.extend(blankrow_bordered)
				else:
					l.extend(blankrow_borderless)
			cl = zlib.compress(l)
			b.extend(int.to_bytes(len(cl), length=4, byteorder="little"))
			b.extend(cl)
			b.extend(b"\x44\x41\x54\x41\x01\x00\x00\x00\x00")
			datbin = zlib.compress(bytearray([3] * (len(l) // 2)))
			b.extend(len(datbin).to_bytes(length=4, byteorder="little"))
			b.extend(datbin)
		zipbuf = BytesIO()
		with zipfile.PyZipFile(zipbuf, "x") as f:
			f.writestr(f"{width-2}x{height-2}.l", b)
			f.writestr(
				f"{width-2}x{height-2}.ld",
				f"""[general]
selectorY=-1
unlockcount=0
leveltype=0
specials=0
disableshake=0
levels=0
selectorX=-1
disableruleeffect=0
customruleword=
music=baba
rhythm=10
author=robot is chill
levelid=-1
subtitle=
currobjlist_total=0
paths=0
particles=
disableparticles=0
levelz=20
palette=default.png
localmusic=0
customparent=
paletteroot=1
name={width-2}x{height-2}""",
			)
			f.writestr(f"{width-2}x{height-2}.png", b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\xd8\x00\x00\x00\x70\x08\x03\x00\x00\x00\x4b\xb1\x4e\x5c\x00\x00\x00\x01\x73\x52\x47\x42\x00\xae\xce\x1c\xe9\x00\x00\x00\x04\x67\x41\x4d\x41\x00\x00\xb1\x8f\x0b\xfc\x61\x05\x00\x00\x00\x0f\x50\x4c\x54\x45\x5b\x82\x38\xa4\xb0\x3e\x72\x72\x72\x10\x10\x10\x00\x00\x00\x0a\xbd\x8d\x19\x00\x00\x00\x05\x74\x52\x4e\x53\xff\xff\xff\xff\x00\xfb\xb6\x0e\x53\x00\x00\x00\x09\x70\x48\x59\x73\x00\x00\x0e\xc2\x00\x00\x0e\xc2\x01\x15\x28\x4a\x80\x00\x00\x02\x5c\x49\x44\x41\x54\x78\x5e\xed\xd4\xcb\x72\xa4\x30\x10\x04\x40\xef\xda\xff\xff\xcd\x7b\x50\xb6\x22\xc0\x30\x88\x97\xc7\xc1\x56\x5e\x26\x10\xa8\xab\xeb\x32\x1f\x5f\xbb\x7c\x0c\xf1\xf1\x5b\xa5\x58\x63\xf3\x0d\x3e\x7e\xab\x14\x6b\x6c\xfe\xf1\x67\x99\xb7\xd7\x10\x79\x50\x8a\x35\x32\x53\xec\x0c\x91\x07\x9d\x2b\xf6\x17\x8f\xd7\x10\xd1\x89\xde\x29\xc5\x1a\x59\x29\x76\x84\x88\x4e\xf4\x4e\xd7\x16\xf3\x78\x8c\x19\x9d\xa8\x62\x83\x51\x29\xd6\x08\x49\xb1\x3d\xcc\xe8\x44\x15\x1b\x8c\xba\xb5\xd8\xe7\x8c\xe3\x0d\x66\x75\x22\x6d\x30\x2a\xc5\x1a\x21\x29\x36\xe5\x78\x83\x59\x9d\x48\x1b\x8c\x3a\x57\x6c\xce\x6a\x45\x9f\xce\x71\x71\x5a\x9c\x76\x66\xa6\xd8\x54\x8a\x35\x42\x9e\x5b\x0c\xd1\xf3\x42\x83\x14\x2a\x4e\x3b\xb3\x53\x6c\x2a\xc5\x1a\x21\x45\xf4\x03\x8a\x15\x61\xa2\x0f\x16\xdb\x60\x76\x8a\x4d\xa5\xd8\x94\x30\xd1\xcf\x2d\x56\xac\x74\x8e\x59\x9d\x2c\xc9\xa3\x52\x6c\x4a\x98\xe8\xce\x6a\xe7\x98\xd5\xc9\x92\x3c\x2a\xc5\xa6\x84\x15\x2b\x14\x1b\x1e\x64\x48\x15\x2a\x92\x47\xa5\xd8\x94\xb0\x62\x95\x62\xc3\x83\x0c\x49\xb1\x65\xd7\x14\x2b\x56\xba\x86\x99\x47\x37\xf4\xbb\x93\xcc\x39\x2b\x5d\xc3\xcc\x14\x9b\x4a\xb1\x46\x56\xb1\xc2\x1a\xff\x06\x83\x5c\xea\x64\xa4\xd8\x54\x8a\x35\xb2\x8a\x15\xd6\xd8\x78\x90\x4b\x9d\x8c\x9f\x2d\x26\xbb\x58\xad\x38\xed\x1c\x0f\x72\xa9\x17\xc3\x06\xa3\x52\xac\x11\x92\x62\x8d\xe3\x41\x2e\xbd\xb5\x98\x55\xd6\xf8\xea\x60\xb1\x22\xd2\x06\xa3\x52\xac\x11\x92\x62\x8d\xc7\x41\x2e\x75\x22\x6d\x30\xea\x17\x16\x2b\x2e\x1f\xfc\x13\x49\xb1\xc6\xf0\x14\x6b\x3c\xee\xe4\xf2\xbd\xc5\x0c\xed\x64\x5a\x61\x8d\xaf\x4e\x16\x43\x72\x8a\xf9\xdd\x60\x68\x27\xcc\x0a\x6b\x7c\x95\x62\x53\x2e\x17\xc9\x37\x15\x13\x52\xac\x30\xe7\xed\xc1\x42\x73\x86\xd9\x20\xc5\xfc\x6e\x30\x34\xc5\x1a\x8f\x27\x19\x66\x83\x9b\x8b\xc9\x5c\xe1\xa3\x9d\xc5\x5c\xfa\x76\xcb\xb1\x0d\x52\xcc\xef\x06\x43\x53\xac\xf1\x38\xc8\xa5\x14\x7b\x2d\xc5\x1a\x21\x29\xb6\x87\x4b\xbf\xb2\x98\xb7\xc5\xe9\x20\x97\x52\xec\xb5\x14\x6b\x84\x3c\xb7\xd8\x1b\xd8\x20\xc5\xfc\x6e\x30\x34\xc5\x6e\x60\x83\x9b\x8a\xad\x90\xbd\xf5\x6f\xf1\xfa\x2b\x6f\x7b\x83\x15\x36\xda\x92\x62\x2f\x59\x29\xc5\x16\x79\xfb\xb3\xc5\x56\xc8\xda\x2a\xe6\x6d\x71\x3a\xe7\xed\xb7\x62\xb2\x76\x4a\xb1\x45\xa2\x53\xac\x71\x3a\xe7\xed\x73\x8b\x15\xb3\x53\x6c\x2a\xc5\x16\x89\xfe\x6f\x8b\x95\x5d\x5f\xa5\xd8\xb2\x14\x5b\x24\x3a\xc5\xf6\x7c\x95\x62\xcb\x52\x6c\x91\xe8\xe7\x17\xbb\x94\xd9\x29\x36\x95\x62\x8b\x44\x3f\xb7\xd8\xad\x64\xed\x94\x62\x8b\x44\xdf\x4b\xd6\x4e\x0f\x2d\xf6\xf5\xf5\x0f\xe1\xac\xed\x45\x1e\x29\xc0\x22\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82")
		zipbuf.seek(0)
		return await ctx.send(f"Generated in {int((time.time()-t)*1000)} ms.\nUnzip this into your `Baba Is You/Data/Worlds/levels/` folder to view.", files=[discord.File(BytesIO(zipbuf.read()), filename=f"{width-2}x{height-2}.zip")])


def setup(bot: Bot):
	bot.add_cog(GeneratorCog(bot))
