from __future__ import annotations

from discord.ext import commands
import copy
import math
import os
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
import src.cogs.seamcarving as seamcarving
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

def grayscale(arr,influence):
    hsv=rgb_to_hsv(arr)
    hsv[...,1] = np.clip(hsv[...,1]*1-influence,[0],[1])
    rgb=hsv_to_rgb(hsv)
    return rgb

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
        rc = rc.point(lambda i: i * (r/256))
        gc = gc.point(lambda i: i * (g/256))
        bc = bc.point(lambda i: i * (b/256))
        
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
        printme: bool = False,
        crop: tuple[int,int,int,int] | None = None
    ):
        '''Takes a list of tile objects and generates a gif with the associated sprites.

        `out` is a file path or buffer. Renders will be saved there, otherwise to `target/renders/render.gif`.

        `palette` is the name of the color palette to refer to when rendering.

        `images` is a list of background image filenames. Each image is retrieved from `data/images/{image_source}/image`.

        `background` is a palette index. If given, the image background color is set to that color, otherwise transparent. Background images overwrite this. 
        '''
        palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")
        sprite_cache: dict[str, Image.Image] = {}
        imgs = []
        times = []
        width = len(grid[0][0])
        height = len(grid[0])
        img_width_raw = int(width * (constants.DEFAULT_SPRITE_SIZE*scaleddef))
        img_height_raw =  int(height * (constants.DEFAULT_SPRITE_SIZE*scaleddef))
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
        img_width = int(img_width_raw + (2 * padding))
        img_height = int(img_height_raw + (2 * padding))
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
                    t = time.time()
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
                                imgs[frame].paste(
                                    Image.fromarray(cv2.min(np.asarray(imgs[frame]),imgtempar)),
                                    (
                                        0,
                                        0
                                    ), 
                                    mask=imgtemp
                                )
                            elif tile.blending == 'multiply':
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
                                imgtempar[:,:,3] = 1
                                imgs[frame].paste(
                                    Image.fromarray(cv2.bitwise_and(np.asarray(imgs[frame]),np.asarray(imgtemp))),
                                    (
                                        0,
                                        0
                                    ), 
                                    mask=imgtemp
                                )
                            else:
                                imgs[frame].paste(
                                    sprite, 
                                    (
                                        int(x * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - x_offset_disp),
                                        int(y * (constants.DEFAULT_SPRITE_SIZE*scaleddef) + padding - y_offset_disp)
                                    ), 
                                    mask=sprite
                                )
                        times.append(tile.delta + (time.time() - t))
        
        if printme:
            q = ''
            for i in imgs:
                for vy in np.array(i):
                    for vx in vy:
                        q = q + (f'\x1b[48;2;{vx[0]};{vx[1]};{vx[2]}m  \x1b[0m' if vx[3]>0 else '  ')
                    q = q + '\n'
            print(q)
        if before_image:
            bfr=0
            for frame in ImageSequence.Iterator(before_image):
                im = frame.convert('RGBA').resize((frame.width//2,frame.height//2),Image.NEAREST)
                newImage = Image.new('RGBA', (img_width,img_height), (0, 0, 0, 0))
                newImage.paste(im, (padding-pad_l,padding-pad_u),mask=im)
                imgs.insert(bfr,newImage) 
                bfr += 1 #i dont wanna use an enumerate on an iterator
        outs = []
        for img in imgs:
            if type(gridol) != type(None):
                img = np.array(img,dtype=np.uint8)
                for col in range(img.shape[0]//(gridol[0]*2)):
                    img[col*gridol[0]*2,:,:] = ~img[col*gridol[0]*2,:,:]
                    img[col*gridol[0]*2,:,3] = 255
                for row in range(img.shape[1]//(gridol[1]*2)):
                    img[:,row*gridol[1]*2,:] = ~img[:,row*gridol[1]*2,:]
                    img[:,row*gridol[1]*2,3] = 255
                img = Image.fromarray(img)
            img = img.crop((padding - pad_l, padding - pad_u, img.width - padding + pad_r, img.height - padding + pad_d))
            if crop != None:
                print(crop)
                img = img.crop((crop[0],crop[1],img.width-crop[2],img.height-crop[3]))
            if upscale != 1:
                img = img.resize((int(upscale * img.width), int(upscale * img.height)), resample=Image.NEAREST)
            outs.append(img)
    
        self.save_frames(
            outs,
            out,
            speed=speed,
            extra_out=extra_out,
            extra_name=extra_name,
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

        t = time.time()
        if tile.empty:
            return ReadyTile(None)
        out = []
        x, y = position
        for frame in range(3):
            wobble = (11 * x + 13 * y + frame) % 3 if random_animations else frame
            if 'freeze' in tile.filters.keys():
                wobble=0
            if tile.custom:
                sprite = await self.generate_sprite(
                    tile.name,
                    style=tile.custom_style or "noun",
                    direction=tile.custom_direction,
                    meta_level=tile.meta_level,
                    wobble=wobble,
                    filters=tile.filters,
                    position=(x,y),
                    gscale=gscale
                )
            else:
                if tile.name in ("icon",):
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
                sprite = cached_open(path, cache=sprite_cache, fn=Image.open).convert("RGBA")
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
                    overlay = Image.open(f"data/overlays/{tile.overlay}.png").convert("RGBA").resize((int(constants.DEFAULT_SPRITE_SIZE*gscale),int(constants.DEFAULT_SPRITE_SIZE*gscale)),Image.NEAREST)
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
            numpysprite = np.array(sprite)
            numpysprite[np.all(numpysprite[:,:,:3]<=(0,0,0),axis=2)&(numpysprite[:,:,3]>1),:3]=8
            sprite = Image.fromarray(numpysprite)
            out.append(sprite)
        if 'land' in tile.filters.keys():
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
        return ReadyTile((f0, f1, f2), tile.cut_alpha, tile.mask_alpha, tile.displace, tile.blending, time.time()-t)

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
        filters: dict,
        gscale: float,
        position: tuple[int,int]
    ) -> Image.Image:
        '''Generates a custom text sprite'''
        text = text[5:]
        raw = text.replace("/", "")
        newline_count = text.count("/")

        if seed is None:
            seed = int((7+position[0])/(3+position[1])*100000000)
        seed_digits = [(seed >> 8 * i ) | 0b11111111 for i in range(len(raw))]
        
        if len(text) == 1:
            style = "letter"
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
        filters: dict,
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
        filters: dict, #as of python 3.6, dictionaries are ordered
    ):
      '''Takes an image, with or without a plate, and applies the given options to it.'''
      #rocket's handlers are such a better system but i have no idea how they work since i don't really know OOP 
      try:
        meta_level = filters['meta_level']
      except KeyError:
        meta_level = 0
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
      if style == 'property': #properties are so weird 
        print(original_style)
        if original_style != "property":
          # box: position of upper-left coordinate of "inner text" in the larger text tile
          plate, box = self.bot.db.plate(original_direction, wobble)
          plate_alpha = plate.getchannel("A")
          sprite_alpha = sprite.getchannel("A")
          alpha = ImageChops.subtract(plate_alpha, sprite_alpha)
          sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
          sprite = sprite.crop((box[0], box[1], constants.DEFAULT_SPRITE_SIZE + box[0], constants.DEFAULT_SPRITE_SIZE + box[1]))
        else:
          plate, box = self.bot.db.plate(direction, wobble)
          plate_alpha = plate.getchannel("A")
          sprite_alpha = sprite.getchannel("A")
          if meta_level % 2 == 0:
            alpha = ImageChops.subtract(plate_alpha, sprite_alpha)
          else: 
            alpha = ImageChops.add(plate_alpha, sprite_alpha)
          sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
      if original_style == "property" and style == 'noun':
          # box: position of upper-left coordinate of "inner text" in the larger text tile
          plate, box = self.bot.db.plate(original_direction, wobble)
          plate_alpha = plate.getchannel("A")
          sprite_alpha = sprite.getchannel("A")
          alpha = ImageChops.subtract(plate_alpha, sprite_alpha)
          sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
          sprite = sprite.crop((box[0], box[1], constants.DEFAULT_SPRITE_SIZE + box[0], constants.DEFAULT_SPRITE_SIZE + box[1]))
      for filter in filters.items():
        print(filter[1])
        if filter[0] == 'threeoo' and filter[1] != None:
            sprite = np.array(sprite,dtype=np.uint8)
            h,w,_ = sprite.shape
            assert h <= 24 and w <= 24, 'Image too large for 3oo filter!'
            sprite = Image.fromarray(seamcarving.seam_carve(sprite,(sprite.shape[0]//filter[1],sprite.shape[1]//filter[1]))).resize((w,h),Image.NEAREST)
        elif filter[0] == 'normalize':
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
            print(displacement,absolute_center,center,(miny,minx),(maxy,maxx))
            sprite = Image.fromarray(np.roll(sprite,displacement[::-1],(1,0))) 
        elif filter[0] == 'pad' and any(filter[1]):
            sprite = Image.fromarray(np.pad(np.array(sprite),((filter[1][1],filter[1][3]),(filter[1][0],filter[1][2]),(0,0))))
        elif filter[0] == 'floodfill' and type(filter[1]) == float:
            f = lambda x: 420 if x > 0 else 0
            g = lambda x: 0 if x == 69 else 255
            im = np.array(sprite)
            ima = im[:,:,3]
            ima = np.pad([[f(b) for b in a] for a in ima],((1,1),(1,1)))
            imf = np.array([[g(b) for b in a] for a in cv2.floodFill(ima, np.full((ima.shape[0]+2,ima.shape[1]+2),np.uint8(0)),(0,0),69,flags=4)[1]])
            im[:,:,3] = imf[1:-1,1:-1]
            for y in range(len(im)):
                for x in range(len(im[0])):
                    if all([x == 0 for x in im[y,x,:3]]) and im[y,x,3] == 255:
                        im[y,x,:] = np.array([round(filter[1]*255),round(filter[1]*255),round(filter[1]*255),255])  #somehow this doesn't fuck up anywhere
            sprite = Image.fromarray(np.array(im))
        elif filter[0] == 'colslice' and filter[1] != None:
          im = np.array(sprite)
          colors = []
          for x in range(im.shape[1]):
            for y in range(im.shape[0]):
              if im[y,x,3] > 0 :
                colors.append(tuple(im[y,x]))
          if len(filter[1]) == 1:
            try:
              color = np.array([collections.Counter(colors).most_common()[filter[1][0]][0]])
            except:
              color = [(0,0,0,0)]
          elif len(filter[1]) in (2,3):
            try:
                if len(filter[1]) == 2:
                    color = np.array([n[0] for n in collections.Counter(colors).most_common()[filter[1][0]:filter[1][1]]])
                else:
                    color = np.array([n[0] for n in collections.Counter(colors).most_common()[filter[1][0]:filter[1][1]:filter[1][2]]])
            except:
                pass
          out = np.zeros((im.shape[0],im.shape[1],im.shape[2]),dtype=np.uint8)
          for x in range(im.shape[1]):
            for y in range(im.shape[0]):
              if any([all([a==b for a,b in zip([n for n in im[y,x]],c)]) for c in color]):
                out[y,x] = im[y,x]    
          sprite = Image.fromarray(out)
        elif filter[0] == 'meta_level' and filter[1] != 0:
                sprite = self.make_meta(sprite, filter[1])
        elif filter[0] == 'crop' and any(filter[1]):
            cropped = sprite.crop((filter[1][0],filter[1][1],filter[1][0]+filter[1][2],filter[1][1]+filter[1][3]))
            im = Image.new('RGBA',(sprite.width,sprite.height),(0,0,0,0))
            im.paste(cropped,(filter[1][0],filter[1][1]))
            sprite = im
        elif filter[0] == 'scale' and any([x!=1 for x in filter[1]]):
            sprite = sprite.resize((math.floor(sprite.width*filter[1][0]),math.floor(sprite.height*filter[1][1])), resample=Image.NEAREST)
        elif filter[0] == 'wrap' and any([x!=0 for x in filter[1]]):
            sprite = Image.fromarray(np.roll(np.array(sprite),filter[1],(1,0)))
        elif filter[0] == 'pixelate' and filter[1] > 1:
            wid,hgt = sprite.size
            sprite = sprite.resize((math.floor(sprite.width/filter[1]),math.floor(sprite.height/filter[1])), resample=Image.NEAREST)
            sprite = sprite.resize((wid,hgt), resample=Image.NEAREST)
        elif filter[0] == 'glitch' and all([x!=0.0 for x in filter[1]]):
            clamp = lambda m,mn,mx: min(mx,max(mn,m))
            filter = np.array([[[clamp(random.randint(128-filter[1][0],128+filter[1][0]),0,255),clamp(random.randint(128-filter[1][0],128+filter[1][0]),0,255),255,255] if filter[1][1] > random.random() else [128,128,255,255] for _ in range(sprite.size[1])] for _ in range(sprite.size[0    ])], dtype=np.uint8)
            sprite = filterimage.apply_filterimage(sprite,filter,absolute=False)
        elif filter[0] == 'wavex' and filter[1][1]!=0:
            numpysprite = np.array(sprite)
            for l in range(len(numpysprite)):
                off = np.sin(((l/numpysprite.shape[0])*filter[1][2]*np.pi*2)+(filter[1][0]/numpysprite.shape[0]*np.pi*2))*filter[1][1]
                numpysprite[l]=rotate(numpysprite[l].tolist(),int(off+0.5))
            sprite = Image.fromarray(numpysprite)
        elif filter[0] == 'wavey' and filter[1][1]!=0:
            numpysprite = np.array(sprite).swapaxes(0,1)
            for l in range(len(numpysprite)):
                off = np.sin(((l/numpysprite.shape[0])*filter[1][2]*np.pi*2)+(filter[1][0]/numpysprite.shape[0]*np.pi*2))*-filter[1][1]
                numpysprite[l]=rotate(numpysprite[l].tolist(),int(off+0.5))
            sprite = Image.fromarray(numpysprite.swapaxes(0,1))
        elif filter[0] == 'gradientx' and filter[1]!=(1,1,1,1):
            numpysprite = np.array(sprite).swapaxes(0,1)
            for l in range(len(numpysprite)):
                v=gradient(l,*(filter[1]*np.array([24,24,1,1])))
                numpysprite[l]=numpysprite[l]*(v,v,v,1)
            sprite = Image.fromarray(numpysprite.swapaxes(0,1))
        elif filter[0] == 'gradienty' and filter[1]!=(1,1,1,1):
            numpysprite = np.array(sprite)
            for l in range(len(numpysprite)):
                v=gradient(l,*(filter[1]*np.array([24,24,1,1])))
                numpysprite[l]=numpysprite[l]*(v,v,v,1)
            sprite = Image.fromarray(numpysprite)
        elif filter[0] == 'flipx':
            sprite = ImageOps.mirror(sprite)
        elif filter[0] == 'flipy':
            sprite = ImageOps.flip(sprite)
        elif filter[0] == 'blank':
            sprite = Image.composite(Image.new("RGBA", (sprite.width, sprite.height), (255,255,255,255)),sprite,sprite)
        elif filter[0] == 'scanx':
            sprite = Image.fromarray(scan(np.array(sprite),filter[1]))
        elif filter[0] == 'scany':
            sprite = Image.fromarray(scan(np.array(sprite).swapaxes(0,1),filter[1]).swapaxes(0,1))
        elif filter[0] == 'invert':
            sprite = Image.fromarray(np.dstack((~np.array(sprite)[:,:,:3],np.array(sprite)[:,:,3])))
        elif filter[0] == 'reverse':
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
        elif filter[0] == 'fisheye' and filter[1] != 0:
            spritefish = fish.fish(np.array(sprite),filter[1])
            sprite = Image.fromarray(spritefish)
        elif filter[0] == 'opacity' and filter[1] < 1:
            r,g,b,a = sprite.split()
            sprite = Image.merge('RGBA',(r,g,b,a.point(lambda i: i * filter[1])))
        elif filter[0] == 'neon' and filter[1] != 1:
            spritenp = np.array(sprite)
            spritenp2 = copy.deepcopy(spritenp)
            for x in range(spritenp.shape[1]):
                for y in range(spritenp.shape[0]):
                    if spritenp[y][x][3] > 0 :
                        neighbors = 0
                        for xo,yo in [[1,0],[0,1],[-1,0],[0,-1]]:
                            if (x+xo in range(spritenp.shape[1])) and (y+yo in range(spritenp.shape[0])):
                                neighbors += int(all(spritenp[y+yo,x+xo]==spritenp[y,x]))
                            else:
                                neighbors += int(not name.startswith('text_'))
                        if neighbors >= 4:
                            spritenp2[y,x,3] //= abs(filter[1])
                        for xo,yo in [[-1,-1],[-1,1],[1,-1],[1,1]]:
                            if (x+xo in range(spritenp.shape[1])) and (y+yo in range(spritenp.shape[0])):
                                neighbors += int(all(spritenp[y+yo,x+xo]==spritenp[y,x]))
                            else:
                                neighbors += int(not name.startswith('text_'))
                        if neighbors >= 8:
                            spritenp2[y,x,3] //= abs(filter[1])
            if filter[1] < 0:
                spritenp2 = np.array([[[r,g,b,(255-a if a != 0 else 0)] for r,g,b,a in row] for row in spritenp2],dtype=np.uint8)
            sprite = Image.fromarray(spritenp2)
        elif filter[0] == 'warp' and filter[1] != ((0,0),(0,0),(0,0),(0,0)):
            widwarp = [-1*min(filter[1][0][0],filter[1][3][0],0),max((filter[1][2][0]),(filter[1][1][0]),0)]
            hgtwarp = [-1*min(filter[1][0][1],filter[1][1][1],0),max((filter[1][2][1]),(filter[1][3][1]),0)]
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
                    [filter[1][0][0]+paddedwidth,filter[1][0][1]+paddedheight],
                    [sprite.width+filter[1][1][0]+paddedwidth,filter[1][1][1]+paddedheight],
                    [sprite.width+filter[1][2][0]+paddedwidth,sprite.height+filter[1][2][1]+paddedheight],
                    [filter[1][3][0]+paddedwidth,sprite.height+filter[1][3][1]+paddedheight]
                ]
            )
            srcpoints = np.float32(srcpoints.tolist())
            dstpoints = np.float32(dstpoints.tolist())
            Mwarp = cv2.getPerspectiveTransform(srcpoints, dstpoints)
            spritenumpywarp = np.pad(spritenumpywarp,((paddedheight,paddedheight),(paddedwidth,paddedwidth),(0,0)), 'constant', constant_values=0)
            warped = cv2.warpPerspective(spritenumpywarp, Mwarp, dsize=(int((paddedwidth*2)+sprite.width), int((paddedheight*2)+sprite.height)), flags = cv2.INTER_NEAREST)
            sprite = Image.fromarray(warped)
        elif filter[0] == 'angle' and filter[1] != 0:
            sprite = sprite.rotate(-filter[1],expand=True)
        elif filter[0] == 'blur_radius' and filter[1] != 0:
            sprite = sprite.filter(ImageFilter.GaussianBlur(radius = filter[1]))
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
        speed: int,
        extra_out: str | BinaryIO | None = None,
        extra_name: str = 'render',
    ) -> None:
        '''Saves the images as a gif to the given file or buffer.
        
        If a buffer, this also conveniently seeks to the start of the buffer.

        If extra_out is provided, the frames are also saved as a zip file there.
        '''
        imgs[0].save(
            out, 
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            loop=0,
            duration=speed,
            disposal=2, # Frames don't overlap
            transparency=255,
            background=255,
            optimize=False # Important in order to keep the color palettes from being unpredictable
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

