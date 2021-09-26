from __future__ import annotations

import copy
import math
import random
import zipfile
from collections import Counter
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO
import cv2

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageSequence
from src.tile import FullTile, ReadyTile

from .. import constants, errors
from ..utils import cached_open

if TYPE_CHECKING:
    from ...ROBOT import Bot

import src.cogs.fish as fish

def create_perspective_transform_matrix(src, dst):
    """ Creates a perspective transformation matrix which transforms points
        in quadrilateral ``src`` to the corresponding points on quadrilateral
        ``dst``.

        Will raise a ``np.linalg.LinAlgError`` on invalid input.
        """
    # See:
    # * http://xenia.media.mit.edu/~cwren/interpolator/
    # * http://stackoverflow.com/a/14178717/71522
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))


def create_perspective_transform(src, dst, round=False, splat_args=False):
    """ Returns a function which will transform points in quadrilateral
        ``src`` to the corresponding points on quadrilateral ``dst``::

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ... )
            >>> transform((5, 5))
            (74.99999999999639, 74.999999999999957)

        If ``round`` is ``True`` then points will be rounded to the nearest
        integer and integer values will be returned.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     round=True,
            ... )
            >>> transform((5, 5))
            (75, 75)

        If ``splat_args`` is ``True`` the function will accept two arguments
        instead of a tuple.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     splat_args=True,
            ... )
            >>> transform(5, 5)
            (74.99999999999639, 74.999999999999957)

        If the input values yield an invalid transformation matrix an identity
        function will be returned and the ``error`` attribute will be set to a
        description of the error::

            >>> tranform = create_perspective_transform(
            ...     np.zeros((4, 2)),
            ...     np.zeros((4, 2)),
            ... )
            >>> transform((5, 5))
            (5.0, 5.0)
            >>> transform.error
            'invalid input quads (...): Singular matrix
        """
    try:
        transform_matrix = create_perspective_transform_matrix(src, dst)
        error = None
    except np.linalg.LinAlgError as e:
        transform_matrix = np.identity(3, dtype=np.float)
        error = "invalid input quads (%s and %s): %s" %(src, dst, e)
        error = error.replace("\n", "")

    to_eval = "def perspective_transform(%s):\n" %(
        splat_args and "*pt" or "pt",
    )
    to_eval += "  res = np.dot(transform_matrix, ((pt[0], ), (pt[1], ), (1, )))\n"
    to_eval += "  res = res / res[2]\n"
    if round:
        to_eval += "  return (int(round(res[0][0])), int(round(res[1][0])))\n"
    else:
        to_eval += "  return (res[0][0], res[1][0])\n"
    locals = {
        "transform_matrix": transform_matrix,
    }
    locals.update(globals())
    exec(to_eval in locals, locals)
    res = locals["perspective_transform"]
    res.matrix = transform_matrix
    res.error = error
    return res

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
        background: tuple[int, int] | None = None,
        upscale: bool = True,
        extra_out: str | BinaryIO | None = None,
        extra_name: str | None = None,
        frames: list[int] = [1,2,3],
        speed: int = 200
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
        img_width_raw = len(grid[0]) * constants.DEFAULT_SPRITE_SIZE
        img_height_raw =  len(grid) * constants.DEFAULT_SPRITE_SIZE
        i = 0
        if before_image:
            for frame in ImageSequence.Iterator(before_image):
                i += 1
                im = frame.convert('RGBA').resize((frame.width//2,frame.height//2),Image.NEAREST)
                newImage = Image.new('RGBA', (img_width_raw, img_height_raw), (255, 255, 255, 0))
                newImage.paste(im, (0, 0))
                frame = newImage
                imgs.append(frame)
        
        # This is appropriate padding, no sprites can go beyond it
        padding = constants.DEFAULT_SPRITE_SIZE
        for frame in frames:
            width = len(grid[0])
            height = len(grid)
            img_width = width * constants.DEFAULT_SPRITE_SIZE + 2 * padding 
            img_height = height * constants.DEFAULT_SPRITE_SIZE + 2 * padding

            if images and image_source is not None:
                img = Image.new("RGBA", (img_width, img_height))
                # for loop in case multiple background images are used (i.e. baba's world map)
                for image in images:
                    overlap = Image.open(f"data/images/{image_source}/{image}_{frame}.png")
                    img.paste(overlap, (padding, padding), mask=overlap)
            # bg color
            elif background is not None:
                palette_color = palette_img.getpixel(background)
                img = Image.new("RGBA", (img_width, img_height), color=palette_color)
            # neither
            else: 
                img = Image.new("RGBA", (img_width, img_height))
            imgs.append(img)
        
        # keeping track of the amount of padding we can slice off
        pad_r=pad_u=pad_l=pad_d=0
        width = len(grid[0])
        height = len(grid)
        for y, row in enumerate(grid):
            for x, stack in enumerate(row):
                for tile in stack:
                    if tile.frames is None:
                        continue
                    tframes = []
                    for f in frames:
                        tframes.append(tile.frames[f-1])
                    for frame, sprite in enumerate(tframes[:len(frames)]):
                        x_offset = ((sprite.width - constants.DEFAULT_SPRITE_SIZE ) // 2 )
                        y_offset = ((sprite.height - constants.DEFAULT_SPRITE_SIZE ) // 2 )
                        x_offset_disp = ((sprite.width - constants.DEFAULT_SPRITE_SIZE ) // 2 ) + tile.displace[0]
                        y_offset_disp = ((sprite.height - constants.DEFAULT_SPRITE_SIZE ) // 2 ) + tile.displace[1]
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
                            imgs[frame+i].paste(
                                sprite, 
                                (
                                    x * constants.DEFAULT_SPRITE_SIZE + padding - x_offset_disp,
                                    y * constants.DEFAULT_SPRITE_SIZE + padding - y_offset_disp
                                ), 
                                alpha
                            ) 
                        elif tile.cut_alpha:
                            if background is not None: 
                                imgs[frame+i].paste(
                                    Image.new("RGBA", (sprite.width, sprite.height), palette_img.getpixel(background)), 
                                    (
                                        x * constants.DEFAULT_SPRITE_SIZE + padding - x_offset_disp,
                                        y * constants.DEFAULT_SPRITE_SIZE + padding - y_offset_disp
                                    ), 
                                    alpha
                                )
                            else:
                                imgs[frame+i].paste(
                                    Image.new("RGBA", (sprite.width, sprite.height)), 
                                    (
                                        x * constants.DEFAULT_SPRITE_SIZE + padding - x_offset_disp,
                                        y * constants.DEFAULT_SPRITE_SIZE + padding - y_offset_disp
                                    ), 
                                    alpha
                                )
                        else:
                            imgs[frame+i].paste(
                                sprite, 
                                (
                                    x * constants.DEFAULT_SPRITE_SIZE + padding - x_offset_disp,
                                    y * constants.DEFAULT_SPRITE_SIZE + padding - y_offset_disp
                                ), 
                                mask=sprite
                            )
        
        outs = []
        n = 0
        for img in imgs:
            n += 1
            if n > i:
               img = img.crop((padding - pad_l, padding - pad_u, img.width - padding + pad_r, img.height - padding + pad_d))
            if upscale:
                img = img.resize((2 * img.width, 2 * img.height), resample=Image.NEAREST)
            outs.append(img)

        self.save_frames(
            outs,
            out,
            speed=speed,
            extra_out=extra_out,
            extra_name=extra_name,
            
        )

    async def render_full_tile(self,
        tile: FullTile,
        *,
        position: tuple[int, int],
        palette_img: Image.Image,
        random_animations: bool = False,
        sprite_cache: dict[str, Image.Image]
    ) -> ReadyTile:
        '''woohoo'''
        if tile.empty:
            return ReadyTile(None)
        out = []
        x, y = position
        for frame in range(3):
            wobble = (11 * x + 13 * y + frame) % 3 if random_animations else frame
            if tile.freeze:
                wobble=0
            if tile.custom:
                sprite = await self.generate_sprite(
                    tile.name,
                    style=tile.custom_style or "noun",
                    direction=tile.custom_direction,
                    meta_level=tile.meta_level,
                    wobble=wobble,
                    filters=tile.filters,
                    blur=tile.blur_radius,
                    angle=tile.angle,
                    glitch=tile.glitch,
                    scale=tile.scale,
                    warp=tile.warp,
                    neon=tile.neon,
                    opacity=tile.opacity,
                    pixelate=tile.pixelate
                )
            else:
                if tile.name in ("icon",):
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}.png"
                elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}_1.png"
                elif tile.name == "default":
                    path = f"data/sprites/{constants.BABA_WORLD}/default_{wobble + 1}.png"
                elif tile.variant_number == -1: #this is so :(
                    source, sprite_name = tile.sprite
                    path = f"data/sprites/vanilla/error_0_{wobble + 1}.png"
                else:
                    source, sprite_name = tile.sprite
                    path = f"data/sprites/{source}/{sprite_name}_{tile.variant_number}_{wobble + 1}.png"
                sprite = cached_open(path, cache=sprite_cache, fn=Image.open).convert("RGBA")
                
                sprite = await self.apply_options_name(
                    tile.name,
                    sprite,
                    style=tile.custom_style,
                    direction=tile.custom_direction,
                    meta_level=tile.meta_level,
                    wobble=wobble,
                    filters=tile.filters,
                    blur=tile.blur_radius,
                    angle=tile.angle,
                    glitch=tile.glitch,
                    scale=tile.scale,
                    warp=tile.warp,
                    neon=tile.neon,
                    opacity=tile.opacity,
                    pixelate=tile.pixelate
                )
            # Color conversion
            if tile.palette=="":
                rgb = tile.color_rgb if tile.color_rgb is not None else palette_img.getpixel(tile.color_index)
            else:
                rgb = tile.color_rgb if tile.color_rgb is not None else Image.open(f"data/palettes/{tile.palette}.png").convert("RGB").getpixel(tile.color_index)
            sprite = self.recolor(sprite, rgb)
            if tile.negative:
                inverted = 255-np.array(sprite)
                inverted[:,:,3] = 255-inverted[:,:,3]
                sprite = Image.fromarray(abs(inverted))
            out.append(sprite)
        f0, f1, f2 = out
        return ReadyTile((f0, f1, f2), tile.cut_alpha, tile.mask_alpha, tile.displace, tile.scale)

    async def render_full_tiles(
        self,
        grid: list[list[list[FullTile]]],
        *,
        palette: str = "default",
        random_animations: bool = False
    ) -> list[list[list[ReadyTile]]]:
        '''Final individual tile processing step'''
        sprite_cache = {}
        palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")

        a = []
        for y, row in enumerate(grid):
            b = []
            for x, stack in enumerate(row):
                b.append([
                    await self.render_full_tile(
                        tile,
                        position=(x, y),
                        palette_img=palette_img,
                        random_animations=random_animations,
                        sprite_cache=sprite_cache
                    )
                    for tile in stack
                ])
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
        filters: list[str],
        blur: int,
        angle: int,
        glitch: int,
        scale: tuple[float,float],
        warp: tuple[tuple[float,float],tuple[float,float],tuple[float,float],tuple[float,float]],
        neon: float,
        opacity: float,
        pixelate: int
    ) -> Image.Image:
        '''Generates a custom text sprite'''
        text = text[5:]
        raw = text.replace("/", "")
        newline_count = text.count("/")

        if seed is None:
            seed = random.randint(0, 8 ** len(raw))
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
                index = len(raw) - len(raw) // 2
    
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

        # Wxpand widths where possible
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
        return self.apply_options(
            sprite, 
            original_style="noun",
            style=style,
            original_direction=None,
            direction=direction,
            meta_level=meta_level,
            wobble=wobble,
            filters=filters,
            name="text"+raw,
            blur=blur,
            angle=angle,
            glitch=glitch,
            scale=scale,
            warp=warp,
            neon=neon,
            opacity=opacity,
            pixelate=pixelate
        )

    async def apply_options_name(
        self,
        name: str,
        sprite: Image.Image,
        *,
        style: str | None,
        direction: int | None,
        meta_level: int,
        wobble: int,
        filters: list[str],
        blur: int,
        angle: int,
        glitch: int,
        scale: tuple[float,float],
        warp: tuple[tuple[float,float],tuple[float,float],tuple[float,float],tuple[float,float]],
        neon: float,
        opacity: float,
        pixelate: int
    ) -> Image.Image:
        '''Takes an image, taking tile data from its name, and applies the given options to it.'''
        tile_data = await self.bot.db.tile(name)
        assert tile_data is not None
        original_style = constants.TEXT_TYPES[tile_data.text_type]
        original_direction = tile_data.text_direction
        if style is None:
            style = original_style
        try:
            return self.apply_options(
                sprite,
                original_style=original_style,
                style=style,
                original_direction=original_direction,
                direction=direction,
                meta_level=meta_level,
                wobble=wobble,
                filters=filters,
                name=name,
                blur=blur,
                angle=angle,
                glitch=glitch,
                scale=scale,
                warp=warp,
                neon=neon,
                opacity=opacity,
                pixelate=pixelate
            )
        except ValueError as e:
            size = e.args[0]
            raise errors.BadTileProperty(name, size)

    def apply_options(
        self,
        sprite: Image.Image,
        *, 
        original_style: str,
        style: str,
        original_direction: int | None,
        direction: int | None,
        meta_level: int,
        wobble: int,
        filters: list[str],
        name: str,
        blur: int,
        angle: float,
        glitch: int,
        scale: tuple[float,float],
        warp: tuple[tuple[float,float],tuple[float,float],tuple[float,float],tuple[float,float]],
        neon: float,
        opacity: float,
        pixelate: int
    ):
        '''Takes an image, with or without a plate, and applies the given options to it.'''
        if "face" in filters:
            colors = []
            for x in range(sprite.size[0]):
                for y in range(sprite.size[1]):
                    if sprite.getchannel("A").getpixel((x,y)) > 0 :
                        colors += [sprite.getpixel((x,y))]
            colors = [item for items, c in Counter(colors).most_common() for item in [items] * c]
            for x in range(sprite.size[0]):
                for y in range(sprite.size[1]):
                    r1,g1,b1,a1 = sprite.getpixel((x,y))
                    r2,g2,b2,a2 = colors[-1]
                    if r1 // 8 != r2 // 8 or g1 // 8 != g2 // 8 or b1 // 8 != b2 // 8 or a1 // 8 != a2 // 8 :
                        sprite.putpixel((x,y),(0,0,0,0))
                    else:
                        sprite.putpixel((x,y),(r1,g1,b1,a1))
        if scale != (1,1):
            wid = int(max(sprite.width*scale[0],sprite.width))
            hgt = int(max(sprite.height*scale[1],sprite.height))
            sprite = sprite.resize((math.floor(sprite.width*scale[0]),math.floor(sprite.height*scale[1])), resample=Image.NEAREST) 
            if (wid,hgt) != (sprite.width,sprite.height):
                im = Image.new('RGBA',(wid,hgt),(0,0,0,0))
                im.paste(sprite,(wid-int(sprite.width*(2-scale[0])), hgt-int(sprite.height*(2-scale[1]))))
                sprite = im
        if pixelate > 1:
            wid,hgt = sprite.size
            sprite = sprite.resize((math.floor(sprite.width/pixelate),math.floor(sprite.height/pixelate)), resample=Image.NEAREST)
            sprite = sprite.resize((wid,hgt), resample=Image.NEAREST)
        if glitch != 0:
            randlist = []
            width, height = sprite.size
            widthold, heightold = sprite.size
            width *= 3
            height *= 3
            for _ in range(glitch):
                a = random.randint(-180,180)
                sprite = sprite.rotate(a, expand=True)
                sprite = sprite.crop(((sprite.width - width)//2, (sprite.height - height)//2, (sprite.width + width)//2, (sprite.height + height)//2))
                randlist.append(a)
            sprite = sprite.rotate(-1*sum(randlist))
            sprite = sprite.crop(((sprite.width - widthold)//2, (sprite.height - heightold)//2, (sprite.width + widthold)//2, (sprite.height + heightold)//2))
            
        if meta_level != 0 or original_style != style or (style == "property" and original_direction != direction):
            if original_style == "property":
                # box: position of upper-left coordinate of "inner text" in the larger text tile
                plate, box = self.bot.db.plate(original_direction, wobble)
                plate_alpha = ImageChops.invert(plate.getchannel("A"))
                sprite_alpha = ImageChops.invert(sprite.getchannel("A"))
                alpha = ImageChops.subtract(sprite_alpha, plate_alpha)
                sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
                sprite = sprite.crop((box[0], box[1], constants.DEFAULT_SPRITE_SIZE + box[0], constants.DEFAULT_SPRITE_SIZE + box[1]))
            if style == "property":
                plate, box = self.bot.db.plate(direction, wobble)
                if scale != (1,1):
                    plate = plate.resize((int(math.floor(plate.width*scale[0])),int(math.floor(plate.height*scale[1]))), resample=Image.NEAREST) 
                plate = self.make_meta(plate, meta_level)
                plate_alpha = plate.getchannel("A")
                sprite_alpha = sprite.getchannel("A").crop(
                    (-meta_level, -meta_level, sprite.width + meta_level, sprite.height + meta_level)
                )
                sprite_alpha = sprite_alpha.crop(
                    (-box[0], -box[0], sprite_alpha.width + box[0], sprite_alpha.height + box[1])
                )
                if meta_level % 2 == 0:
                    alpha = ImageChops.subtract(plate_alpha, sprite_alpha)
                else:
                    alpha = ImageChops.add(plate_alpha, sprite_alpha)
                sprite = Image.merge("RGBA", (alpha, alpha, alpha, alpha))
            else:
                sprite = self.make_meta(sprite, meta_level)
        def scan(spritenumpyscan):
            for i in range(len(spritenumpyscan)):
                if (i%2)==1:
                    spritenumpyscan[i]=0
            return spritenumpyscan
        for filter in filters:
            if filter == "flipx":
                sprite = ImageOps.mirror(sprite)
            if filter == "flipy":
                sprite = ImageOps.flip(sprite)
            if filter == "blank":
                sprite = Image.composite(Image.new("RGBA", (sprite.width, sprite.height), (255,255,255,255)),sprite,sprite)
            if filter == "scanx":
                spritenumpyscan = np.array(sprite)
                sprite = Image.fromarray(scan(spritenumpyscan))
            if filter == "scany":
                spritenumpyscan = np.array(sprite).swapaxes(0,1)
                sprite = Image.fromarray(scan(spritenumpyscan).swapaxes(0,1))
            if filter == "invert":
                inverted = 255-np.array(sprite)
                inverted[:,:,3] = 255-inverted[:,:,3]
                sprite = Image.fromarray(abs(inverted))
            if filter == "fisheye":
                spritenumpyscan = np.array(sprite)
                spritenumpyscan = fish.fish(spritenumpyscan,0.5)
                sprite = Image.fromarray(spritenumpyscan)
        if opacity < 1:
            r,g,b,a = sprite.split()
            sprite = Image.merge('RGBA',(r,g,b,a.point(lambda i: i * opacity)))
        if neon > 1:
            def clamp(mini, num, maxi):
                return sorted((mini, num, maxi))[1]
            sprite2 = copy.deepcopy(sprite)
            for x in range(sprite.size[0]):
                for y in range(sprite.size[1]):
                    if sprite.getchannel("A").getpixel((x,y)) > 0 :
                        neighbors = 0
                        for xo,yo in [[1,0],[0,1],[-1,0],[0,-1]]:
                            if name.startswith("text_"):
                                if (sprite.size[0]-1 < x+xo) or (x+xo < 0) or (sprite.size[1]-1 < y+yo) or (y+yo < 0):
                                        a = (0,0,0,0)
                                else:
                                    try:
                                        a = sprite.getpixel((x+xo,y+yo))
                                    except:
                                        a = (0,0,0,0)
                            else:
                                a = sprite.getpixel((clamp(0,x+xo,sprite.size[0]-1),clamp(0,y+yo,sprite.size[1]-1)))
                            b = sprite.getpixel((x,y))
                            neighbors += int(a[0]==b[0] and a[1]==b[1] and a[2]==b[2] and a[3]==b[3])
                        if neighbors==4:
                            r,g,b,a = sprite.getpixel((x,y))
                            a = max(int(round(a / neon)),30)
                            sprite2.putpixel((x,y),(r,g,b,a))
                        neighbors = 0
                        for xo in [-1,0,1]:
                            for yo in [-1,0,1]:
                                if name.startswith("text_"):
                                    if (sprite.size[0]-1 < x+xo) or (x+xo < 0) or (sprite.size[1]-1 < y+yo) or (y+yo < 0):
                                        a = (0,0,0,0)
                                    else:
                                        try:
                                            a = sprite.getpixel((x+xo,y+yo))
                                        except:
                                            a = (0,0,0,0)
                                else:
                                    a = sprite.getpixel((clamp(0,x+xo,sprite.size[0]-1),clamp(0,y+yo,sprite.size[1]-1)))
                                b = sprite.getpixel((x,y))
                                neighbors += int(a[0]==b[0] and a[1]==b[1] and a[2]==b[2] and a[3]==b[3])
                        if neighbors==9:
                            r,g,b,a = sprite2.getpixel((x,y))
                            a = int(round(a / 1.2))
                            sprite2.putpixel((x,y),(r,g,b,a))
            sprite = sprite2
        if warp != ((0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)):
            size2warp = sprite.size
            spritenumpywarp = np.array(sprite)
            widwarp = [-1*min(warp[0][0],warp[3][0],0),max((warp[1][0]+sprite.width),(warp[2][0]+sprite.width),sprite.width)]
            hgtwarp = [-1*min(warp[0][1],warp[1][1],0),max((warp[2][1]+sprite.height),(warp[3][1]+sprite.height),sprite.height)]
            dummywarp = Image.new('RGBA',(math.floor(sum(widwarp)),math.floor(sum(hgtwarp))),(0,0,0,0))
            srcpoints = np.array([[0,0],[sprite.width,0],[sprite.width,sprite.height],[0,sprite.height]])
            dstpoints = np.array([[warp[0][0], warp[0][1]], [(warp[1][0]+sprite.width),warp[1][1]], [(warp[2][0]+sprite.width),(warp[2][1]+sprite.height)], [warp[3][0], (warp[3][1]+sprite.height)]])
            srcpoints = np.float32(srcpoints.tolist())
            dstpoints = np.float32(dstpoints.tolist())
            Mwarp = cv2.getPerspectiveTransform(srcpoints, dstpoints)
            print(dummywarp.size[1::-1])
            warped = cv2.warpPerspective(spritenumpywarp, Mwarp, dsize=dummywarp.size[1::-1], flags = cv2.INTER_NEAREST)
            sprite = Image.fromarray(warped)
        if angle != 0:
            sprite = sprite.rotate(-angle)
        if blur != 0:
            sprite = sprite.filter(ImageFilter.GaussianBlur(radius = blur))
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
