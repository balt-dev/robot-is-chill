from __future__ import annotations

import functools
import glob
import math
import random
import re
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageChops, ImageSequence

from src.tile import ProcessedTile, Tile
from . import filterimage
from .. import constants, errors
from ..types import Variant, Color
from ..utils import cached_open, recolor, composite

if TYPE_CHECKING:
    from ...ROBOT import Bot


def shift_hue(arr, hueshift):
    arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + int(hueshift // 2), 180)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.dstack((rgb, arr_a))


def lock(t, arr, lock, nonzero: bool = False):
    arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
    hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
    if nonzero:
        hsv[..., t][hsv[..., t] != 0] = lock
    else:
        hsv[..., t] = lock
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.dstack((rgb, arr_a))


def grayscale(arr, influence):
    arr = arr.astype(np.float64)
    arr[:, :, 0:3] = (((arr[:, :, 0:3].sum(2) / 3).repeat(3).reshape(
        arr.shape[:2] + (3,))) * influence) + (arr[:, :, 0:3] * (1 - influence))
    return arr.astype(np.uint8)


def alpha_paste(img1, img2, coords, func=None):
    if func is None:
        func = Image.alpha_composite
    imgtemp = Image.new('RGBA', img1.size, (0, 0, 0, 0))
    imgtemp.paste(
        img2,
        coords
    )
    return func(img1, imgtemp)


def delta_e(img1, img2):
    # compute the Euclidean distance with pixels of two images
    return np.sqrt(np.sum((img1 - img2) ** 2, axis=-1))


class Renderer:
    """This class exposes various image rendering methods.

    Some of them require metadata from the bot to function properly.
    """

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.palette_cache = {}
        for path in glob.glob("data/palettes/*.png"):
            with Image.open(path) as im:
                self.palette_cache[Path(path).stem] = im.copy()
        self.overlay_cache = {}
        for path in glob.glob("data/overlays/*.png"):
            with Image.open(path) as im:
                self.overlay_cache[Path(path).stem] = np.array(im.convert("RGBA"))

    async def render(
            self,
            grid: list[list[list[list[ProcessedTile]]]],
            *,
            before_image: Image = None,
            palette: str = "default",
            images: list[str] | None = None,
            image_source: str = constants.BABA_WORLD,
            out: str | BinaryIO = "target/renders/render.gif",
            background: tuple[int] | None = None,
            upscale: int = 2,
            extra_out: str | BinaryIO | None = None,
            extra_name: str | None = None,
            frames: list[int] = (1, 2, 3),
            animation: tuple[int, int] = (0, None),
            speed: int = 200,
            crop: tuple[int, int, int, int] | None = None,
            pad: tuple[int, int, int, int] | None = (0, 0, 0, 0),
            image_format: str = 'gif',
            loop: bool = True,
            spacing: int = constants.DEFAULT_SPRITE_SIZE,
            expand: bool = False,
            boomerang: bool = False,
            random_animations: bool = True,
            **_
    ):
        """Takes a list of tile objects and generates a gif with the associated sprites."""
        start_time = time.perf_counter()
        animation, animation_delta = animation  # number of frames per wobble frame, number of frames per timestep
        grid = np.array(grid, dtype=object)
        durations = [speed] * (len(frames) if animation_delta is None else animation_delta) * grid.shape[0]
        if animation:
            frames = np.repeat(frames, animation).tolist()
            frames = (frames * (math.ceil(len(durations) /
                                          animation_delta)))[:len(durations)]

        def get_first_frame(tile):
            for frame in tile.frames:
                if frame is not None:
                    return frame.shape
            else:
                return (0, 0) # Empty tile

        sizes = np.array(
            [get_first_frame(tile)[:2] if not (tile is None or tile.empty) else (0, 0) for tile in grid.flatten()])
        sizes = (sizes.reshape((*grid.shape, 2)) - spacing)
        #                     D   L   Y   X   S
        left = np.max(sizes[:, :, :, 0, 0], (0, 1, 2)) // 2
        top = np.max(sizes[:, :, 0, :, 1], (0, 1, 2)) // 2
        right = np.max(sizes[:, :, :, -1, 0], (0, 1, 2)) // 2
        bottom = np.max(sizes[:, :, -1, :, 1], (0, 1, 2)) // 2
        default_size = (int(sizes.shape[2] * spacing + top + bottom),
                        int(sizes.shape[3] * spacing + left + right))
        steps = np.zeros((grid.shape[0] * len(frames), *default_size, 4), dtype=np.uint8)
        if images and image_source is not None:
            for step in range(steps.shape[0]):
                for frame in frames:
                    img = Image.new("RGBA", default_size)
                    # for loop in case multiple background images are used
                    # (i.e. baba's world map)
                    for image in images:
                        overlap = Image.open(
                            f"data/images/{image_source}/{image}_{frame}.png")
                        img.paste(overlap, (0, 0), mask=overlap)
                    steps[step * len(frames) + frame - 1] = np.asarray(img)
        else:
            if background is not None:
                if len(background) < 4:
                    background = Color.parse(Tile(palette=palette), self.palette_cache, background)
                steps += background
        for t, step in enumerate(grid):
            for z, layer in enumerate(step):
                for y, row in enumerate(layer):
                    for x, tile in enumerate(row):
                        first_frame = get_first_frame(tile)
                        if tile.empty:
                            continue
                        displacement = (x * spacing + (left - (first_frame[1] - spacing) // 2) - tile.displacement[0],
                                        y * spacing + (top  - (first_frame[0] - spacing) // 2) - tile.displacement[1])
                        for frame in frames:
                            wobble = (11 * x + 13 * y + frame - 1) % 3 if random_animations else frame - 1
                            image = tile.frames[wobble]
                            # Cut the pasted tile to fit inside the image
                            a = -max(displacement[1] + first_frame[0] - default_size[0], 0)
                            b = -max(displacement[0] + first_frame[1] - default_size[1], 0)
                            dst_slice = (
                                slice(max(-displacement[1], 0),
                                      a if a < 0 else None),
                                slice(max(-displacement[0], 0),
                                      b if b < 0 else None)
                            )
                            image = image[*dst_slice]
                            if image.size < 1:
                                continue
                            # Get the part of the image to paste on
                            src_slice = (
                                slice(displacement[1], image.shape[0] + displacement[1]),
                                slice(displacement[0], image.shape[1] + displacement[0])
                            )
                            steps[t + wobble, *src_slice] = self.blend(tile.blending, steps[t + wobble, *src_slice], image)
        comp_ovh = time.perf_counter() - start_time
        start_time = time.perf_counter()
        images = []
        for step in steps:
            im = Image.fromarray(step)
            images.append(im.resize((int(im.width * upscale), int(im.height * upscale)), Image.NEAREST))
        self.save_frames(images,
                         out,
                         durations,
                         extra_out=extra_out,
                         extra_name=extra_name,
                         image_format=image_format,
                         loop=loop,
                         boomerang=boomerang,
                         background=background is not None)
        return comp_ovh, time.perf_counter() - start_time

    def blend(self, mode, src, dst) -> np.ndarray:
        if mode not in ("mask", "cut"):
            out_a = (src[..., 3] + dst[..., 3] * (1 - src[..., 3] / 255)).astype(np.uint8)
            if mode == "add":
                out_rgb = np.clip(src[..., :3].astype(np.int16) + dst[..., :3].astype(np.int16), 0, 255).astype(np.uint8)
            else:
                dst_alpha = dst[..., 3].astype(float) / 255
                dst_alpha = dst_alpha[:, :, np.newaxis]
                out_rgb = (1.0 - dst_alpha) * src[..., :3] + dst_alpha * dst[..., :3]
            return np.dstack((out_rgb, out_a[..., np.newaxis]))

    async def render_full_frame(self,
                                tile: Tile,
                                frame: int,
                                raw_sprite_cache: dict[str, Image],
                                gscale: float,
                                x: int,
                                y: int,
                                ) -> Image.Image:
        if tile.custom and type(tile.sprite) == tuple:
            sprite = await self.generate_sprite(
                tile,
                style=tile.style or (
                    "noun" if len(tile.name) < 1 else "letter"),
                wobble=frame,
                position=(x, y),
                gscale=gscale
            )
        else:
            if isinstance(tile.sprite, np.ndarray):
                sprite = tile.sprite[(tile.frame * 3) + frame]
            else:
                path_fallback = None
                if tile.name == "icon":
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}.png"
                elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                    path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}_1.png"
                elif tile.name == "default":
                    path = f"data/sprites/{constants.BABA_WORLD}/default_{frame + 1}.png"
                else:
                    if tile.frame == -1:
                        source, sprite_name = tile.sprite
                        path = f"data/sprites/vanilla/error_0_{frame + 1}.png"
                    else:
                        source, sprite_name = tile.sprite
                        path = f"data/sprites/{source}/{sprite_name}_{tile.frame}_{frame + 1}.png"
                    try:
                        path_fallback = f"data/sprites/{source}/{sprite_name}_{tile.fallback_frame}_{frame + 1}.png"
                    except BaseException:
                        path_fallback = None
                try:
                    sprite = cached_open(
                        path, cache=raw_sprite_cache, fn=Image.open).convert("RGBA")
                except FileNotFoundError:
                    try:
                        assert path_fallback is not None
                        sprite = cached_open(
                            path_fallback,
                            cache=raw_sprite_cache,
                            fn=Image.open).convert("RGBA")
                    except (FileNotFoundError, AssertionError):
                        raise AssertionError(f'The tile `{tile.name}:{tile.frame}` was found, but the files '
                                             f'don\'t exist for it.')
                sprite = np.array(sprite)
            sprite = cv2.resize(sprite, (int(sprite.shape[1] * gscale), int(sprite.shape[0] * gscale)),
                                interpolation=cv2.INTER_NEAREST)
        return await self.apply_options_name(
            tile,
            sprite,
            frame
        )

    async def render_full_tile(self,
                               tile: Tile,
                               *,
                               position: tuple[int, int],
                               raw_sprite_cache: dict[str, Image.Image],
                               gscale: float = 1,
                               tile_cache=None,
                               frames: tuple[int] = (1, 2, 3),
                               random_animations: bool = True
                               ) -> tuple[ProcessedTile, int]:
        """woohoo."""
        if tile_cache is None:
            tile_cache = {}
        final_tile = ProcessedTile()
        if tile.empty:
            return final_tile, 0
        final_tile.empty = False
        final_tile.displacement = tile.displacement
        final_tile.blending = tile.blending
        x, y = position

        rendered_frames = 0
        tile_hash = hash((tile, gscale))
        cached = tile_hash in tile_cache.keys()
        if cached:
            final_tile = tile_cache[tile_hash]
        done_frames = [frame is not None for frame in final_tile.frames]
        frame_range = tuple(set(frames)) if tile.wobble is None else (tile.wobble,)
        for frame in frame_range:
            frame -= 1
            wobble = (11 * x + 13 * y + frame) % 3 if random_animations else frame
            if not done_frames[wobble]:
                final_tile.frames[wobble] = await self.render_full_frame(tile, wobble, raw_sprite_cache, gscale, x, y)
                rendered_frames += 1
        if not cached:
            tile_cache[tile_hash] = final_tile
        return final_tile, rendered_frames

    async def render_full_tiles(self, grid: list[list[list[list[Tile]]]], *, random_animations: bool = False,
                                gscale: float = 2, frames: tuple[int] = (1, 2, 3)) -> tuple[
        list[list[list[list[ProcessedTile]]]], int, int, int]:
        """Final individual tile processing step."""
        sprite_cache = {}
        tile_cache = {}
        rendered_frames = 0
        d = []
        render_overhead = time.perf_counter()
        for timestep in grid:
            a = []
            for layer in timestep:
                b = []
                for y, row in enumerate(layer):
                    c = []
                    for x, tile in enumerate(row):
                        processed_tile, new_frames = await self.render_full_tile(
                            tile,
                            position=(x, y),
                            random_animations=random_animations,
                            gscale=gscale,
                            raw_sprite_cache=sprite_cache,
                            tile_cache=tile_cache,
                            frames=frames
                        )
                        rendered_frames += new_frames
                        for variant in tile.variants["post"]:
                            variant.apply(processed_tile, renderer=self)
                        c.append(
                            processed_tile
                        )
                    b.append(c)
                a.append(b)
            d.append(a)
        return d, len(tile_cache), rendered_frames, time.perf_counter() - render_overhead

    async def generate_sprite(
            self,
            tile: Tile,
            *,
            style: str,
            wobble: int,
            seed: int | None = None,
            gscale: float,
            position: tuple[int, int]
    ) -> np.ndarray:
        """Generates a custom text sprite."""
        text = tile.name[5:].lower()
        raw = text.replace("/", "")
        newline_count = text.count("/")
        assert len(text) <= 64, 'Text has a maximum length of `64` characters.'
        if seed is None:
            seed = int((7 + position[0]) / (3 + position[1]) * 100000000)
        seed_digits = [(seed >> 8 * i) | 0b11111111 for i in range(len(raw))]
        # Get mode and split status
        if newline_count >= 1:
            fixed = True
            mode = "small"
            indices = []
            offset = 0
            for match in re.finditer("/", text):
                indices.append(match.start() + offset)
                offset -= 1
        else:
            fixed = False
            mode = "big"
            indices = []
            if len(raw) >= 4:
                mode = "small"
                indices = [len(raw) - math.ceil(len(raw) / 2)]

        indices.insert(0, 0)
        indices.append(len(raw))  # can't use -1 here because of a range() later on

        if style == "letter":
            if mode == "big":
                mode = "letter"

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

        max_width = constants.DEFAULT_SPRITE_SIZE
        old_index = 0
        for index in indices:
            max_width = max(max_width, sum(widths[old_index:index]))
            old_index = index

        def check_or_adjust(widths: list[int], indices: list[int]) -> list[int]:
            """Is the arrangement valid?"""
            if mode == "small":
                if not fixed:
                    for i in range(1, len(indices) - 1):
                        width_distance = (
                                sum(widths[indices[i]:indices[i + 1]]) - sum(widths[indices[i - 1]:indices[i]]))
                        old_wd = width_distance
                        debug_flag = 0
                        while old_wd > width_distance:
                            debug_flag += 1
                            indices[i] -= 1
                            new_width_distance = (
                                    sum(widths[indices[i]:indices[i + 1]]) - sum(widths[indices[i - 1]:indices[i]]))
                            if new_width_distance > width_distance:
                                indices[i] += 2
                            assert debug_flag > 200, "Ran into an infinite loop while trying to create a text! " \
                                                     "This shouldn't happen."
            return indices

        # Check if the arrangement is valid with minimum sizes
        # If allowed, shift the index to make the arrangement valid
        indices = check_or_adjust(widths, indices)

        # Expand widths where possible
        stable = [False for _ in range(len(widths))]
        while not all(stable):
            old_width, i = min((w, i)
                               for i, w in enumerate(widths) if not stable[i])
            try:
                new_width = width_greater_than(raw[i], old_width)
                a = max([0] if not len(max_list := [j for j in indices if i >= j]) else max_list)
                b = min([len(indices)] if not len(min_list := [j for j in indices if i < j]) else min_list)
                temp_widths = widths[:i] + [old_width + (new_width - old_width) * (b - a)] + widths[i:]  # BAD
                assert sum(temp_widths[a:b]) <= constants.DEFAULT_SPRITE_SIZE
            except (KeyError, AssertionError):
                stable[i] = True
                continue
            widths[i] = new_width
            try:
                indices = check_or_adjust(widths, indices)
            except errors.CustomTextTooLong:
                widths[i] = old_width
                stable[i] = True

        # Arrangement is now the widest it can be
        # Kerning: try for 1 pixel between sprites, and rest to the edges
        gaps: list[int] = []
        bounds: list[tuple(int, int)] = list(zip(indices[:-1], indices[1:]))
        if mode == "small":
            rows = [widths[a:b] for a, b in bounds]
        else:
            rows = [widths[:]]
        for row in rows:
            space = max_width - sum(row)
            # Extra -1 is here to not give kerning space outside the
            # left/rightmost char
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

        size = (max(max(sum(row) for row in rows),
                    constants.DEFAULT_SPRITE_SIZE),
                (max(len(rows), 2) * constants.DEFAULT_SPRITE_SIZE) // 2)
        sprite = Image.new("L",
                           (max(max(sum(row) for row in rows),
                                constants.DEFAULT_SPRITE_SIZE),
                            (max(len(rows), 2) * constants.DEFAULT_SPRITE_SIZE) // 2))
        if mode == "small":
            for j, (a, b) in enumerate(bounds):
                x = gaps[a]
                y_center = ((constants.DEFAULT_SPRITE_SIZE // 2) * j) + (constants.DEFAULT_SPRITE_SIZE // 4)
                for i in range(a, b):
                    letter = letters[i]
                    y_top = y_center - letter.height // 2
                    sprite.paste(letter, (x, y_top), mask=letter)
                    x += widths[i]
                    if i != b - 1:
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
        sprite = sprite.resize(
            (int(sprite.width * gscale), int(sprite.height * gscale)), Image.NEAREST)
        return self.apply_options(
            tile,
            np.array(sprite),
            wobble
        )

    async def apply_options_name(
            self,
            tile: Tile,
            sprite: np.ndarray,
            wobble: int
    ) -> Image.Image:
        """Takes an image, taking tile data from its name, and applies the
        given options to it."""
        try:
            return self.apply_options(
                tile,
                sprite,
                wobble
            )
        except ValueError as e:
            size = e.args[0]
            raise errors.BadTileProperty(tile.name, size)

    def apply_options(
            self,
            tile: Tile,
            sprite: np.ndarray,
            wobble: int,
            seed: int | None = None
    ):
        random.seed(seed)
        for variant in tile.variants["sprite"]:
            sprite = variant.apply(sprite, tile=tile, wobble=wobble, renderer=self)  # NOUN/PROP ARE ANNOYING
        return sprite

    def save_frames(
            self,
            images: list[Image.Image],
            out: str | BinaryIO,
            durations: list[int],
            extra_out: str | BinaryIO | None = None,
            extra_name: str = 'render',
            image_format: str = 'gif',
            loop: bool = True,
            boomerang: bool = False,
            background: bool = False
    ) -> None:
        """Saves the images as a gif to the given file or buffer.

        If a buffer, this also conveniently seeks to the start of the
        buffer. If extra_out is provided, the frames are also saved as a
        zip file there.
        """
        if boomerang and len(images) > 2:
            images += images[-2:0:-1]
            durations += durations[-2:0:-1]
        if image_format == 'gif':
            if not background:
                for i, im in enumerate(images):
                    np_im = np.array(im.convert("RGBA"))
                    colors = np.unique(np_im.reshape(-1, 4),
                                       axis=0)
                    colors = [0, 0, 0] + colors[colors[:, 3]
                                                != 0][:254, :3].flatten().tolist()
                    dummy = Image.new('P', (16, 16))
                    dummy.putpalette(colors)
                    images[i] = im.convert('RGB').quantize(
                        palette=dummy, dither=0)
            kwargs = {
                'format': "GIF",
                'interlace': True,
                'save_all': True,
                'append_images': images[1:],
                'loop': 0,
                'duration': durations,
                'disposal': 2,  # Frames don't overlap
                'background': 0,
                'transparency': 0,
                'optimize': False
            }
            if not loop:
                del kwargs['loop']
            if background:
                del kwargs['transparency']
                del kwargs['background']
                del kwargs['disposal']
            images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'png':
            kwargs = {
                'format': "PNG",
                'save_all': True,
                'append_images': images,
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            images[0].save(
                out,
                **kwargs
            )
        if not isinstance(out, str):
            out.seek(0)
        if extra_name is None:
            extra_name = 'render'
        if extra_out is not None:
            file = zipfile.PyZipFile(extra_out, "x")
            for i, img in enumerate(images):
                buffer = BytesIO()
                img.save(buffer, "PNG")
                file.writestr(
                    f"{extra_name}_{i // 3}_{(i % 3) + 1}.png",
                    buffer.getvalue())
            file.close()


async def setup(bot: Bot):
    bot.renderer = Renderer(bot)
