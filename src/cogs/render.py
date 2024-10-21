from __future__ import annotations

import asyncio
import functools
import glob
import math
import random
import re
import struct
import sys
import time
import traceback
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
import PIL.ImageFont as ImageFont

from src.tile import ProcessedTile, Tile
from .. import constants, errors
from ..types import Color, RenderContext
from ..utils import cached_open

try:
    FONT = ImageFont.truetype("data/fonts/default.ttf")
except OSError:
    pass

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


def get_first_frame(tile):
    for tile_frame in tile.frames:
        if tile_frame is not None:
            return np.array(tile_frame.shape[:2])  # Done for convenience on math operations
    else:
        return np.array((0, 0))  # Empty tile


class Renderer:
    """This class exposes various image rendering methods.

    Some of them require metadata from the bot to function properly.
    """

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.palette_cache = {}
        for path in glob.glob("data/palettes/*.png"):
            with Image.open(path) as im:
                self.palette_cache[Path(path).stem] = im.convert("RGBA").copy()
        self.overlay_cache = {}
        for path in glob.glob("data/overlays/*.png"):
            with Image.open(path) as im:
                self.overlay_cache[Path(path).stem] = np.array(im.convert("RGBA"))

    async def render(
            self,
            grid: list[list[list[list[ProcessedTile]]]],
            ctx: RenderContext
    ):
        """Takes a list of tile objects and generates a gif with the associated sprites."""
        start_time = time.perf_counter()
        if ctx.animation is not None:
            animation_wobble, animation_timestep = ctx.animation
        else:
            animation_wobble, animation_timestep = 1, len(
                ctx.frames)  # number of frames per wobble frame, number of frames per timestep
        grid = np.array(grid, dtype=object)
        durations = [ctx.speed for _ in range(animation_timestep * grid.shape[0] + len(ctx.before_images))]
        frames = np.repeat(ctx.frames, animation_wobble).tolist()
        frames = (frames * (math.ceil(len(durations) / animation_timestep)))
        sizes = np.array(
            [get_first_frame(tile)[:2] if not (tile is None or tile.empty) else (0, 0) for tile in grid.flatten()])
        sizes = sizes.reshape((*grid.shape, 2))
        assert sizes.size > 0, "The render must have at least one tile in it."
        assert len(ctx.sign_texts) <= constants.MAX_SIGN_TEXTS or ctx._no_sign_limit, \
            f"Too many sign texts! The limit is `{constants.MAX_SIGN_TEXTS}`, while you have `{len(ctx.sign_texts)}`."
        if len(ctx.sign_texts):
            for i, sign_text in enumerate(ctx.sign_texts):
                size = int(
                    ctx.spacing * (ctx.upscale / 2) * sign_text.size * constants.FONT_MULTIPLIERS.get(sign_text.font,
                                                                                                      1))
                assert size <= constants.DEFAULT_SPRITE_SIZE * 2, f"Font size of `{size}` is too large! The maximum is `{constants.DEFAULT_SPRITE_SIZE * 2}`."
                if sign_text.font is not None:
                    ctx.sign_texts[i].font = ImageFont.truetype(f"data/fonts/{sign_text.font}.ttf", size=size)
                else:
                    ctx.sign_texts[i].font = FONT.font_variant(size=size)
        if ctx.cropped:
            left = right = top = bottom = 0
        else:
            left_influence = np.arange(0, -ctx.spacing * sizes.shape[3], -ctx.spacing) * 2
            top_influence = np.arange(0, -ctx.spacing * sizes.shape[2], -ctx.spacing) * 2
            right_influence = np.arange(-ctx.spacing * (sizes.shape[3] - 1), ctx.spacing, ctx.spacing) * 2
            bottom_influence = np.arange(-ctx.spacing * (sizes.shape[2] - 1), ctx.spacing, ctx.spacing) * 2
            left_sizes = sizes[..., 1] + left_influence.reshape((1, 1, 1, -1))
            top_sizes = sizes[..., 0] + top_influence.reshape((1, 1, -1, 1))
            right_sizes = sizes[..., 1] + right_influence.reshape((1, 1, 1, -1))
            bottom_sizes = sizes[..., 0] + bottom_influence.reshape((1, 1, -1, 1))
            left = max((np.max(left_sizes - ctx.spacing) // 2) + ctx.pad[0], 0)
            top = max((np.max(top_sizes - ctx.spacing) // 2) + ctx.pad[1], 0)
            right = max((np.max(right_sizes - ctx.spacing) // 2) + ctx.pad[2], 0)
            bottom = max((np.max(bottom_sizes - ctx.spacing) // 2) + ctx.pad[3], 0)
            if ctx.expand:
                displacements = np.array(
                    [tile.displacement if not (tile is None or tile.empty) else (0, 0)
                     for tile in grid.flatten()])
                displacements = (displacements.reshape((*grid.shape, 2)))
                left_disp = displacements[:, :, :, 0, 0]
                top_disp = displacements[:, :, 0, :, 1]
                right_disp = displacements[:, :, :, -1, 0]
                bottom_disp = displacements[:, :, -1, :, 1]
                left += max(max(left_disp.max((0, 1, 2)), left_disp.min((0, 1, 2)), key=abs), 0)
                top += max(max(top_disp.max((0, 1, 2)), top_disp.min((0, 1, 2)), key=abs), 0)
                right -= min(max(right_disp.max((0, 1, 2)), right_disp.min((0, 1, 2)), key=abs), 0)
                bottom -= min(max(bottom_disp.max((0, 1, 2)), bottom_disp.min((0, 1, 2)), key=abs), 0)
        default_size = np.array((int(sizes.shape[2] * ctx.spacing + top + bottom),
                                 int(sizes.shape[3] * ctx.spacing + left + right)))
        true_size = default_size * ctx.upscale
        if not ctx._disable_limit:
            assert all(
                true_size[::-1] <= constants.MAX_IMAGE_SIZE), f"Image of size `{true_size[::-1]}` is larger than the maximum allowed size of `{constants.MAX_IMAGE_SIZE}`!"
        steps = np.zeros(
            (((animation_timestep if animation_wobble else len(frames)) * grid.shape[0]), *default_size, 4),
            dtype=np.uint8)

        if ctx.background_images:
            for f, frame in enumerate(frames):
                img = Image.new("RGBA", tuple(default_size[::-1]))
                # for loop in case multiple background images are used
                # (i.e. baba's world map)
                bg_img: Image.Image = ctx.background_images[(frame - 1) % len(ctx.background_images)].convert("RGBA")
                bg_img = bg_img.resize((bg_img.width // ctx.upscale, bg_img.height // ctx.upscale), Image.NEAREST)
                img.paste(bg_img, (0, 0), mask=bg_img)
                for i in range(animation_wobble):
                    q = i + animation_wobble * f
                    steps[q] = np.array(img)
        for t, step in enumerate(grid):
            for z, layer in enumerate(step):
                for y, row in enumerate(layer):
                    for x, tile in enumerate(row):
                        await asyncio.sleep(0)
                        first_frame = get_first_frame(tile)
                        if tile.empty:
                            continue
                        displacement = (
                        y * ctx.spacing - int((first_frame[0] - ctx.spacing) / 2) + top - tile.displacement[1],
                        x * ctx.spacing - int((first_frame[1] - ctx.spacing) / 2) + left - tile.displacement[0])
                        for i, frame in enumerate(frames[animation_timestep * t:animation_timestep * (t + 1)]):
                            image_index = i + animation_timestep * t
                            wobble = tile.wobble_frames[
                                min(len(tile.wobble_frames) - 1, frame - 1)] if tile.wobble_frames is not None \
                                else (11 * x + 13 * y + frame - 1) % 3 if ctx.random_animations \
                                else frame - 1
                            final_wobble = functools.reduce(lambda a, b: a if b is None else b, tile.frames)
                            image = tile.frames[wobble] if tile.frames[wobble] is not None else final_wobble
                            dslice = (default_size - (first_frame + displacement))
                            dst_slice = (
                                slice(max(-displacement[0], 0), dslice[0] if dslice[0] < 0 else None),
                                slice(max(-displacement[1], 0), dslice[1] if dslice[1] < 0 else None)
                            )
                            image = image[*dst_slice]
                            if image.size < 1:
                                continue
                            # Get the part of the image to paste on
                            src_slice = (
                                slice(max(displacement[0], 0), image.shape[0] + max(displacement[0], 0)),
                                slice(max(displacement[1], 0), image.shape[1] + max(displacement[1], 0))
                            )
                            index = image_index, *src_slice
                            try:
                                steps[index] = self.blend(tile.blending, steps[index], image, tile.keep_alpha)
                            except IndexError:
                                pass  # warnings.warn(f"Couldn't place {tile} at {x}, {y}, {index}")
        background_images = [np.array(image.convert("RGBA")) for image in ctx.before_images]
        l, u, r, d = ctx.crop
        r = r
        d = d
        wobble_range = np.arange(steps.shape[0]) // animation_timestep
        for i, step in enumerate(steps):
            step = step[u:-d if d > 0 else None, l:-r if r > 0 else None]
            if ctx.background is not None:
                if len(ctx.background) < 4:
                    ctx.background = Color.parse(Tile(palette=ctx.palette), self.palette_cache, ctx.background)
                ctx.background = np.array(ctx.background).astype(np.float32)
                step_f = step.astype(np.float32) / 255
                step_f[..., :3] = step_f[..., 3, np.newaxis]
                c = ((1 - step_f) * ctx.background + step_f * step.astype(np.float32))
                step = c.astype(np.uint8)
            step = cv2.resize(
                step,
                (int(step.shape[1] * ctx.upscale), int(step.shape[0] * ctx.upscale)),
                interpolation=cv2.INTER_NEAREST
            )
            if len(ctx.sign_texts):
                anchor_disps = {
                    "l": 0.0,
                    "t": 0.0,
                    "m": 0.5,
                    "r": 1.0,
                    "s": 1.0,
                    "d": 1.0
                }
                im = Image.new("RGBA", step.shape[1::-1])
                draw = ImageDraw(im)
                if ctx.image_format == "gif" and ctx.background is None:
                    draw.fontmode = "1"
                for sign_text in ctx.sign_texts:
                    if wobble_range[i] in range(sign_text.time_start, sign_text.time_end):
                        text = sign_text.text
                        text = re.sub(r"(?<!\\)\\n", "\n", text)
                        text = re.sub(r"\\(.)", r"\1", text)
                        assert len(
                            text) <= constants.MAX_SIGN_TEXT_LENGTH, f"Sign text of length {len(text)} is too long! The maximum is `{constants.MAX_SIGN_TEXT_LENGTH}`."
                        pos = (left + sign_text.xo + (
                                    ctx.spacing * ctx.upscale * (sign_text.x + anchor_disps[sign_text.anchor[0]])),
                               top + sign_text.yo + (ctx.spacing * ctx.upscale * (
                                           sign_text.y + anchor_disps[sign_text.anchor[1]])))
                        draw.multiline_text(pos, text, font=sign_text.font,
                                            align=sign_text.alignment, anchor=sign_text.anchor,
                                            fill=sign_text.color, features=("liga", "dlig", "clig"),
                                            stroke_fill=sign_text.stroke[0], stroke_width=sign_text.stroke[1])
                sign_arr = np.array(im)
                if ctx.image_format == "gif" and ctx.background is None:
                    sign_arr[..., 3][sign_arr[..., 3] > 0] = 255
                step = self.blend("normal", step, sign_arr, True)
            if ctx.image_format == "gif":
                step_a = step[..., 3]
                step = np.multiply(step[..., :3], np.dstack([step_a] * 3).astype(float) / 255,
                                   casting="unsafe").astype(np.uint8)
                true_rgb = step.astype(float) * (step_a.astype(float) / 255).reshape(*step.shape[:2], 1)
                too_dark_mask = np.logical_and(np.all(true_rgb < 8, axis=2), step_a != 0)
                step[too_dark_mask, :3] = 4
                step = np.dstack((step, step_a))
            background_images.append(step)
        comp_ovh = time.perf_counter() - start_time
        start_time = time.perf_counter()
        if self.bot.config["debug"]:
            # Print to thermal printer (I have one for this)
            first = background_images[0]
            ratio = first.shape[0] / first.shape[1]
            first = cv2.resize(
                first,
                (int(min(1 / ratio, 1) * 384), int(min(ratio, 1) * 384)),
                interpolation=cv2.INTER_NEAREST
            )
            image = Image.fromarray(first)
            background = Image.new("RGB", first.shape[1::-1], (0, 0, 0))
            background.paste(image, mask=image.split()[3])
            arr = ~np.array(background.convert("1"))
            (height, width) = arr.shape[:2]
            sys.stdout.buffer.write(b"\x1Dv00")
            bytewidth = (width + 7) // 8
            buf = struct.pack("<2H", bytewidth, height)
            sys.stdout.buffer.write(buf)
            packed = np.packbits(arr, axis=1)
            sys.stdout.buffer.write(packed.tobytes())
            sys.stdout.flush()

        self.save_frames(background_images,
                         ctx.out,
                         durations,
                         extra_out=ctx.extra_out,
                         extra_name=ctx.extra_name,
                         image_format=ctx.image_format,
                         loop=ctx.loop,
                         boomerang=ctx.boomerang,
                         background=ctx.background is not None)
        return comp_ovh, time.perf_counter() - start_time, background_images[0].shape[1::-1]

    def blend(self, mode, src, dst, keep_alpha: bool = True) -> np.ndarray:
        keep_alpha &= mode not in ("mask", "cut")
        if keep_alpha:
            out_a = (src[..., 3] + dst[..., 3] * (1 - src[..., 3] / 255)).astype(np.uint8)
            a, b = src[..., :3].astype(float) / 255, dst[..., :3].astype(
                float) / 255  # This is super convenient actually
        else:
            a, b = src.astype(float) / 255, dst.astype(float) / 255
        if mode == "add":
            c = a + b
        elif mode in ("subtract", "sub"):
            c = a - b
        elif mode == "multiply":
            c = a * b
        elif mode == "divide":
            c = np.clip(a / b, 0.0, 1.0)  # catch divide by 0
        elif mode == "max":
            c = np.maximum(a, b)
        elif mode == "min":
            c = a
            c[dst[..., 3] > 0] = np.minimum(a, b)[dst[..., 3] > 0]
        elif mode == "screen":
            c = 1 - (1 - a) * (1 - b)
        elif mode in ("overlay", "hardlight"):
            if mode == "hardlight":
                a, b = b, a
            c = 1 - (2 * (1 - a) * (1 - b))
            c[np.where(a < 0.5)] = (2 * a * b)[np.where(a < 0.5)]
        elif mode == "softlight":
            c = (1 - a) * a * b + a * (1 - (1 - a) * (1 - b))
        elif mode == "burn":
            c = 1 - ((1 - a) / b)
        elif mode == "dodge":
            c = a / (1 - b)
        elif mode == "normal":
            c = b
        elif mode in ("mask", "cut"):
            c = a
            if mode == "cut":
                b[..., 3] = 1 - b[..., 3]
            c[..., 3] *= b[..., 3]
            c[c[..., 3] == 0] = 0
        else:
            raise AssertionError(f"Blending mode `{mode}` isn't implemented yet.")
        if keep_alpha:
            dst_alpha = dst[..., 3].astype(float) / 255
            dst_alpha = dst_alpha[:, :, np.newaxis]
            c = ((1 - dst_alpha) * a + dst_alpha * c)
            c[out_a == 0] = 0
            return np.dstack((np.clip(c * 255, 0, 255).astype(np.uint8), out_a[..., np.newaxis]))
        return np.clip(c * 255, 0, 255).astype(np.uint8)

    async def render_full_frame(self,
                                tile: Tile,
                                frame: int,
                                raw_sprite_cache: dict[str, Image.Image],
                                x: int,
                                y: int,
                                ctx: RenderContext
                                ) -> Image.Image:
        sprite = None
        if tile.custom:
            if type(tile.sprite) == tuple:
                sprite = await self.generate_sprite(
                    tile,
                    style=tile.style or (
                        "noun" if len(tile.name) < 1 else "letter"),
                    wobble=frame,
                    position=(x, y),
                    ctx=ctx
                )
            elif isinstance(tile.sprite, np.ndarray):
                sprite = tile.sprite[(tile.frame * 3) + frame]
        else:
            source, sprite_name = tile.sprite
            path = f"data/sprites/{source}/{sprite_name}_{tile.frame}_{frame + 1}.png"
            if source in constants.VANILLA_WORLDS:
                if tile.name == "icon":
                    path = f"data/sprites/{source}/{sprite_name}.png"
                elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                    path = f"data/sprites/{source}/{sprite_name}_1.png"
                elif tile.name == "default":
                    path = f"data/sprites/{source}/default_{frame + 1}.png"
            try:
                sprite = cached_open(
                    path, cache=raw_sprite_cache, fn=Image.open).convert("RGBA")
            except (FileNotFoundError, AssertionError):
                raise AssertionError(f'The tile `{tile.name}:{tile.frame}` was found, but the files '
                                         f'don\'t exist for it.\nThis is a bug - please notify the author of the tile.\nSearched path: `{path}`')
            sprite = np.array(sprite)
        sprite = cv2.resize(sprite, (int(sprite.shape[1] * ctx.gscale), int(sprite.shape[0] * ctx.gscale)),
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
                               ctx: RenderContext) -> tuple[ProcessedTile, list[int], bool]:
        """woohoo."""
        final_tile = ProcessedTile()
        if tile.empty:
            return final_tile, [], True
        final_tile.empty = False
        final_tile.name = tile.name
        x, y = position

        rendered_frames = []
        tile_hash = hash(tile)
        cached = tile_hash in ctx.tile_cache.keys()
        if cached:
            final_tile.frames = ctx.tile_cache[tile_hash]
        final_tile.wobble_frames = tile.wobble_frames
        done_frames = [frame is not None for frame in final_tile.frames]
        frame_range = tuple(set(tile.wobble_frames)) if tile.wobble_frames is not None \
            else tuple(set(ctx.frames))
        for frame in frame_range:
            frame -= 1
            wobble = final_tile.wobble_frames[
                min(len(final_tile.wobble_frames) - 1, frame)] if final_tile.wobble_frames is not None \
                else (11 * x + 13 * y + frame) % 3 if ctx.random_animations \
                else frame
            if not done_frames[wobble]:
                final_tile.frames[wobble] = await self.render_full_frame(tile, wobble, ctx.sprite_cache, x, y, ctx)
                rendered_frames.append(wobble)
        if not cached:
            ctx.tile_cache[tile_hash] = final_tile.frames.copy()
        return final_tile, rendered_frames, cached

    async def render_full_tiles(self, grid: list[list[list[list[Tile]]]], ctx: RenderContext) -> tuple[
        list[list[list[list[ProcessedTile]]]], int, int, float]:
        """Final individual tile processing step."""
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
                        processed_tile, new_frames, cached = await self.render_full_tile(
                            tile,
                            position=(x, y),
                            ctx=ctx
                        )
                        rendered_frames += len(new_frames)
                        for variant in tile.variants["post"]:
                            await variant.apply(processed_tile, renderer=self, new_frames=new_frames)
                        c.append(
                            processed_tile
                        )
                    b.append(c)
                a.append(b)
            d.append(a)
        return d, len(ctx.tile_cache), rendered_frames, time.perf_counter() - render_overhead

    async def generate_sprite(
            self,
            tile: Tile,
            *,
            style: str,
            wobble: int,
            seed: int | None = None,
            position: tuple[int, int],
            ctx: RenderContext
    ) -> np.ndarray:
        """Generates a custom text sprite."""
        text = tile.name[5:].lower().replace(" ", "~")
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
                if tile.style != "oneline":
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
            traceback.print_exc()
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
                SELECT sprite_{int(wobble)} FROM letters
                WHERE char == ? AND mode == ? AND width == ?
                ''',
                c, mode, width
            )
            options = list(l_rows)
            letter_sprite, *_ = *options[seed_digit % len(options)],
            buf = BytesIO(letter_sprite)
            letters.append(Image.open(buf))

        sprite = Image.new("L",
                           (max(max(sum(row) for row in rows),
                                constants.DEFAULT_SPRITE_SIZE),
                            (max(len(rows), 2) * constants.DEFAULT_SPRITE_SIZE) // 2))
        if mode == "small":
            for j, (a, b) in enumerate(bounds):
                x = gaps[a]
                y_center = ((constants.DEFAULT_SPRITE_SIZE // 2) * j) + (constants.DEFAULT_SPRITE_SIZE // 4) \
                    if style != "oneline" else 12
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
            (int(sprite.width * ctx.gscale), int(sprite.height * ctx.gscale)), Image.NEAREST)
        return np.array(sprite)

    async def apply_options_name(
            self,
            tile: Tile,
            sprite: np.ndarray,
            wobble: int
    ) -> Image.Image:
        """Takes an image, taking tile data from its name, and applies the
        given options to it."""
        try:
            return await self.apply_options(
                tile,
                sprite,
                wobble
            )
        except ValueError as e:
            size = e.args[0]
            raise errors.BadTileProperty(tile.name, size)

    async def apply_options(
            self,
            tile: Tile,
            sprite: np.ndarray,
            wobble: int,
            seed: int | None = None
    ):
        random.seed(seed)
        for variant in tile.variants["sprite"]:
            sprite = await variant.apply(sprite, tile=tile, wobble=wobble, renderer=self)  # NOUN/PROP ARE ANNOYING
            if not all(np.array(sprite.shape[:2]) <= constants.MAX_TILE_SIZE):
                raise errors.TooLargeTile(sprite.shape[1::-1])
        return sprite

    def save_frames(
            self,
            images: list[np.ndarray],
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
            if background:
                save_images = [Image.fromarray(im) for im in images]
            else:
                save_images = []
                for i, im in enumerate(images):
                    # TODO: THIS IS EXTREMELY SLOW. BETTER WAY IS NEEDED.
                    colors, counts = np.unique(im.reshape(-1, 4), axis=0, return_counts=True)
                    sort_indices = np.argsort(counts)
                    colors = colors[sort_indices[::-1]] # Sort in descending order
                    palette_colors = [0, 0, 0]
                    formatted_colors = colors[colors[:, 3] != 0][..., :3]
                    formatted_colors = formatted_colors[:255].flatten()
                    palette_colors.extend(formatted_colors)
                    dummy = Image.new('P', (16, 16))
                    dummy.putpalette(palette_colors)
                    save_images.append(Image.fromarray(im).convert('RGB').quantize(
                        palette=dummy, dither=0))
            kwargs = {
                'format': "GIF",
                'interlace': True,
                'save_all': True,
                'append_images': save_images[1:],
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
            save_images[0].save(
                out,
                **kwargs
            )
        elif image_format == 'png':
            save_images = [Image.fromarray(im) for im in images]
            kwargs = {
                'format': "PNG",
                'save_all': True,
                'append_images': save_images,
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            save_images[0].save(
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
                Image.fromarray(img).save(buffer, "PNG")
                file.writestr(
                    f"{extra_name}_{i // 3}_{(i % 3) + 1}.png",
                    buffer.getvalue())
            file.close()


async def setup(bot: Bot):
    bot.renderer = Renderer(bot)
