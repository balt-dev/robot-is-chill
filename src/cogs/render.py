from __future__ import annotations

import math
import random
import time
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageChops, ImageSequence

from src.tile import ProcessedTile, Tile
from . import filterimage
from .. import constants, errors
from ..types import Variant
from ..utils import cached_open

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


def alpha_paste(img1, img2, coords):
    imgtemp = Image.new('RGBA', img1.size, (0, 0, 0, 0))
    imgtemp.paste(
        img2,
        coords
    )
    return Image.alpha_composite(img1, imgtemp)


def recolor(sprite: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Apply rgb color multiplication (0-255)"""
    r, g, b = rgb
    rc, gc, bc, ac = sprite.split()
    rc = rc.point(lambda i: int(i * (r / 255)))
    gc = gc.point(lambda i: int(i * (g / 255)))
    bc = bc.point(lambda i: int(i * (b / 255)))
    return Image.merge('RGBA', (rc, gc, bc, ac))


def delta_e(img1, img2):
    # compute the Euclidean distance with pixels of two images
    return np.sqrt(np.sum((img1 - img2) ** 2, axis=-1))


class Renderer:
    """This class exposes various image rendering methods.

    Some of them require metadata from the bot to function properly.
    """

    def __init__(self, bot: Bot) -> None:
        self.sprite_cache = {}
        self.bot = bot

    async def render(
            self,
            grid: list[list[list[list[ProcessedTile]]]],
            *,
            before_image: Image = None,
            palette: str = "default",
            images: list[str] | None = None,
            image_source: str = constants.BABA_WORLD,
            out: str | BinaryIO = "target/renders/render.gif",
            background: tuple[int, int] | str | None = None,
            upscale: int = 2,
            extra_out: str | BinaryIO | None = None,
            extra_name: str | None = None,
            frames: list[int] = (1, 2, 3),
            animation: tuple[int, int] = (0, None),
            speed: int = 200,
            gridol: tuple[int] = None,
            scaleddef: float = 1,
            crop: tuple[int, int, int, int] | None = None,
            pad: tuple[int, int, int, int] | None = (0, 0, 0, 0),
            image_format: str = 'gif',
            loop: bool = True,
            spacing: int = constants.DEFAULT_SPRITE_SIZE,
            expand: bool = False,
            boomerang: bool = False,
            **_
    ):
        """Takes a list of tile objects and generates a gif with the associated
        sprites.

        `out` is a file path or buffer. Renders will be saved there, otherwise to `target/renders/render.gif`.

        `palette` is the name of the color palette to refer to when rendering.

        `images` is a list of background image filenames. Each image is retrieved from `data/images/{image_source}/image`.

        `background` is a palette index. If given, the image background color is set to that color, otherwise transparent. Background images overwrite this.
        """
        animation, animation_delta = animation  # number of frames per wobble frame, number of frames per timestep
        palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")
        imgs = []
        times = []
        width = len(grid[0][0][0])
        height = len(grid[0][0])
        # don't need depth anywhere
        steps = len(grid)
        img_width_raw = max(int(width * (spacing * scaleddef)),
                            (constants.DEFAULT_SPRITE_SIZE * scaleddef))
        img_height_raw = max(int(height * (spacing * scaleddef)),
                             (constants.DEFAULT_SPRITE_SIZE * scaleddef))
        durations = [
            speed] * (len(frames) if animation_delta is None else animation_delta) * steps
        if animation:
            frames = np.repeat(frames, animation).tolist()
            frames = (frames * (math.ceil(len(durations) /
                                          animation_delta)))[:len(durations)]

        def wl(a):
            try:
                return int(a.frames[0].width * scaleddef)
            except BaseException:
                return 0

        def hl(a):
            try:
                return int(a.frames[0].height * scaleddef)
            except BaseException:
                return 0

        padding = np.amax(
            np.vectorize(
                lambda tile: max(
                    hl(tile),
                    wl(tile)))(
                np.array(grid)))
        img_width = int(img_width_raw + (2 * padding) + pad[0] + pad[2])
        img_height = int(img_height_raw + (2 * padding) + pad[1] + pad[3])
        bg_iter = [0] if animation else range(
            steps)  # no iteration on animation
        for _ in bg_iter:
            for layer, frame in enumerate(frames):
                if images and image_source is not None:
                    img = Image.new("RGBA", (img_width, img_height))
                    # for loop in case multiple background images are used
                    # (i.e. baba's world map)
                    for image in images:
                        overlap = Image.open(
                            f"data/images/{image_source}/{image}_{frame}.png")
                        img.paste(overlap, (padding, padding), mask=overlap)
                # bg color
                elif isinstance(background, tuple):
                    palette_color = palette_img.getpixel(background)
                    img = Image.new("RGBA", (img_width, img_height), color=tuple(
                        [max(c, 8) for c in palette_color]))
                elif isinstance(background, str):
                    img = Image.new("RGBA", (img_width, img_height), color=tuple(
                        [max(int(a + b, 16), 8) for a, b in np.reshape(list(background), (3, 2))]))
                # neither
                else:
                    img = Image.new(
                        "RGBA", (img_width, img_height), color=(
                            0, 0, 0, 0))
                imgs.append(img)
        # keeping track of the amount of padding we can slice off
        pad_r = pad_u = pad_l = pad_d = 0
        dframes = animation_delta if animation else len(frames)
        for d, timestep in enumerate(grid):
            for layer in timestep:
                for y, row in enumerate(layer):
                    for x, tile in enumerate(row):
                        # i should recode this whole section but i'm too scared
                        # of breaking something badly
                        t = time.perf_counter()
                        if tile.frames is None:
                            continue
                        tframes = []
                        anim_frames = frames
                        if animation:
                            anim_frames = anim_frames[d *
                                                      animation_delta:(d +
                                                                       1) *
                                                      animation_delta]
                        for f in anim_frames:
                            tframes.append(tile.frames[f - 1])
                        for frame, sprite in enumerate(tframes[:dframes]):
                            if animation:
                                dst_frame = (d * animation_delta) + frame
                            else:
                                dst_frame = frame + (d * len(frames))
                            x_offset = int(
                                (sprite.width - (spacing * scaleddef)) / 2)
                            y_offset = int(
                                (sprite.height - (spacing * scaleddef)) / 2)
                            x_offset_disp = int(
                                ((sprite.width - (spacing * scaleddef)) / 2) + tile.displacement[
                                    0] * scaleddef)
                            y_offset_disp = int(
                                ((sprite.height - (spacing * scaleddef)) / 2) + tile.displacement[
                                    1] * scaleddef)
                            x_offset_pad = int(
                                ((sprite.width - (spacing * scaleddef)) / 2) - tile.displacement[
                                    0] * scaleddef)
                            y_offset_pad = int(
                                ((sprite.height - (spacing * scaleddef)) / 2) - tile.displacement[
                                    1] * scaleddef)
                            if x == 0:
                                pad_l = max(
                                    pad_l, x_offset_pad if expand else x_offset)
                            if x == width - 1:
                                pad_r = max(
                                    pad_r, x_offset_pad if expand else x_offset)
                            if y == 0:
                                pad_u = max(
                                    pad_u, y_offset_pad if expand else y_offset)
                            if y == height - 1:
                                pad_d = max(
                                    pad_d, y_offset_pad if expand else y_offset)
                            alpha = sprite.getchannel("A")
                            x_offset_disp -= pad[0]
                            y_offset_disp -= pad[1]
                            coord_tuple = (
                                int(x * (spacing * scaleddef) +
                                    padding - x_offset_disp),
                                int(y * (spacing * scaleddef) +
                                    padding - y_offset_disp)
                            )
                            if tile.blending == "MASK":  # i should really rewrite all of this lmao
                                alpha = ImageChops.invert(alpha)
                                if background is not None:
                                    palette_color = palette_img.getpixel(
                                        background)
                                else:
                                    palette_color = (0, 0, 0, 0)
                                sprite = Image.new(
                                    "RGBA", (sprite.width, sprite.height), color=palette_color)
                                imgs[dst_frame].paste(
                                    sprite,
                                    coord_tuple,
                                    alpha
                                )
                            elif tile.blending == "CUT":
                                if background is not None:
                                    imgs[dst_frame].paste(
                                        Image.new(
                                            "RGBA",
                                            (sprite.width,
                                             sprite.height),
                                            palette_img.getpixel(background)),
                                        coord_tuple,
                                        alpha)
                                else:
                                    imgs[dst_frame].paste(
                                        Image.new(
                                            "RGBA", (sprite.width, sprite.height)),
                                        coord_tuple,
                                        alpha
                                    )
                            elif tile.blending == 'ADD':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (0, 0, 0, 0))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                imgs[dst_frame] = Image.fromarray(
                                    cv2.add(np.asarray(imgs[dst_frame]), np.asarray(imgtemp)))
                            elif tile.blending == 'SUB':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (0, 0, 0, 0))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                inmp = np.array(imgtemp)
                                inmp[:, :, 3] = 0
                                imgs[dst_frame] = Image.fromarray(
                                    cv2.absdiff(np.array(imgs[dst_frame]), inmp))
                            elif tile.blending == 'MAX':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (0, 0, 0, 0))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                imgs[dst_frame] = Image.fromarray(
                                    cv2.max(np.asarray(imgs[dst_frame]), np.asarray(imgtemp)))
                            elif tile.blending == 'XOR':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (0, 0, 0, 0))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                i1 = np.asarray(imgs[dst_frame])
                                i2 = np.asarray(imgtemp)
                                rgb = (i1 ^ i2)
                                rgb[:, :, 3] = cv2.max(
                                    i1[:, :, 3], i2[:, :, 3])
                                imgs[dst_frame] = Image.fromarray(rgb)
                            elif tile.blending == 'MIN':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (0, 0, 0, 0))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                imgtempar = np.asarray(imgtemp)
                                imgtempar[:, :, 3] = np.asarray(
                                    imgs[dst_frame])[:, :, 3]
                                imgs[dst_frame] = Image.fromarray(
                                    cv2.min(np.asarray(imgs[dst_frame]), imgtempar))
                            elif tile.blending == 'MUL':
                                imgtemp = Image.new(
                                    'RGBA', imgs[dst_frame].size, (255, 255, 255, 255))
                                imgtemp.paste(
                                    sprite,
                                    coord_tuple,
                                    mask=sprite
                                )
                                imgtempar = np.array(
                                    imgtemp, dtype=np.uint8) / 255
                                imgtempar[:, :, 3] = 1.0
                                imgs[dst_frame] = Image.fromarray(
                                    (np.array(imgs[dst_frame]) * np.array(imgtempar)).astype(np.uint8))
                            else:
                                imgs[dst_frame] = alpha_paste(
                                    imgs[dst_frame],
                                    sprite,
                                    coord_tuple
                                )
                        times.append(tile.delta + (time.perf_counter() - t))
        if before_image:
            bfr = 0
            before_durations = []
            for frame in ImageSequence.Iterator(before_image):
                try:
                    before_durations.append(frame.info['duration'])
                    im = frame.convert('RGBA')
                    im = im.resize((im.width // 2, im.height // 2), Image.NEAREST)
                    new_image = Image.new(
                        'RGBA', (img_width, img_height), (0, 0, 0, 0))
                    new_image.paste(
                        im, (padding - pad_l, padding - pad_u), mask=im)
                    imgs.insert(bfr, new_image)
                    bfr += 1  # I don't want to use an enumerate on an iterator
                except KeyError:
                    pass
            durations = before_durations + durations
        outs = []
        for img in imgs:
            img = np.array(img, dtype=np.uint8)
            img[np.all(img[:, :, :3] <= (8, 8, 8), axis=2)
                & (img[:, :, 3] > 0), :3] = 8
            if gridol is not None:
                for col in range(img.shape[0] // (gridol[0] * 2)):
                    img[col * gridol[0] * 2, :,
                        :] = ~img[col * gridol[0] * 2, :, :]
                    img[col * gridol[0] * 2, :, 3] = 255
                for row in range(img.shape[1] // (gridol[1] * 2)):
                    img[:, row * gridol[1] * 2,
                        :] = ~img[:, row * gridol[1] * 2, :]
                    img[:, row * gridol[1] * 2, 3] = 255
            if image_format == 'gif':
                img[:, :, :3] = (img[:, :, :3] *
                                 (img[:, :, 3] /
                                  255).repeat(3).reshape(img.shape[:2] +
                                                         (3,))).astype("uint8")
            img = Image.fromarray(img)
            img = img.crop(
                (padding - pad_l,
                 padding - pad_u,
                 img.width - padding + pad_r,
                 img.height - padding + pad_d))
            if crop is not None:
                img = img.crop(
                    (crop[0], crop[1], img.width - crop[2], img.height - crop[3]))
            if upscale != 1:
                img = img.resize(
                    (int(upscale * img.width), int(upscale * img.height)), resample=Image.NEAREST)
            outs.append(img)

        self.save_frames(
            outs,
            out,
            durations=durations,
            extra_out=extra_out,
            extra_name=extra_name,
            image_format=image_format,
            loop=loop,
            boomerang=boomerang,
            background=(background is not None)
        )
        if len(times) == 0:
            return 0, 0, 0
        else:
            return sum(times) / \
                len(times), max(times), sum(times),

    async def render_full_tile(self,
                               tile: Tile,
                               *,
                               position: tuple[int, int],
                               palette_img: Image.Image,
                               palette: str,
                               random_animations: bool = False,
                               raw_sprite_cache: dict[str, Image.Image],
                               gscale: float = 1,
                               tile_cache: dict[int, Tile] = {}
                               ) -> ProcessedTile | Tile | tuple[ProcessedTile, int]:
        """woohoo."""
        t = time.perf_counter()
        if tile.empty:
            return ProcessedTile(None)
        x, y = position
        wobble_hash = (11 * x + 13 * y) % 3 if random_animations else 0
        tile_hash = hash((tile, wobble_hash, palette, gscale))
        if tile_hash in tile_cache.keys():
            return tile_cache[tile_hash]
        out = []
        for frame in range(3):
            wobble = (
                11 * x + 13 * y + frame) % 3 if random_animations else frame
            if tile.custom and type(tile.sprite) == tuple:
                sprite = await self.generate_sprite(
                    tile.name,
                    style=tile.custom_style or (
                        "noun" if len(tile.name) < 1 else "letter"),
                    wobble=wobble,
                    variants=tile.variants["sprite"],
                    position=(x, y),
                    gscale=gscale
                )
            else:
                if isinstance(tile.sprite, np.ndarray):
                    sprite = Image.fromarray(
                        tile.sprite[(tile.frame * 3) + wobble])
                else:
                    path_fallback = None
                    if tile.name == "icon":
                        path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}.png"
                    elif tile.name in ("smiley", "hi") or tile.name.startswith("icon"):
                        path = f"data/sprites/{constants.BABA_WORLD}/{tile.name}_1.png"
                    elif tile.name == "default":
                        path = f"data/sprites/{constants.BABA_WORLD}/default_{wobble + 1}.png"
                    else:
                        if tile.frame == -1:
                            source, sprite_name = tile.sprite
                            path = f"data/sprites/vanilla/error_0_{wobble + 1}.png"
                        else:
                            source, sprite_name = tile.sprite
                            path = f"data/sprites/{source}/{sprite_name}_{tile.frame}_{wobble + 1}.png"
                        try:
                            path_fallback = f"data/sprites/{source}/{sprite_name}_{tile.fallback_frame}_{wobble + 1}.png"
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
                            raise AssertionError(f'The tile `{tile.name}:{tile.frame}` was found, but the files don\'t exist for it.')
                sprite = sprite.resize(
                    (int(sprite.width * gscale), int(sprite.height * gscale)), Image.NEAREST)
                computed_hash = hash((
                    tile_hash,
                    tuple(sprite.getdata()),
                    wobble,
                    gscale
                ))
                if computed_hash in self.sprite_cache:
                    sprite = self.sprite_cache[computed_hash]
                sprite = await self.apply_options_name(
                    tile.name,
                    sprite,
                    style=tile.custom_style,
                    wobble=wobble,
                    variants=tile.variants["sprite"]
                )
            for variant in tile.variants["post"]:
                variant.apply(tile) # something like this should replace most of this shit below, maybe tile.variants["post"]?
            # Color augmentation
            if tile.overlay is None:
                try:
                    if tile.palette is None:
                        rgb = tile.color if len(tile.color) == 3 else palette_img.getpixel(
                            tile.color)
                    else:
                        rgb = tile.color if len(tile.color) == 3 else Image.open(
                            f"data/palettes/{tile.palette}.png").convert("RGB").getpixel(tile.color)
                except IndexError:
                    raise errors.BadPaletteIndex(tile.name, tile.color)
                sprite = recolor(sprite, rgb)
            else:
                try:
                    overlay = Image.open(
                        f"data/overlays/{tile.overlay}.png").convert("RGBA")
                    if overlay.width < sprite.width or overlay.height < sprite.height:
                        width = math.ceil(sprite.width / overlay.width)
                        height = math.ceil(sprite.height / overlay.height)
                        rgb = np.tile(
                            np.array(overlay), (height, width, 1)) / 255
                    else:
                        rgb = np.array(overlay) / 255
                except FileNotFoundError:
                    raise errors.OverlayNotFound(tile.overlay)
                ovsprite = np.array(sprite).astype("float64")
                ovsprite *= rgb[:ovsprite.shape[0], :ovsprite.shape[1]]
                ovsprite = ovsprite.astype("uint8")
                sprite = Image.fromarray(ovsprite)
            if tile.hue != 0.0:
                sprite = Image.fromarray(
                    shift_hue(
                        np.array(
                            sprite,
                            dtype="uint8"),
                        tile.hue))
            if tile.gamma != 1:
                bsprite = np.array(sprite, dtype="float64")
                bsprite *= (tile.gamma, tile.gamma,
                            tile.gamma, 1)
                bsprite[bsprite > 255] = 255
                bsprite[bsprite < 0] = 0
                sprite = Image.fromarray(bsprite.astype("uint8"))
            if tile.palette_snapping is True:  # it defaults to true without this for some reason
                palette_colors = np.array(Image.open(
                    f"data/palettes/{tile.palette or 'default'}.png").convert("RGB")).reshape(-1, 3)
                im = np.array(sprite)
                im_lab = cv2.cvtColor(
                    im.astype(
                        np.float32) / 255,
                    cv2.COLOR_RGB2Lab)
                diff_matrix = np.full(
                    (palette_colors.shape[0], *im.shape[:-1]), 999)
                for i, color in enumerate(
                        palette_colors):  # still slow, but faster than iterating through every pixel
                    filled_color_array = np.array([[color]]).repeat(
                        im.shape[0], 0).repeat(im.shape[1], 1)
                    filled_color_array = cv2.cvtColor(
                        filled_color_array.astype(
                            np.float32) / 255, cv2.COLOR_RGB2Lab)
                    im_delta_e = delta_e(im_lab, filled_color_array)
                    diff_matrix[i] = im_delta_e
                min_indexes = np.argmin(diff_matrix, 0, keepdims=True).reshape(
                    diff_matrix.shape[1:])
                result = np.full(im.shape, 0, dtype=np.uint8)
                for i, color in enumerate(palette_colors):
                    result[:, :, :3][min_indexes == i] = color
                result[:, :, 3] = im[:, :, 3]
                sprite = Image.fromarray(result)
            if tile.saturation != 1:
                sprite = Image.fromarray(
                    grayscale(
                        np.array(sprite),
                        tile.saturation))
            if tile.filterimage is not None:
                url = tile.filterimage
                absolute = False
                if url.startswith("abs"):
                    url = url[3:]
                    absolute = True
                if url.startswith("db!"):
                    url = url[3:]
                    command = "SELECT url FROM filterimages WHERE name == ?;"
                    args = (url,)
                    async with self.bot.db.conn.cursor() as cursor:
                        await cursor.execute(command, args)
                        results = await cursor.fetchone()
                        if results is None:
                            raise requests.exceptions.ConnectionError
                        url = results[0]
                p = requests.get(url, stream=True).raw.read()
                # try:
                ifilterimage = Image.open(BytesIO(p)).convert("RGBA")
                sprite = filterimage.apply_filterimage(sprite, ifilterimage.resize(
                    (int(ifilterimage.width * gscale), int(ifilterimage.height * gscale)), Image.NEAREST), absolute)
                # except OSError:
                #    raise AssertionError('Image wasn\'t able to be accessed, or is invalid!')
            if tile.normalize_gamma:
                arr = np.array(sprite)
                arr_rgb, sprite_a = arr[:, :, :3], arr[:, :, 3]
                arr_hls = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HLS).astype(
                    np.float64)  # since WHEN was it HLS???? huh?????
                max_l = np.max(arr_hls[:, :, 1])
                arr_hls[:, :, 1] *= (255 / max_l)
                sprite_rgb = cv2.cvtColor(
                    arr_hls.astype(
                        np.uint8),
                    cv2.COLOR_HLS2RGB)  # my question still stands
                sprite = np.dstack((sprite_rgb, sprite_a))
            numpysprite = np.array(sprite)
            numpysprite[np.all(numpysprite[:, :, :3] <= (8, 8, 8), axis=2) & (
                numpysprite[:, :, 3] > 0), :3] = 8
            sprite = Image.fromarray(numpysprite)
            out.append(sprite)
        f0, f1, f2 = out
        final_tile = ProcessedTile(
            (f0,
             f1,
             f2),
            tile.blending,
            tile.displacement,
            time.perf_counter() - t)
        tile_cache[tile_hash] = final_tile
        return final_tile

    async def render_full_tiles(
            self,
            grid: list[list[list[list[Tile]]]],
            *,
            palette: str = "default",
            random_animations: bool = False,
            gscale: float = 1
    ) -> tuple[list[list[list[list[ProcessedTile]]]], int]:
        """Final individual tile processing step."""
        palette = palette or "default"
        random_animations = random_animations if random_animations is not None else False
        gscale = gscale if gscale is not None else 2
        sprite_cache = {}
        palette_img = Image.open(f"data/palettes/{palette}.png").convert("RGB")
        tile_cache = {}
        d = []
        for timestep in grid:
            a = []
            for layer in timestep:
                b = []
                for y, row in enumerate(layer):
                    c = []
                    for x, tile in enumerate(row):
                        c.append(
                            await self.render_full_tile(
                                tile,
                                position=(x, y),
                                palette_img=palette_img,
                                palette=palette,
                                random_animations=random_animations,
                                raw_sprite_cache=sprite_cache,
                                gscale=gscale,
                                tile_cache=tile_cache
                            )
                        )
                    b.append(c)
                a.append(b)
            d.append(a)
        return d, len(tile_cache)

    async def generate_sprite(
            self,
            text: str,
            *,
            style: str,
            wobble: int,
            seed: int | None = None,
            variants: list[Variant],
            gscale: float,
            position: tuple[int, int]
    ) -> Image.Image:
        """Generates a custom text sprite."""
        text = text[5:]
        raw = text.replace("/", "")
        newline_count = text.count("/")
        assert len(text) <= 64, 'Text has a maximum length of `64` characters.'
        if seed is None:
            seed = int((7 + position[0]) / (3 + position[1]) * 100000000)
        seed_digits = [(seed >> 8 * i) | 0b11111111 for i in range(len(raw))]
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

        max_width = max(sum(widths[:index]), constants.DEFAULT_SPRITE_SIZE)

        def check_or_adjust(widths: list[int], index: int) -> int:
            """Is the arrangement valid?"""
            if mode == "small":
                if not fixed:
                    while sum(widths[:index]) > max_width:
                        index -= 1
                    while sum(widths[index:]) > max_width:
                        index += 1
                    return index
            return index

        def too_squished(widths: list[int], index: int) -> bool:
            """Is the arrangement too squished?

            (bad letter spacing)
            """
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
            old_width, i = min((w, i)
                               for i, w in enumerate(widths) if not stable[i])
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

        sprite = Image.new("L",
                           (max(sum(widths[:index]),
                                constants.DEFAULT_SPRITE_SIZE),
                            constants.DEFAULT_SPRITE_SIZE))
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
        sprite = sprite.resize(
            (int(sprite.width * gscale), int(sprite.height * gscale)), Image.NEAREST)
        return self.apply_options(
            sprite,
            original_style="noun",
            wobble=wobble,
            variants=variants,
            style=style
        )

    async def apply_options_name(
            self,
            name: str,
            sprite: Image.Image,
            *,
            wobble: int,
            variants: list[Variant],
            style: str
    ) -> Image.Image:
        """Takes an image, taking tile data from its name, and applies the
        given options to it."""
        tile_data = await self.bot.db.tile(name)
        if tile_data is not None:
            original_style = constants.TEXT_TYPES[tile_data.text_type]
        else:  # catch generated sprites
            original_style = style
        try:
            return self.apply_options(
                sprite,
                original_style=original_style,
                wobble=wobble,
                variants=variants,
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
            wobble: int,
            style: str | None = None,
            variants: list[Variant],
            seed: int | None = None
    ):
        random.seed(seed)
        for variant in variants:
            sprite = variant.apply(sprite)
        return sprite

    def save_frames(
            self,
            imgs: list[Image.Image],
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
        if boomerang and len(imgs) > 2:
            imgs += imgs[-2:0:-1]
            durations += durations[-2:0:-1]
        if image_format == 'gif':
            if not background:
                for i, im in enumerate(imgs):
                    np_im = np.array(im.convert("RGBA"))
                    colors = np.unique(np_im.reshape(-1, 4),
                                       axis=0)
                    colors = [0, 0, 0] + colors[colors[:, 3]
                                                != 0][:254, :3].flatten().tolist()
                    dummy = Image.new('P', (16, 16))
                    dummy.putpalette(colors)
                    imgs[i] = im.convert('RGB').quantize(
                        palette=dummy, dither=0)
            kwargs = {
                'format': "GIF",
                'interlace': True,
                'save_all': True,
                'append_images': imgs[1:],
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
            imgs[0].save(
                out,
                **kwargs
            )
        elif image_format == 'png':
            kwargs = {
                'format': "PNG",
                'save_all': True,
                'append_images': imgs,
                'default_image': True,
                'loop': 0,
                'duration': durations
            }
            if not loop:
                kwargs['loop'] = 1
            imgs[0].save(
                out,
                **kwargs
            )
        if not isinstance(out, str):
            out.seek(0)
        if extra_name is None:
            extra_name = 'render'
        if extra_out is not None:
            file = zipfile.PyZipFile(extra_out, "x")
            for i, img in enumerate(imgs):
                buffer = BytesIO()
                img.save(buffer, "PNG")
                file.writestr(
                    f"{extra_name}_{i // 3}_{(i % 3) + 1}.png",
                    buffer.getvalue())
            file.close()


async def setup(bot: Bot):
    bot.renderer = Renderer(bot)
