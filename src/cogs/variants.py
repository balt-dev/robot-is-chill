import glob
import inspect
import math
import re
import sys
import types
import typing
from typing import Any, Literal, Optional, Union, get_origin, get_args, Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageChops, Image, ImageOps, ImageFilter

from . import liquify
from ..errors import VariantError
from ..utils import recolor
from .. import constants
from ..types import Variant, RegexDict, VaryingArgs, Color

"""
TODO:
- def f(*args: type)
"""

CARD_KERNEL = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)))
OBLQ_KERNEL = np.array(((1, 0, 1), (0, 0, 0), (1, 0, 1)))

def class_init(self, *args, **kwargs):
    print(args)
    self.args = args
    self.kwargs = kwargs


def parse_signature(v: list[str], t: list[type | types.GenericAlias]) -> list[Any]:
    out = []
    t = list(t).copy()
    v = list(v).copy()
    if v is None or not len(v) or v[0] is None:
        val = None
    while len(t) > 0:
        if len(v) == 0:
            break
        curr_type = t.pop(0)
        if get_origin(curr_type) is Union:
            curr_type = get_args(curr_type)[0]
            if v[0] is None:
                continue
        if get_origin(curr_type) is Literal:
            curr_type = get_args(curr_type)
        if type(curr_type) is VaryingArgs:
            print(v, [curr_type.type] * len(v))
            out.extend(parse_signature(v, [curr_type.type] * len(v)))
            return out
        elif isinstance(curr_type, list):  # tree branch
            num_values = len(curr_type)
            val = tuple(parse_signature(v[:num_values], curr_type))
            del v[:num_values]
        elif isinstance(curr_type, tuple):  # literal, does not need to be checked
            val = v.pop(0)
        elif curr_type is bool:
            g = v.pop(0)
            val = g == "true"
        else:
            try:
                raw_val = v.pop(0)
                val = curr_type(raw_val)
            except ValueError:
                val = None
        out.append(val)
    return out


async def setup(bot):
    """Get the variants."""
    bot.variants = []

    def generate_pattern(params: list[inspect.Parameter], keep_slash=False):
        pattern = f""
        for i, p in enumerate(params):
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                # You can't match a variable number of groups in RegEx.
                fstring_backslash_hack = r"^/|(?<=^\(\?:)/"
                pattern += f"(?:{'/' if i - 1 else ''}{(generate_pattern([inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=p.annotation)]).replace(fstring_backslash_hack, '', 1) + '/?')})?" * constants.VAR_POSITIONAL_MAX
            elif get_origin(p.annotation) == Literal and len(get_args(p.annotation)) > 1:
                pattern += f"/({'|'.join([str(arg) for arg in get_args(p.annotation)])})"
            elif get_origin(p.annotation) == list:
                pattern += rf"/\({generate_pattern([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno) for name, anno in zip(p.name.split('_'), get_args(p.annotation))])}\)"
            elif get_origin(p.annotation) == Union:
                pattern += f"(?:{'/' if i - 1 else ''}{generate_pattern([inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=get_args(p.annotation)[0])])})?"
            elif p.annotation in (patterns := {
                int: r"/(-?\d+)",
                float: r"/([+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+))",
                # From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers/42629198#42629198
                str: r"/(.+?)",
                bool: r"/(true|false)",
                Color: rf"/((?:#(?:[0-9A-Fa-f]{{2}}){{3,4}})|"
                       rf"(?:#(?:[0-9A-Fa-f]){{3,4}})|"
                       rf"(?:{'|'.join(constants.COLOR_NAMES.keys())})|"
                       rf"(?:-?\d+\/-?\d+)|"
                       rf"\((?:-?\d+\/-?\d+)\))"
            }):
                pattern += patterns[p.annotation]
            else:
                continue
        return pattern if keep_slash else pattern.lstrip("/")  # Remove starting /

    def generate_syntax(params):
        syntax = ""
        print(f"! {params}")
        for name, param in dict(params).items():
            if param.annotation == inspect.Parameter.empty:
                continue
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                print(">", typing.get_args(param.annotation))
                syntax += f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m" \
                          f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m" \
                          f"/[0;30m..."
                break
            elif get_origin(param.annotation) is Union:
                syntax += f"[0;30m[[1;34m{generate_syntax({name: inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typing.get_args(param.annotation)[0])})}" \
                          f"[0;30m: [1;37m{repr(param.default)}[0;30m][0m"
            elif get_origin(param.annotation) == list:
                syntax += f"""[0;34m({generate_syntax({
                    name: inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno)
                    for name, anno in zip(name.split('_'), get_args(param.annotation))})}[0;34m)[0m"""
            elif get_origin(param.annotation) == Literal:
                syntax += f"""[0;30m<[32m{'[0;30m/[32m'.join([repr(arg) for arg in get_args(param.annotation)])} [0;36m{name}[0;30m>[0m"""
            else:
                syntax += f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m"
            syntax += "/"
        return syntax.rstrip("[0m").rstrip("/")  # Remove ending /

    def get_type_tree(unparsed_tree):
        tree = []
        for i, p in enumerate(unparsed_tree):
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                t = p.annotation
                if isinstance(t, types.GenericAlias):
                    t = get_type_tree(inspect.Parameter(p.name, p.kind, annotation=anno) for anno in get_args(t))
                tree.append(VaryingArgs(t))
                return tree
            elif get_origin(p.annotation) == Literal:
                tree.append(tuple(get_args(p.annotation)))
            elif isinstance(p.annotation, types.GenericAlias):
                tree.append(get_type_tree(
                    inspect.Parameter(p.name, p.kind, annotation=anno) for anno in get_args(p.annotation)))
            else:
                tree.append(p.annotation)
        return tree

    def create_variant(func: Callable, aliases: Iterable[str], no_function_name=False):
        assert func.__doc__ is not None, f"Variant `{func.__name__}` is missing a docstring!"
        sig = inspect.signature(func)
        params = sig.parameters
        has_kwargs = any([p.kind == inspect.Parameter.KEYWORD_ONLY for p in params.values()])
        if not no_function_name:  # HACKY
            aliases = list(aliases)
            aliases.append(func.__name__)
            aliases = tuple(aliases)
        pattern = rf"(?:{'|'.join(aliases)}){generate_pattern(list(params.values()))}"
        print(func.__name__, pattern)
        syntax = (f"\u001b[0;30m[\u001b[0;35m{'[0;30m|[0;35m'.join(aliases)}[0;30m]" if len(
            aliases) else "") + generate_syntax(params)
        class_name = func.__name__.replace("_", " ").title().replace(" ", "") + "Variant"
        variant_type = tuple(params.keys())[0]
        type_tree = get_type_tree([p for p in params.values() if p.kind != inspect.Parameter.KEYWORD_ONLY])[1:]
        bot.variants.append(
            type(
                class_name,
                (Variant,),
                {
                    "__init__": class_init,
                    "__doc__": func.__doc__,
                    "__repr__": (lambda self: f"{self.__class__.__name__}{self.args}"),
                    "apply": (lambda self, obj, **kwargs: func(obj, *self.args, **(self.kwargs | kwargs) if has_kwargs else self.kwargs)),
                    "pattern": pattern,
                    "signature": type_tree,
                    "syntax": syntax,
                    "type": variant_type,
                }
            )
        )

    def add_variant(*aliases, no_function_name=False):
        def wrapper(func):
            create_variant(func, aliases, no_function_name)
            return func

        return wrapper

    # --- SPECIAL ---

    @add_variant("", "noop")
    def nothing(tile):
        """Does nothing. Useful for resetting persistent variants."""
        pass

    @add_variant(no_function_name=True)
    def blending(tile, mode: Literal[*tuple(constants.BLENDING_MODES.keys())]):
        """Sets the blending mode for the tile."""
        tile.blending = constants.BLENDING_MODES[mode]

    # --- ANIMATION FRAMES ---

    @add_variant(no_function_name=True)
    def frame(tile, anim_frame: int):
        """Sets the animation frame of a sprite."""
        tile.altered_frame = True
        tile.frame = anim_frame
        tile.surrounding = 0

    @add_variant(no_function_name=True)
    def direction(tile, direction: Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]):
        """Sets the direction of a tile."""
        tile.altered_frame = True
        tile.frame = constants.DIRECTION_VARIANTS[direction]

    @add_variant(no_function_name=True)
    def tiling(tile, tiling: Literal[*tuple(constants.AUTO_VARIANTS.keys())]):
        """Alters the tiling of a tile. Only works on tiles that tile."""
        tile.altered_frame = True
        tile.surrounding |= constants.AUTO_VARIANTS[tiling]

    @add_variant("a", no_function_name=True)
    def animation_frame(tile, a_frame: int):
        """Sets the animation frame of a tile."""
        tile.altered_frame = True
        tile.frame += a_frame

    @add_variant("s", "sleep", no_function_name=True)
    def sleep(tile):
        """Makes the tile fall asleep. Only functions correctly on character tiles."""
        tile.altered_frame = True
        tile.frame = (tile.frame - 1) % 32

    # --- COLORING ---

    palette_names = tuple([Path(p).stem for p in glob.glob("data/palettes/*.png")])

    @add_variant("palette/", "p!", no_function_name=True)
    def palette(tile, palette: str):
        """Sets the tile's palette."""
        assert palette in palette_names, f"Palette `{palette}` was not found!"
        tile.palette = palette

    @add_variant(no_function_name=True)
    def color(sprite, color: Color, *, tile, wobble, renderer, _default_color=False):
        """Sets the tile's color.
Can take:
- A hexadecimal RGB/RGBA value, as #RGB, #RGBA, #RRGGBB, or #RRGGBBAA
- A color name, as is (think the color properties from the game)
- A palette index, as an x/y coordinate"""
        print(color)
        if _default_color:
            if tile.custom_color:
                print(f"Skipping color {color}")
                return sprite
        else:
            tile.custom_color = True
        if len(color) == 4:
            rgba = color
        else:
            rgba = *renderer.palette_cache[tile.palette].getpixel(color), 0xFF
        return recolor(sprite, rgba)

    @add_variant("rot")
    def rotate(sprite, angle: float, expand: Optional[bool]=False):
        """Rotates a sprite."""
        arr = np.array(sprite.convert("RGBA"))
        if expand:
            scale = math.cos(math.radians(angle)) + math.sin(math.radians(angle))
            arr = np.pad(arr, ((int(arr.shape[0] * ((scale - 1) / 2)), int(arr.shape[0] * ((scale - 1) / 2))),
                               (int(arr.shape[0] * ((scale - 1) / 2)), int(arr.shape[0] * ((scale - 1) / 2))),
                               (0, 0)))
        image_center = tuple(np.array(arr.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(arr, rot_mat, arr.shape[1::-1], flags=cv2.INTER_NEAREST)
        return Image.fromarray(result)

    @add_variant()
    def scale(sprite, w: float, h: Optional[float] = None):
        """Scales a sprite by the given multipliers."""
        arr = np.array(sprite)
        if h is None:
            h = w
        if int(w*sprite.width) <= 0 or int(h*sprite.height) <= 0:
            raise AssertionError(f"Can't scale a tile to `{int(w*sprite.width)}/{int(h*sprite.height)}`, as it has a non-positive target area.")
        dim = arr.shape[:2] * np.array([w, h])
        dim = dim.astype(int)
        result = cv2.resize(arr, dim, interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(result)

    @add_variant("grad")
    def gradient(sprite, color: Color, angle: Optional[float] = 0.0, width: Optional[float] = 1.0,
                 offset: Optional[float] = 0, steps: Optional[int] = 0, raw: Optional[bool] = False, extrapolate: Optional[bool] = False, *, tile, wobble, renderer):
        """Applies a gradient to a tile.
Interpolates color through CIELUV color space by default. This can be toggled with [0;36mraw[0m.
If [0;36mextrapolate[0m is on, then colors outside the gradient will be extrapolated, as opposed to clamping from 0% to 100%."""
        tile.custom_color = True
        src = Color.parse(tile, renderer.palette_cache)
        dst = Color.parse(tile, renderer.palette_cache, color=color)
        if not raw:
            src = np.hstack((cv2.cvtColor(np.array([[src[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], src[3]))
            dst = np.hstack((cv2.cvtColor(np.array([[dst[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], dst[3]))
        # thank you hutthutthutt#3295 you are a lifesaver
        scale = math.cos(math.radians(angle % 90)) + math.sin(math.radians(angle % 90))
        maxside = max(sprite.width, sprite.height) + 1
        grad = np.mgrid[offset:width+offset:maxside * 1j]
        grad = np.tile(grad[..., np.newaxis], (maxside, 1, 4))
        if not extrapolate:
            grad = np.clip(grad, 0, 1)
        grad_center = maxside / 2, maxside / 2
        rot_mat = cv2.getRotationMatrix2D(grad_center, angle, scale)
        warped_grad = cv2.warpAffine(grad, rot_mat, (sprite.width, sprite.height), flags=cv2.INTER_LINEAR)
        if steps:
            warped_grad = np.round(warped_grad * steps) / steps
        mult_grad = np.clip(((1 - warped_grad) * src + warped_grad * dst), 0, 255)
        if not raw:
            mult_grad[:, :, :3] = cv2.cvtColor(mult_grad[:, :, :3].astype(np.uint8), cv2.COLOR_Luv2RGB).astype(np.float64)
        mult_grad /= 255
        arr = np.array(sprite.convert("RGBA"))
        return Image.fromarray((arr * mult_grad).astype(np.uint8))

    # --- TEXT MANIPULATION ---

    @add_variant("noun", "prop")
    def property(sprite, plate: Optional[Literal["blank", "left", "up", "right", "down", "turn", "deturn"]] = None, *,
                 tile, wobble, renderer):
        """Applies a property plate to a sprite."""
        if plate is None:
            plate = tile.frame if tile.altered_frame else None
        else:
            plate = {v: k for k, v in constants.DIRECTIONS.items()}[plate]
        plate, box = renderer.bot.db.plate(plate, wobble)
        size = (max(sprite.width, plate.width), max(sprite.height, plate.height))
        dummy = Image.new("1", size)
        delta = (plate.size[0] - sprite.size[0]) // 2, \
                (plate.size[0] - sprite.size[1]) // 2
        plate_dummy = Image.new("1", size)
        plate_dummy.paste(
            plate.getchannel("A").convert("1", dither=Image.NONE),
            (max((sprite.width - plate.width) // 2, 0), max((sprite.height - plate.height) // 2, 0))
        )
        dummy.paste(
            Image.new("1", (sprite.width, sprite.height), 1),
            delta,
            sprite.getchannel("A").convert("1", dither=Image.NONE)
        )
        band = ImageChops.logical_and(ImageChops.logical_xor(dummy, plate_dummy), plate_dummy).convert("L")
        return Image.merge("RGBA", (band, band, band, band))

    @add_variant()
    def custom(tile):
        """Forces custom generation of the text."""
        tile.custom = True
        tile.style = "noun"

    @add_variant("let")
    def letter(tile):
        """Makes 1 or 2 letter custom words appear as letter groups."""
        tile.style = "letter"

    @add_variant()
    def hide(tile):
        """Hides the tile."""
        tile.empty = True

    # --- FILTERS ---

    @add_variant()
    def pad(sprite, left: int, top: int, right: int, bottom: int):
        """Pads the sprite by the specified values."""
        return Image.fromarray(
            np.pad(
                np.array(sprite), ((top, bottom), (left, right), (0, 0)),
            )
        )

    @add_variant()
    def pixelate(sprite, x: int, y: Optional[int] = None):
        """Pixelates the sprite."""
        if y is None:
            y = x
        return sprite.resize((sprite.size[0]//x, sprite.size[1]//y), Image.NEAREST).resize(sprite.size, Image.NEAREST)

    @add_variant("m")
    def meta(sprite, level: Optional[int] = 1) -> Image.Image:
        """Applies a meta filter to an image."""
        assert abs(level) <= constants.MAX_META_DEPTH, f"Meta depth of {level} too large!"
        orig = sprite.copy()
        base = sprite.getchannel("A")
        if level < 0:
            level = abs(level)
            base = ImageOps.invert(base)
        for _ in range(level):
            temp = base.crop((-2, -2, base.width + 2, base.height + 2))
            filtered = ImageChops.invert(temp).filter(ImageFilter.FIND_EDGES)
            base = filtered.crop(
                (1, 1, filtered.width - 1, filtered.height - 1))
        base = Image.merge("RGBA", (base, base, base, base))
        if level % 2 == 0 and level != 0:
            base.paste(orig, (level, level), mask=orig)
        elif level % 2 == 1 and level != 1:
            blank = Image.new("RGBA", orig.size)
            base.paste(blank, (level, level), mask=orig)
        return base

    @add_variant()
    def land(sprite, direction: Optional[Literal["left", "up", "right", "down"]] = "down"):
        """Removes all space between the tile and its bounding box on the specified side."""
        #sprite = np.array(sprite dtype=np.uint8)
        #for
        #return Image.fromarray(warped_sprite)

    @add_variant()
    def warp(sprite, x1_y1: list[int, int], x2_y2: list[int, int], x3_y3: list[int, int], x4_y4: list[int, int]):
        """Warps the sprite by displacing the bounding box's corners.
    Point 1 is top-left, point 2 is top-right, point 3 is bottom-right, and point 4 is bottom-left.
    The sprite may look incorrect if the described quadrilateral is not convex.
    If the sprite grows past its original bounding box, it will need to be recentered manually."""
        sprite = np.array(sprite.convert("RGBA"), dtype=np.uint8)
        src_shape = np.array(sprite.shape[-2::-1])
        src = np.array([[0,0], [1,0], [1,1], [0, 1]]) * (src_shape - 1)
        dst = src + np.array([x1_y1, x2_y2, x3_y3, x4_y4])
        # Set padding values
        before_padding = np.array([
            max(-x1_y1[0], -x4_y4[0], 0),  # Added padding for left
            max(-x1_y1[1], -x2_y2[1], 0)  # Added padding for top
        ])
        after_padding = np.array([
            (max(x2_y2[0], x3_y3[0], 0)),  # Added padding for right
            (max(x3_y3[1], x4_y4[1], 0))  # Added padding for bottom
        ])
        new_shape = (src_shape + before_padding + after_padding).astype(np.uint32)
        print(dst + before_padding, new_shape)
        # Get a polygon that represents the wanted shape of the image
        clip_poly = cv2.fillConvexPoly(np.zeros(new_shape), (dst + before_padding).astype(np.int32), 1).astype(bool)
        # Transform the sprite
        matrix = cv2.getPerspectiveTransform(src.astype(np.float32), (dst + before_padding).astype(np.float32))
        warped_sprite = cv2.warpPerspective(sprite, matrix, dsize=new_shape, flags=cv2.INTER_NEAREST)
        # Clip the image to the clip polygon defined earlier
        warped_sprite[~clip_poly] = 0
        return Image.fromarray(warped_sprite)

    @add_variant("mm")
    def matrix(sprite,
               aa_ab_ac_ad: list[float, float, float, float],
               ba_bb_bc_bd: list[float, float, float, float],
               ca_cb_cc_cd: list[float, float, float, float],
               da_db_dc_dd: list[float, float, float, float]):
        """Multiplies the sprite by the given RGBA matrix."""
        matrix = np.array((aa_ab_ac_ad, ba_bb_bc_bd, ca_cb_cc_cd, da_db_dc_dd)).T
        img = np.array(sprite.convert('RGBA'), dtype=np.float64) / 255.0
        immul = img.reshape(-1, 4) @ matrix  # @ <== matmul
        immul = (np.clip(immul, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(immul.reshape(img.shape))

    @add_variant()
    def neon(sprite, strength: Optional[float] = 0.714):
        """Darkens the inside of each region of color."""
        # This is approximately 2.14x faster than Charlotte's neon, profiling at strength 0.5 with 2500 iterations on baba/frog_0_1.png.
        arr = np.array(sprite, dtype=np.float64)
        unique_colors = liquify.get_colors(arr)
        final_mask = np.ones(arr.shape[:2], dtype=np.float64)
        for color in unique_colors:
            mask = (arr == color).all(axis=2)
            float_mask = mask.astype(np.float64)
            card_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=CARD_KERNEL)
            oblq_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=OBLQ_KERNEL)
            final_mask[card_mask == 4] -= strength / 2
            final_mask[oblq_mask == 4] -= strength / 2
        if strength < 0:
            final_mask = np.abs(1 - final_mask)
        arr[:, :, 3] *= np.clip(final_mask, 0, 1)
        return Image.fromarray(arr.astype(np.uint8))

    @add_variant()
    def flip(sprite, *axis: Literal["x", "y"]):
        """Flips the sprite along the specified axes."""
        for a in axis:
            sprite = {"x": ImageOps.mirror, "y": ImageOps.flip}[a](sprite)
        return sprite

    @add_variant()
    def posterize(sprite, bits: int):
        """Posterizes the sprite's colors to the specified number of bits."""
        return ImageOps.posterize(sprite, bits)

    @add_variant()
    def solarize(sprite, threshold: Optional[int] = 128):
        """Inverts all pixels in the sprite above a threshold."""
        return ImageOps.solarize(sprite, threshold)

    @add_variant("norm")
    def normalize(sprite):
        """Centers the sprite on its visual bounding box."""
        left, top, right, bottom = sprite.getbbox()
        sprite_center = sprite.width // 2, sprite.height // 2
        center = int((left + right) // 2), int((top + bottom) // 2)
        displacement = sprite_center[0] - center[0] + left, sprite_center[1] - center[1] + top
        dummy = Image.new("RGBA", sprite.size)
        dummy.paste(sprite.crop((left, top, right, bottom)), displacement)
        return dummy

    @add_variant("disp")
    def displace(tile, x: int, y: int):
        """Displaces the tile by the specified coordinates."""
        tile.displacement = (tile.displacement[0] - x, tile.displacement[1] - y)

    # Original code by Charlotte (CenTdemeern1)
    @add_variant("flood")
    def floodfill(sprite, brightness: Optional[float] = 1.0, inside: Optional[bool] = False):
        """Floodfills either inside or outside a sprite with a given brightness value."""
        brightness = int(brightness * 255)
        im = np.array(sprite)
        im[im[:, :, 3] == 0] = 0  # Optimal
        im_alpha = im[:, :, 3]  # Stores the alpha channel separately
        im_alpha[im_alpha > 0] = -1  # Sets all nonzero numbers to a number that's neither 0 nor 255.
        # Pads the alpha channel by 1 on each side to allow flowing past
        # where the sprite touches the edge of the bounding box.
        im_alpha = np.pad(im_alpha, ((1, 1), (1, 1)))
        im_flooded = cv2.floodFill(
            image=im_alpha,
            mask=None,
            seedPoint=(0, 0),
            newVal=255
        )[1]
        mask = im_flooded != (0 if inside else 255)
        im_flooded[mask] = (255 if inside else 0)
        mask = mask[1:-1, 1:-1]
        if not inside:
            im_flooded = 255 - im_flooded
        # Crops the alpha channel back to the original size and positioning
        im[:, :, 3][mask] = im_flooded[1:-1, 1:-1][mask].astype(np.uint8)
        im[(im[:, :] == [0, 0, 0, 255]).all(2)] = [brightness, brightness, brightness, 255]  # Optimal
        return Image.fromarray(im)

    def slice_image(sprite, color_slice: slice):
        img = np.array(sprite)
        colors = liquify.get_colors_unsorted(img)
        if len(colors) > 1:
            colors = list(sorted(
                colors,
                key=lambda color: liquify.count_instances_of_color(img, color),
                reverse=True
            ))
            try:
                selection = np.arange(len(colors))[color_slice]
            except IndexError:
                raise AssertionError(f'The color slice `{color_slice}` is invalid.')
            if isinstance(selection, np.ndarray):
                selection = selection.flatten().tolist()
            else:
                selection = [selection]
            # Modulo the value field
            positivevalue = [(color % len(colors)) for color in selection]
            # Remove most used color
            for color_index, color in enumerate(colors):
                if color_index not in positivevalue:
                    img = liquify.remove_instances_of_color(img, color)

        return Image.fromarray(img)

    @add_variant("csel", "c")
    def color_select(sprite, *index: int):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, list(index))

    @add_variant("cslice", "cs")
    def color_slice(sprite, start: Optional[int] = None, stop: Optional[int] = None,
                    step: Optional[int] = None):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, slice(start, stop, step))

    @add_variant("abberate")  # misspelling alias because i misspell it all the time
    def aberrate(sprite, x: int, y: int):
        """Abberates the colors of a sprite."""
        arr = np.array(sprite)
        arr = np.pad(arr, ((abs(y), abs(y)), (abs(x), abs(x)), (0, 0)))
        arr[:, :, 0] = np.roll(arr[:, :, 0], -x, 1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], x, 1)
        arr[:, :, 0] = np.roll(arr[:, :, 0], -y, 0)
        arr[:, :, 2] = np.roll(arr[:, :, 2], y, 0)
        arr = arr.astype(np.uint16)
        arr[:, :, 3] += np.roll(np.roll(arr[:, :, 3], -x, 1), -y, 0)
        arr[:, :, 3] += np.roll(np.roll(arr[:, :, 3], x, 1), y, 0)
        arr[arr > 255] = 255
        return Image.fromarray(arr.astype(np.uint8))

    @add_variant()
    def opacity(sprite, amount: float):
        """Sets the opacity of the sprite, from 0 to 1."""
        processable_sprite = np.array(sprite, dtype=np.float64)
        processable_sprite[:, :, 3] *= amount
        return Image.fromarray(np.clip(processable_sprite, 0, 255).astype(np.uint8))

    @add_variant("neg")
    def negative(sprite):
        """Inverts the sprite's RGB values."""
        arr = np.array(sprite)
        arr[:, :, :3] = ~arr[:, :, :3]
        return Image.fromarray(arr)

    @add_variant("hs")
    def hueshift(sprite, angle: int):
        """Shifts the hue of the sprite. 0 to 360."""
        arr = np.array(sprite)
        arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
        hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + int(angle // 2), 180)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(np.dstack((rgb, arr_a)))

    @add_variant("gamma", "g")
    def brightness(sprite, brightness: float):
        """Sets the brightness of the sprite. Can go above 1.0, does nothing below 0.0."""
        arr = np.array(sprite, dtype=np.float64)
        arr[:, :, :3] *= brightness
        arr = arr.clip(0.0, 255.0)
        return Image.fromarray(arr.astype(np.uint8))

    @add_variant("ps")
    def palette_snap(sprite, *, tile, wobble, renderer):
        """Snaps all the colors in the tile to the specified palette."""
        palette_colors = np.array(renderer.palette_cache[tile.palette].convert("RGB")).reshape(-1, 3)
        im = np.array(sprite)
        im_lab = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        diff_matrix = np.full((palette_colors.shape[0], *im.shape[:-1]), 999)
        for i, color in enumerate(palette_colors):
            filled_color_array = np.array([[color]]).repeat(
                im.shape[0], 0).repeat(im.shape[1], 1)
            filled_color_array = cv2.cvtColor(
                filled_color_array.astype(
                    np.float32) / 255, cv2.COLOR_RGB2Lab)
            im_delta_e = np.sqrt(np.sum((im_lab - filled_color_array) ** 2, axis=-1))
            diff_matrix[i] = im_delta_e
        min_indexes = np.argmin(diff_matrix, 0, keepdims=True).reshape(
            diff_matrix.shape[1:])
        result = np.full(im.shape, 0, dtype=np.uint8)
        for i, color in enumerate(palette_colors):
            result[:, :, :3][min_indexes == i] = color
        result[:, :, 3] = im[:, :, 3]
        return Image.fromarray(result)

    @add_variant("sat", "grayscale", "gscale")
    def saturation(sprite, saturation: float):
        """Saturates or desaturates a sprite."""
        return Image.blend(sprite, sprite.convert("LA").convert("RGBA"), 1.0 - saturation)

    @add_variant()
    def blank(sprite):
        """Sets a sprite to pure white."""
        arr = np.array(sprite.convert("RGBA"))
        arr[:, :, :3] = 255
        return Image.fromarray(arr)

    @add_variant("nl")
    def normalize_lightness(sprite):
        """Normalizes a sprite's HSL lightness, bringing the lightest value up to full brightness."""
        arr = np.array(sprite)
        arr_rgb, sprite_a = arr[:, :, :3], arr[:, :, 3]
        arr_hls = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HLS).astype(np.float64)  # since WHEN was it HLS???? huh?????
        max_l = np.max(arr_hls[:, :, 1])
        arr_hls[:, :, 1] *= (255 / max_l)
        sprite_rgb = cv2.cvtColor(arr_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)  # my question still stands
        return Image.fromarray(np.dstack((sprite_rgb, sprite_a)))

    @add_variant("3oo", "skul")
    def threeoo(sprite, scale: float):
        """Content-aware scales the sprite downwards."""
        assert 0, "i still need to find a way of doing this that isn't O(n^3) -balt\n(If you're seeing this outside of beta, PLEASE go yell at the dev.)"

    @add_variant()
    def crop(sprite, x_y: list[int, int], u_v: list[int, int], true: Optional[bool] = False):
        """Crops the sprite to the specified bounding box."""
        (x, y), (u, v) = x_y, u_v
        cropped = sprite.crop((x, y, x + u, y + v))
        if true:
            return cropped
        else:
            im = Image.new(
                'RGBA', (sprite.width, sprite.height), (0, 0, 0, 0))
            im.paste(cropped, (x, y))
            return im

    @add_variant()
    def snip(sprite, x_y: list[int, int], u_v: list[int, int], true: Optional[bool] = False):
        """Snips the specified box out from the sprite."""
        dummy = Image.new(sprite.mode, u_v)
        sprite.paste(dummy, x_y)
        return snip

    @add_variant()
    def croppoly(sprite, *x_y: list[int, int]):
        """Crops the sprite to the specified polygon."""
        arr = np.array(sprite)
        pts = np.array([x_y], dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        print(pts)
        clip_poly = cv2.fillPoly(np.zeros(arr.shape[:2], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        arr = np.multiply(arr, clip_poly, casting="unsafe")
        return Image.fromarray(arr.astype(np.uint8))

    @add_variant("cvt")
    def convert(sprite, direction: Literal["to", "from"],
                space: Literal["BGR", "HSV", "HLS", "YUV", "YCrCb", "XYZ", "Lab", "Luv"]):
        """Converts the sprite's color space to or from RGB. Mostly for use with :matrix."""
        arr = np.array(sprite)
        space_conversion = {
            "to": {
                "BGR": cv2.COLOR_RGB2BGR,
                "HSV": cv2.COLOR_RGB2HSV,
                "HLS": cv2.COLOR_RGB2HLS,
                "YUV": cv2.COLOR_RGB2YUV,
                "YCrCb": cv2.COLOR_RGB2YCrCb,
                "XYZ": cv2.COLOR_RGB2XYZ,
                "Lab": cv2.COLOR_RGB2Lab,
                "Luv": cv2.COLOR_RGB2Luv,
            },
            "from": {
                "BGR": cv2.COLOR_BGR2RGB,
                "HSV": cv2.COLOR_HSV2RGB,
                "HLS": cv2.COLOR_HLS2RGB,
                "YUV": cv2.COLOR_YUV2RGB,
                "YCrCb": cv2.COLOR_YCrCb2RGB,
                "XYZ": cv2.COLOR_XYZ2RGB,
                "Lab": cv2.COLOR_Lab2RGB,
                "Luv": cv2.COLOR_Luv2RGB,
            }
        }
        arr_converted = cv2.cvtColor(arr[:, :, :3], space_conversion[direction][space])
        return Image.fromarray(np.dstack((arr_converted, arr[:, :, 3])))

    @add_variant()
    def snip(sprite, x_y: list[int, int], u_v: list[int, int]):
        """Snips the specified bounding box out of the sprite."""
        (x, y), (u, v) = x_y, u_v
        dummy = Image.new("RGBA", (u, v))
        im = sprite.copy()
        im.paste(dummy, (x, y))
        return im

    @add_variant()
    def threshold(sprite, r: float, g: Optional[float] = None, b: Optional[float] = None, a: Optional[float] = 0.0):
        """Removes all pixels below a threshold.
This can be used in conjunction with blur, opacity, and additive blending to create a bloom effect!
If a value is negative, it removes pixels above the threshold instead."""
        g = r if g is None else g
        b = r if b is None else b
        im_r, im_g, im_b, im_a = np.split(np.array(sprite), 4, axis=2)
        # Could use np.logical_or, but that's much less readable for very little performance gain
        im_a[np.copysign(im_r, r) < r * 255] = 0
        im_a[np.copysign(im_g, g) < g * 255] = 0
        im_a[np.copysign(im_b, b) < b * 255] = 0
        im_a[np.copysign(im_a, a) < a * 255] = 0
        return Image.fromarray(np.dstack((im_r, im_g, im_b, im_a)))

    @add_variant()
    def blur(sprite, radius: int, gaussian: Optional[bool] = False):
        """Blurs a sprite. Uses box blur by default, though gaussian blur can be used with the boolean toggle."""
        arr = np.pad(np.array(sprite), ((radius, radius), (radius, radius), (0, 0)))
        assert radius > 0, f"Blur radius of {radius} is too small!"
        if gaussian:
            arr = cv2.GaussianBlur(arr, (radius * 2 + 1, radius * 2 + 1), 0)
        else:
            arr = cv2.boxFilter(arr, -1, (radius * 2 + 1, radius * 2 + 1))
        return Image.fromarray(arr)

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
