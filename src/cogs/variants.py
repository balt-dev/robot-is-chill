import glob
import inspect
import math
import types
import typing
from typing import Any, Literal, Optional, Union, get_origin, get_args, Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
import visual_center
from PIL import ImageChops, Image, ImageOps

from . import liquify
from ..utils import recolor, composite
from .. import constants
from ..types import Variant, RegexDict, VaryingArgs, Color, Slice

CARD_KERNEL = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)))
OBLQ_KERNEL = np.array(((1, 0, 1), (0, 0, 0), (1, 0, 1)))
EDGE_KERNEL = np.array(((1, 1, 1), (1, -8, 1), (1, 1, 1)))

def class_init(self, *args, **kwargs):
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
        elif curr_type is Slice:
            g = v[:3]
            del v[:3]
            val = Slice(*[int(i) if len(i) > 0 else None for i in g[0].split("/")])
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
                bool: r"/(true|false)?",
                Slice: r"/\((-?\d*(?:/-?\d*(?:/-?\d*)?)?)\)",
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
        for name, param in dict(params).items():
            if param.annotation == inspect.Parameter.empty:
                continue
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
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
        syntax = (f"\u001b[0;30m[\u001b[0;35m{'[0;30m|[0;35m'.join(aliases)}[0;30m]" if len(
            aliases) else "") + generate_syntax(params)
        class_name = func.__name__.replace("_", " ").title().replace(" ", "") + "Variant"
        variant_type = tuple(params.keys())[0]
        type_tree = get_type_tree([p for p in params.values() if p.kind != inspect.Parameter.KEYWORD_ONLY])[1:]
        variant = type(
            class_name,
            (Variant,),
            {
                "__init__": class_init,
                "__doc__": func.__doc__,
                "__repr__": (lambda self: f"{self.__class__.__name__}{self.args}"),
                "apply": (lambda self, obj, **kwargs:
                          func(obj, *self.args, **(self.kwargs | kwargs) if has_kwargs else self.kwargs)),
                "pattern": pattern,
                "signature": type_tree,
                "syntax": syntax,
                "type": variant_type,
            }
        )
        bot.variants.append(variant)
        return variant

    def add_variant(*aliases, no_function_name=False, debug=False):
        def wrapper(func):
            v = create_variant(func, aliases, no_function_name)
            if debug:
                print(f"""{v.__name__}:
    pattern: {v.pattern},
    type: {v.type},
    syntax: {v.syntax}""")
            return func

        return wrapper

    # --- SPECIAL ---

    @add_variant("", "noop")
    def nothing(tile):
        """Does nothing. Useful for resetting persistent variants."""
        pass

    @add_variant(no_function_name=True)
    def blending(tile, mode: Literal[*constants.BLENDING_MODES]):
        """Sets the blending mode for the tile."""
        tile.blending = mode

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

    @add_variant("f")
    def freeze(tile, frame: Optional[int] = 1):
        """Freezes the wobble of the tile to the specified frame."""
        assert frame in range(1, 4), f"Wobble frame of `{frame}` is outside of the supported range!"
        tile.wobble = frame - 1

    # --- COLORING ---

    palette_names = tuple([Path(p).stem for p in glob.glob("data/palettes/*.png")])

    @add_variant("palette/", "p!", no_function_name=True)
    def palette(tile, palette: str):
        """Sets the tile's palette."""
        assert palette in palette_names, f"Palette `{palette}` was not found!"
        tile.palette = palette

    @add_variant("ac", "~")
    def apply(sprite, *, tile, wobble, renderer):
        """Immediately applies the sprite's default color."""
        tile.custom_color = True
        rgba = *renderer.palette_cache[tile.palette].getpixel(tile.color), 0xFF
        sprite = recolor(sprite, rgba)
        return sprite

    @add_variant(no_function_name=True)
    def color(sprite, color: Color, *, tile, wobble, renderer, _default_color=False):
        """Sets the tile's color.
Can take:
- A hexadecimal RGB/RGBA value, as #RGB, #RGBA, #RRGGBB, or #RRGGBBAA
- A color name, as is (think the color properties from the game)
- A palette index, as an x/y coordinate"""
        if _default_color:
            if tile.custom_color:
                return sprite
        else:
            tile.custom_color = True
        if len(color) == 4:
            rgba = color
        else:
            rgba = *renderer.palette_cache[tile.palette].getpixel(color), 0xFF
        return recolor(sprite, rgba)

    @add_variant("grad")
    def gradient(sprite, color: Color, angle: Optional[float] = 0.0, width: Optional[float] = 1.0,
                 offset: Optional[float] = 0, steps: Optional[int] = 0, raw: Optional[bool] = False,
                 extrapolate: Optional[bool] = False, *, tile, wobble, renderer):
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
        grad = np.mgrid[offset:width + offset:maxside * 1j]
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
            mult_grad[:, :, :3] = cv2.cvtColor(mult_grad[:, :, :3].astype(np.uint8), cv2.COLOR_Luv2RGB).astype(
                np.float64)
        mult_grad /= 255
        return (sprite * mult_grad).astype(np.uint8)

    @add_variant("overlay/", "o!", no_function_name=True)
    def overlay(sprite, overlay: str, *, tile, wobble, renderer):
        """Applies an overlay to a sprite."""
        tile.custom_color = True
        assert overlay in renderer.overlay_cache, f"`{overlay}` isn't a valid overlay!"
        overlay_image = renderer.overlay_cache[overlay]
        tile_amount = np.ceil(overlay_image.shape / sprite.shape)
        overlay_image = np.tile(overlay_image, (*tile_amount, 1))[:sprite.shape[0], :sprite.shape[1]]
        return ImageChops.multiply(sprite, overlay_image)

    # --- TEXT MANIPULATION ---

    @add_variant("noun", "prop")
    def property(sprite, plate: Optional[Literal["blank", "left", "up", "right", "down", "turn", "deturn"]] = None, *,
                 tile, wobble, renderer):
        """Applies a property plate to a sprite."""
        if plate is None:
            plate = tile.frame if tile.altered_frame else None
        else:
            plate = {v: k for k, v in constants.DIRECTIONS.items()}[plate]
        sprite = sprite[:, :, 3] > 0
        plate, box = renderer.bot.db.plate(plate, wobble)
        size = (max(sprite.shape[1], plate.width), max(sprite.shape[0], plate.height))
        dummy = Image.new("1", size)
        delta = (plate.size[0] - sprite.size[0]) // 2, \
                (plate.size[0] - sprite.size[1]) // 2
        plate_dummy = Image.new("1", size)
        plate_dummy.paste(
            plate.getchannel("A").convert("1", dither=Image.NONE),
            (max((sprite.shape[1] - plate.width) // 2, 0), max((sprite.shape[0] - plate.height) // 2, 0))
        )
        dummy.paste(
            Image.new("1", sprite.shape[::-1], 1),
            delta,
            Image.fromarray(sprite)
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

    @add_variant("rot")
    def rotate(sprite, angle: float, expand: Optional[bool] = False):
        """Rotates a sprite."""
        if expand:
            scale = math.cos(math.radians(-angle)) + math.sin(math.radians(-angle))
            sprite = np.pad(sprite, ((int(sprite.shape[0] * ((scale - 1) / 2)), int(sprite.shape[0] * ((scale - 1) / 2))),
                               (int(sprite.shape[0] * ((scale - 1) / 2)), int(sprite.shape[0] * ((scale - 1) / 2))),
                               (0, 0)))
        image_center = tuple(np.array(sprite.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        return cv2.warpAffine(sprite, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_NEAREST)

    @add_variant()
    def scale(sprite, w: float, h: Optional[float] = None):
        """Scales a sprite by the given multipliers."""
        if h is None:
            h = w
        if int(w * sprite.shape[0]) <= 0 or int(h * sprite.shape[1]) <= 0:
            raise AssertionError(
                f"Can't scale a tile to `{int(w * sprite.shape[0])}x{int(h * sprite.shape[1])}`, as it has a non-positive target area.")
        dim = sprite.shape[:2] * np.array((h, w))
        dim = dim.astype(int)
        return cv2.resize(sprite, dim[::-1], interpolation=cv2.INTER_NEAREST)

    @add_variant()
    def pad(sprite, left: int, top: int, right: int, bottom: int):
        """Pads the sprite by the specified values."""
        return np.pad(sprite, ((top, bottom), (left, right), (0, 0)))

    @add_variant()
    def pixelate(sprite, x: int, y: Optional[int] = None):
        """Pixelates the sprite."""
        if y is None:
            y = x
        old_size = sprite.shape[:2]
        size = sprite.shape[:2] // (y, x)
        return cv2.resize(sprite, size, interpolation=Image.NEAREST).resize(sprite, old_size, interpolation=Image.NEAREST)

    @add_variant("m")
    def meta(sprite, level: Optional[int] = 1) -> Image.Image:
        """Applies a meta filter to an image."""
        assert abs(level) <= constants.MAX_META_DEPTH, f"Meta depth of {level} too large!"
        orig = sprite.copy()
        base = sprite[:, :, 3]
        if level < 0:
            level = abs(level)
            base = ImageOps.invert(base)
        for _ in range(level):
            base = cv2.filter2D(src=base, ddepth=0, kernel=EDGE_KERNEL)
        base = np.dstack((base, base, base, base))
        return base

    @add_variant()
    def land(post, direction: Optional[Literal["left", "top", "right", "bottom"]] = "bottom"):
        """Removes all space between the tile and its bounding box on the specified side."""
        frame = next(filter(lambda f: f is not None, post.frames), None)
        if frame is None:
            return post
        rows = np.any(frame[:, :, 3], axis=1)
        cols = np.any(frame[:, :, 3], axis=0)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(cols)[0][[0, -1]]
        displacement = {"left": left, "top": top, "right": -right, "bottom": -bottom}[direction]
        index = {"left": 0, "top": 1, "right": 0, "bottom": 1}[direction]
        post.displacement[index] += displacement

    @add_variant()
    def warp(sprite, x1_y1: list[int, int], x2_y2: list[int, int], x3_y3: list[int, int], x4_y4: list[int, int]):
        """Warps the sprite by displacing the bounding box's corners.
    Point 1 is top-left, point 2 is top-right, point 3 is bottom-right, and point 4 is bottom-left.
    If the sprite grows past its original bounding box, it will need to be recentered manually."""
        src_shape = np.array(sprite.shape[-2::-1])
        src = (np.array([
            [[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
            [[1.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            [[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.0, 1.0], [0.5, 0.5], [0.0, 0.0]],
        ]) * (src_shape - 1)).astype(np.int32)
        pts = np.array((x1_y1, x2_y2, x3_y3, x4_y4))
        # This package is only 70kb and I'm lazy
        center = visual_center.find_pole(pts, precision=1)[0]
        dst = src + np.array([
            [x1_y1, center, x2_y2],
            [x2_y2, center, x3_y3],
            [x3_y3, center, x4_y4],
            [x4_y4, center, x1_y1],
        ], dtype=np.int32)
        # Set padding values
        before_padding = np.array([
            max(-x1_y1[0], -x4_y4[0], 0),  # Added padding for left
            max(-x1_y1[1], -x2_y2[1], 0)  # Added padding for top
        ])
        after_padding = np.array([
            (max(x2_y2[0], x3_y3[0], 0)),  # Added padding for right
            (max(x3_y3[1], x4_y4[1], 0))  # Added padding for bottom
        ])
        dst += before_padding
        new_shape = (src_shape + before_padding + after_padding).astype(np.uint32)[::-1]
        final_arr = np.zeros((*new_shape, 4), dtype=np.uint8)
        for source, destination in zip(src, dst): # Iterate through the four triangles
            clip = cv2.fillConvexPoly(np.zeros(new_shape, dtype=np.uint8), destination, 1).astype(bool)
            M = cv2.getAffineTransform(source.astype(np.float32), destination.astype(np.float32))
            warped_arr = cv2.warpAffine(sprite, M, new_shape[::-1], flags=cv2.INTER_NEAREST)
            final_arr[clip] = warped_arr[clip]
        return Image.fromarray(final_arr)

    @add_variant("mm")
    def matrix(sprite,
               aa_ab_ac_ad: list[float, float, float, float],
               ba_bb_bc_bd: list[float, float, float, float],
               ca_cb_cc_cd: list[float, float, float, float],
               da_db_dc_dd: list[float, float, float, float]):
        """Multiplies the sprite by the given RGBA matrix."""
        matrix = np.array((aa_ab_ac_ad, ba_bb_bc_bd, ca_cb_cc_cd, da_db_dc_dd)).T
        img = sprite.astype(np.float64) / 255.0
        immul = img.reshape(-1, 4) @ matrix  # @ <== matmul
        immul = (np.clip(immul, 0.0, 1.0) * 255).astype(np.uint8)
        return immul.reshape(img.shape)

    @add_variant()
    def neon(sprite, strength: Optional[float] = 0.714):
        """Darkens the inside of each region of color."""
        # This is approximately 2.14x faster than Charlotte's neon, profiling at strength 0.5 with 2500 iterations on baba/frog_0_1.png.
        unique_colors = liquify.get_colors(sprite)
        final_mask = np.ones(sprite.shape[:2], dtype=np.float64)
        for color in unique_colors:
            mask = (sprite == color).all(axis=2)
            float_mask = mask.astype(np.float64)
            card_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=CARD_KERNEL)
            oblq_mask = cv2.filter2D(src=float_mask, ddepth=-1, kernel=OBLQ_KERNEL)
            final_mask[card_mask == 4] -= strength / 2
            final_mask[oblq_mask == 4] -= strength / 2
        if strength < 0:
            final_mask = np.abs(1 - final_mask)
        sprite[:, :, 3] *= np.clip(final_mask, 0, 1)
        return sprite.astype(np.uint8)

    @add_variant()
    def scan(sprite, axis: Literal["x", "y"], on: Optional[int] = 1, off: Optional[int] = 1, offset: Optional[int] = 0):
        """Removes rows or columns of pixels to create a scan line effect."""
        assert on >= 0 and off >= 0 and on + off > 0, f"Scan mask of `{on}` on and `{off}` off is invalid!"
        axis = ("y", "x").index(axis)
        mask = np.roll(np.array([1] * on + [0] * off, dtype=np.uint8), offset)
        mask = np.tile(mask, (
            sprite.shape[1 - axis],
            int(math.ceil(sprite.shape[axis] / mask.shape[0]))
        ))[:, :sprite.shape[axis]]
        if not axis:
            mask = mask.T
        return sprite[:, :, 3] * mask

    @add_variant()
    def flip(sprite, *axis: Literal["x", "y"]):
        """Flips the sprite along the specified axes."""
        for a in axis:
            if a == "x":
                sprite = sprite[:, ::-1, :]
            else:
                sprite = sprite[::-1, :, :]
        return sprite

    @add_variant()
    def mirror(sprite, axis: Literal["x", "y"], half: Literal["back", "front"]):
        """Mirrors the sprite along the specified direction."""
        # NOTE: This code looks ugly, but it's fast and worked first try.
        if axis == "x":
            sprite = np.rot90(sprite)
        if half == "front":
            sprite = np.flipud(sprite)
        sprite[:sprite.shape[0] // 2] = sprite[:sprite.shape[0] // 2 - 1:-1]
        if half == "front":
            sprite = np.flipud(sprite)
        if axis == "x":
            sprite = np.rot90(sprite, -1)
        return sprite

    @add_variant("norm")
    def normalize(sprite):
        """Centers the sprite on its visual bounding box."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(cols)[0][[0, -1]]
        sprite_center = sprite.shape // 2
        center = int((top + bottom) // 2), int((left + right) // 2)
        displacement = sprite_center[0] - center[0] + top, sprite_center[1] - center[1] + left
        return np.roll(sprite, displacement)

    @add_variant("disp")
    def displace(tile, x: int, y: int):
        """Displaces the tile by the specified coordinates."""
        tile.displacement = [tile.displacement[0] - x, tile.displacement[1] - y]

    # Original code by Charlotte (CenTdemeern1)
    @add_variant("flood")
    def floodfill(sprite, brightness: Optional[float] = 1.0, inside: Optional[bool] = False):
        """Floodfills either inside or outside a sprite with a given brightness value."""
        brightness = int(brightness * 255)
        sprite[sprite[:, :, 3] == 0] = 0  # Optspriteal
        sprite_alpha = sprite[:, :, 3]  # Stores the alpha channel separately
        sprite_alpha[sprite_alpha > 0] = -1  # Sets all nonzero numbers to a number that's neither 0 nor 255.
        # Pads the alpha channel by 1 on each side to allow flowing past
        # where the sprite touches the edge of the bounding box.
        sprite_alpha = np.pad(sprite_alpha, ((1, 1), (1, 1)))
        sprite_flooded = cv2.floodFill(
            spriteage=sprite_alpha,
            mask=None,
            seedPoint=(0, 0),
            newVal=255
        )[1]
        mask = sprite_flooded != (0 if inside else 255)
        sprite_flooded[mask] = (255 if inside else 0)
        mask = mask[1:-1, 1:-1]
        if not inside:
            sprite_flooded = 255 - sprite_flooded
        # Crops the alpha channel back to the original size and positioning
        sprite[:, :, 3][mask] = sprite_flooded[1:-1, 1:-1][mask].astype(np.uint8)
        sprite[(sprite[:, :] == [0, 0, 0, 255]).all(2)] = [brightness, brightness, brightness, 255]  # Optspriteal
        return sprite

    def slice_image(sprite, color_slice: slice):
        colors = liquify.get_colors(sprite)
        if len(colors) > 1:
            colors = list(sorted(
                colors,
                key=lambda color: liquify.count_instances_of_color(sprite, color),
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
                    sprite = liquify.remove_instances_of_color(sprite, color)
        return sprite

    @add_variant("csel", "c")
    def color_select(sprite, *index: int):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, list(index))

    @add_variant("cslice", "cs")
    def color_slice(sprite, s: Slice):
        """Keeps only the slice of colors, indexed by their occurrence. This changes per-frame, not per-tile.
Slices are notated as [30m([36mstart[30m|[36mstop[30m|[36mstep[30m)[0m, with stop and step being omittable."""
        return slice_image(sprite, s.slice)

    @add_variant("cshift", "csh")
    def color_shift(sprite, s: Slice):
        """Shifts the colors of a sprite around, by index of occurence."""
        unique_colors = liquify.get_colors(sprite)
        unique_colors = np.spriteay(
            sorted(unique_colors, key=lambda color: liquify.count_instances_of_color(sprite, color), reverse=True))
        final_sprite = np.tile(sprite, (len(unique_colors), 1, 1, 1))
        mask = np.equal(final_sprite[:, :, :, :], unique_colors.reshape((-1, 1, 1, 4))).all(axis=3)
        out = np.zeros(sprite.shape)
        for i, color in enumerate(unique_colors[s.slice]):
            out += np.tile(mask[i].T, (4, 1, 1)).T * color
        return out.astype(np.uint8)

    @add_variant("abberate")  # misspelling alias because i misspell it all the time
    def aberrate(sprite, x: int, y: int):
        """Abberates the colors of a sprite."""
        sprite = np.pad(sprite, ((abs(y), abs(y)), (abs(x), abs(x)), (0, 0)))
        sprite[:, :, 0] = np.roll(sprite[:, :, 0], -x, 1)
        sprite[:, :, 2] = np.roll(sprite[:, :, 2], x, 1)
        sprite[:, :, 0] = np.roll(sprite[:, :, 0], -y, 0)
        sprite[:, :, 2] = np.roll(sprite[:, :, 2], y, 0)
        sprite = sprite.astype(np.uint16)
        sprite[:, :, 3] += np.roll(np.roll(sprite[:, :, 3], -x, 1), -y, 0)
        sprite[:, :, 3] += np.roll(np.roll(sprite[:, :, 3], x, 1), y, 0)
        sprite[sprite > 255] = 255
        return sprite

    @add_variant()
    def opacity(sprite, amount: float):
        """Sets the opacity of the sprite, from 0 to 1."""
        sprite[:, :, 3] = np.multiply(sprite[:, :, 3], amount, casting="unsafe").astype(np.uint8)
        return sprite

    @add_variant("neg")
    def negative(sprite):
        """Inverts the sprite's RGB values."""
        sprite[:, :, :3] = ~sprite[:, :, :3]
        return sprite

    @add_variant()
    def wrap(sprite, x: int, y: int):
        """Wraps the sprite around its image box."""
        return np.roll(sprite, (y, x), (0, 1))

    @add_variant()
    def melt(sprite, side: Optional[Literal["left", "top", "right", "bottom"]] = "bottom"):
        """Removes transparent pixels from each row/column and shifts the remaining ones to the end."""
        is_vertical = side in ("top", "bottom")
        at_end = side in ("right", "bottom")
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        # NOTE: I couldn't find a way to do this without at least one Python loop :/
        for i in range(sprite.shape[0]):
            sprite_slice = sprite[i, sprite[i, :, 3] != 0]
            sprite[i] = np.pad(sprite_slice, ((sprite[i].shape[0] - sprite_slice.shape[0], 0)[::2 * at_end - 1], (0, 0)))
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        return sprite

    @add_variant()
    def wave(sprite, axis: Literal["x", "y"], amplitude: int, offset: int, frequency: float):
        """Displaces the sprite by a wave. All values are percentages of the sprite's heighu"""
        if axis == "y":
            sprite = np.rot90(sprite)
        offset = ((np.sin(np.linspace(offset, np.pi * 2 * frequency + offset, sprite.shape[0])) / 2) * amplitude).astype(
            int)
        # NOTE: np.roll can't be element wise :/
        sprite[:] = sprite[np.mod(np.arange(sprite.shape[0]) + offset, sprite.shape[0])]
        if axis == "y":
            sprite = np.rot90(sprite, -1)
        return sprite

    @add_variant("hs")
    def hueshift(sprite, angle: int):
        """Shifts the hue of the sprite. 0 to 360."""
        hsv = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + int(angle // 2), 180)
        sprite[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sprite

    @add_variant("gamma", "g")
    def brightness(sprite, brightness: float):
        """Sets the brightness of the sprite."""
        sprite[:, :, :3] *= brightness
        sprite = sprite.clip(-255.0, 255.0) % 255
        return sprite

    @add_variant("ps")
    def palette_snap(sprite, *, tile, wobble, renderer):
        """Snaps all the colors in the tile to the specified palette."""
        palette_colors = np.array(renderer.palette_cache[tile.palette].convert("RGB")).reshape(-1, 3)
        sprite_lab = cv2.cvtColor(sprite.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        diff_matrix = np.full((palette_colors.shape[0], *sprite.shape[:-1]), 999)
        for i, color in enumerate(palette_colors):
            filled_color_array = np.array([[color]]).repeat(
                sprite.shape[0], 0).repeat(sprite.shape[1], 1)
            filled_color_array = cv2.cvtColor(
                filled_color_array.astype(
                    np.float32) / 255, cv2.COLOR_RGB2Lab)
            sprite_delta_e = np.sqrt(np.sum((sprite_lab - filled_color_array) ** 2, axis=-1))
            diff_matrix[i] = sprite_delta_e
        min_indexes = np.argmin(diff_matrix, 0, keepdsprites=True).reshape(
            diff_matrix.shape[1:])
        result = np.full(sprite.shape, 0, dtype=np.uint8)
        for i, color in enumerate(palette_colors):
            result[:, :, :3][min_indexes == i] = color
        result[:, :, 3] = sprite[:, :, 3]
        return result

    @add_variant("sat", "grayscale", "gscale")
    def saturation(sprite, saturation: Optional[float] = 0):
        """Saturates or desaturates a sprite."""
        return composite(sprite, sprite.convert("LA").convert("RGBA"), 1.0 - saturation)

    @add_variant()
    def blank(sprite):
        """Sets a sprite to pure white."""
        sprite[:, :, :3] = 255
        return sprite

    @add_variant("nl")
    def normalize_lightness(sprite):
        """Normalizes a sprite's HSL lightness, bringing the lightest value up to full brightness."""
        arr_hls = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HLS).astype(np.float64)  # since WHEN was it HLS???? huh?????
        max_l = np.max(arr_hls[:, :, 1])
        arr_hls[:, :, 1] *= (255 / max_l)
        sprite[:, :, :3] = cv2.cvtColor(arr_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)  # my question still stands
        return sprite

    @add_variant("3oo", "skul")
    def threeoo(sprite, scale: float):
        """Content-aware scales the sprite downwards."""
        assert 0, "i still need to find a way of doing this that isn't O(n^3) -balt\n(If you're seeing this outside of beta, PLEASE go yell at the dev.)"

    @add_variant()
    def crop(sprite, x_y: list[int, int], u_v: list[int, int], true: Optional[bool] = False):
        """Crops the sprite to the specified bounding box."""
        (x, y), (u, v) = x_y, u_v
        if true:
            return sprite[y:v, x:u]
        else:
            dummy = np.zeros_like(sprite)
            dummy[y:v, x:u] = sprite[y:v, x:u]
            return dummy

    @add_variant()
    def snip(sprite, x_y: list[int, int], u_v: list[int, int], true: Optional[bool] = False):
        """Snips the specified box out from the sprite."""
        (x, y), (u, v) = x_y, u_v
        sprite[y:v, x:u] = 0
        return sprite

    @add_variant()
    def croppoly(sprite, *x_y: list[int, int]):
        """Crops the sprite to the specified polygon."""
        pts = np.array([x_y], dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[:2], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        return np.multiply(sprite, clip_poly, casting="unsafe").astype(np.uint8)

    @add_variant("cvt")
    def convert(sprite, direction: Literal["to", "from"],
                space: Literal["BGR", "HSV", "HLS", "YUV", "YCrCb", "XYZ", "Lab", "Luv"]):
        """Converts the sprite's color space to or from RGB. Mostly for use with :matrix."""
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
        sprite[:, :, :3] = cv2.cvtColor(sprite[:, :, :3], space_conversion[direction][space])
        return sprite

    @add_variant()
    def threshold(sprite, r: float, g: Optional[float] = None, b: Optional[float] = None, a: Optional[float] = 0.0):
        """Removes all pixels below a threshold.
This can be used in conjunction with blur, opacity, and additive blending to create a bloom effect!
If a value is negative, it removes pixels above the threshold instead."""
        g = r if g is None else g
        b = r if b is None else b
        im_r, im_g, im_b, im_a = np.split(sprite, 4, axis=2)
        # Could use np.logical_or, but that's much less readable for very little performance gain
        im_a[np.copysign(im_r, r) < r * 255] = 0
        im_a[np.copysign(im_g, g) < g * 255] = 0
        im_a[np.copysign(im_b, b) < b * 255] = 0
        im_a[np.copysign(im_a, a) < a * 255] = 0
        return np.dstack((im_r, im_g, im_b, im_a))

    @add_variant()
    def blur(sprite, radius: int, gaussian: Optional[bool] = False):
        """Blurs a sprite. Uses box blur by default, though gaussian blur can be used with the boolean toggle."""
        arr = np.pad(sprite, ((radius, radius), (radius, radius), (0, 0)))
        assert radius > 0, f"Blur radius of {radius} is too small!"
        if gaussian:
            arr = cv2.GaussianBlur(arr, (radius * 2 + 1, radius * 2 + 1), 0)
        else:
            arr = cv2.boxFilter(arr, -1, (radius * 2 + 1, radius * 2 + 1))
        return arr

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
