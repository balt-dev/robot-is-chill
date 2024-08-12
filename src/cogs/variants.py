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

from . import liquify
from ..utils import recolor, composite
from .. import constants, errors
from ..types import Variant, RegexDict, VaryingArgs, Color, Slice

CARD_KERNEL = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)))
OBLQ_KERNEL = np.array(((1, 0, 1), (0, 0, 0), (1, 0, 1)))
META_KERNELS = {
    "full": np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]]),
    "edge": np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
}


def class_init(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs


def parse_signature(v: list[str], t: list[type | types.GenericAlias]) -> list[Any]:
    out = []
    t = list(t).copy()
    v = list(v).copy()
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
            val = Slice(*[int(i) if len(i) > 0 else None for i in g])
        else:
            try:
                raw_val = v.pop(0)
                if curr_type is float or curr_type is int:
                    raw_val = "-" * (raw_val.count("-") % 2) + raw_val.lstrip("-")
                val = curr_type(raw_val)
            except ValueError:
                val = None
        out.append(val)
    return out


def check_size(*dst_size):
    if dst_size[0] > constants.MAX_TILE_SIZE or dst_size[1] > constants.MAX_TILE_SIZE:
        raise errors.TooLargeTile(dst_size)


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
                pattern += rf"\/?\(?{generate_pattern([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno) for name, anno in zip(p.name.split('_'), get_args(p.annotation))])}\)?"
            elif get_origin(p.annotation) == Union:
                pattern += f"(?:{'/' if i - 1 else ''}{generate_pattern([inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=get_args(p.annotation)[0])])})?"
            elif p.annotation in (patterns := {
                int: r"/(-*\d+)",
                float: r"/(-*(?:(?:\d+\.?\d*)|(?:\.\d+)))",
                # From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers/42629198#42629198
                str: r"/(.+?)",
                bool: r"/(true|false)?",
                Slice: r"/\(?(?:(-*\d*)(?:/(-*\d*)(?:/(-*\d*))?)?)?\)?",
                Color: rf"/((?:#(?:[0-9A-Fa-f]{{2}}){{3,4}})|"
                       rf"(?:#(?:[0-9A-Fa-f]){{3,4}})|"
                       rf"(?:{'|'.join(constants.COLOR_NAMES.keys())})|"
                       rf"(?:-*\d+\/-*\d+)|"
                       rf"\((?:-*\d+\/-*\d+)\))"
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
                syntax += f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m/" \
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

    def create_variant(func: Callable, aliases: Iterable[str], no_function_name=False, hashed=True, hidden=False) -> \
    type[Variant]:
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
                "hashed": hashed,
                "hidden": hidden,
                "name": aliases
            }
        )
        bot.variants.append(variant)
        return variant

    def add_variant(*aliases, no_function_name=False, debug=False, hashed=True, hidden=False):
        def wrapper(func):
            v = create_variant(func, aliases, no_function_name, hashed, hidden)
            if debug:
                print(f"""{v.__name__}:
    pattern: {v.pattern},
    type: {v.type},
    syntax: {v.syntax}""")
            return func

        return wrapper

    # --- SPECIAL ---

    @add_variant("", "noop")
    async def nothing(tile):
        """Does nothing. Useful for resetting persistent variants."""
        pass

    @add_variant(no_function_name=True)
    async def blending(post, mode: Literal[*constants.BLENDING_MODES], keep_alpha: Optional[bool] = True):
        """Sets the blending mode for the tile."""
        post.blending = mode
        post.keep_alpha = keep_alpha and mode != "mask"

    @add_variant("m!", no_function_name=True)
    async def macro(tile, name: str):
        """Applies a variant macro to the tile. Check the macros command for details."""
        assert 0, f"Macro `{name}` not found in the database!"

    # --- SIGN TEXT ---

    @add_variant("font!", no_function_name=True)
    async def font(sign, name: Literal[*tuple(Path(f).stem for f in glob.glob('data/fonts/*.ttf'))]):
        """Applies a font to a sign text object."""
        sign.font = name

    @add_variant("scale", no_function_name=True)
    async def sign_scale(sign, size: float):
        """Sets the font size of a sign text object."""
        sign.size *= size

    @add_variant("disp", "displace", no_function_name=True)
    async def sign_displace(sign, x: float, y: float):
        """Displaces a sign text object."""
        sign.xo += x
        sign.yo += y

    @add_variant(no_function_name=True)
    async def sign_color(sign, color: Color, inactive: Optional[Literal["inactive", "in"]] = None, *, bot, ctx):
        """Sets the sign text's color. See the sprite counterpart for details."""
        if len(color) < 4:
            if inactive is not None:
                color = constants.INACTIVE_COLORS[color]
            try:
                color = bot.renderer.palette_cache[ctx.palette].getpixel(color)
            except IndexError:
                raise errors.BadPaletteIndex(sign.text, color)
        sign.color = color

    @add_variant("align!", no_function_name=True)
    async def alignment(sign, alignment: Literal["left", "center", "right"]):
        """Sets the sign text's alignment."""
        sign.alignment = alignment

    @add_variant("anchor!", no_function_name=True)
    async def anchor(sign, anchor: str):
        """Sets the anchor of a sign text. https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html"""
        assert (
            len(anchor) == 2 and
            anchor[0] in ('l', 'm', 'r') and
            anchor[1] in ('a', 'm', 's', 'd')
        ), f"Anchor of `{anchor}` is invalid!"
        sign.anchor = anchor

    @add_variant()
    async def stroke(sign, color: Color, size: int, *, bot, ctx):
        """Sets the sign text's stroke."""
        if len(color) < 4:
            try:
                color = bot.renderer.palette_cache[ctx.palette].getpixel(color)
            except IndexError:
                raise errors.BadPaletteIndex(sign.text, color)
        sign.stroke = color, size

    # --- ANIMATION FRAMES ---

    @add_variant(no_function_name=True)
    async def frame(tile, anim_frame: int):
        """Sets the animation frame of a sprite."""
        tile.altered_frame = True
        tile.frame = anim_frame
        tile.surrounding = 0

    @add_variant(no_function_name=True)
    async def direction(tile, direction: Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]):
        """Sets the direction of a tile."""
        tile.altered_frame = True
        tile.frame = constants.DIRECTION_VARIANTS[direction]

    @add_variant(no_function_name=True)
    async def tiling(tile, tiling: Literal[*tuple(constants.AUTO_VARIANTS.keys())]):
        """Alters the tiling of a tile. Only works on tiles that tile."""
        tile.altered_frame = True
        tile.surrounding |= constants.AUTO_VARIANTS[tiling]

    @add_variant("a", no_function_name=True)
    async def animation_frame(tile, a_frame: int):
        """Sets the animation frame of a tile."""
        tile.altered_frame = True
        tile.frame += a_frame

    @add_variant("s", "sleep", no_function_name=True)
    async def sleep(tile):
        """Makes the tile fall asleep. Only functions correctly on character tiles."""
        tile.altered_frame = True
        tile.frame = (tile.frame - 1) % 32

    @add_variant("f", hashed=False)
    async def frames(tile, *frame: int):
        """Sets the wobble of the tile to the specified frame(s). 1 or 3 can be specified."""
        assert all(f in range(1, 4) for f in frame), f"One or more wobble frames is outside of the supported range of [1, 3]!"
        assert len(frame) <= 3 and len(frame) != 2, "Only 1 or 3 frames can be specified."
        tile.wobble_frames = [f - 1 for f in frame]

    # --- COLORING ---

    palette_names = tuple([Path(p).stem for p in glob.glob("data/palettes/*.png")])

    @add_variant("palette/", "p!", no_function_name=True)
    async def palette(tile, palette: str):
        """Sets the tile's palette. For a list of palettes, try `search type:palette`."""
        assert palette in palette_names, f"Palette `{palette}` was not found!"
        tile.palette = palette

    @add_variant("ac", "~")
    async def apply(sprite, *, tile, wobble, renderer):
        """Immediately applies the sprite's default color."""
        tile.custom_color = True
        rgba = renderer.palette_cache[tile.palette].getpixel(tile.color)
        sprite = recolor(sprite, rgba)
        return sprite

    @add_variant("dcol", "dc", "%")
    async def default_color(tile, color: Color):
        """Overrides the tile's default color."""
        assert len(color) == 2, "Can't override the default with a hexadecimal color!"
        tile.color = tuple(color)

    @add_variant(no_function_name=True)
    async def color(sprite, color: Color, inactive: Optional[Literal["inactive", "in"]] = None, *, tile, wobble, renderer, _default_color = False):
        """Sets the tile's color.
Can take:
- A hexadecimal RGB/RGBA value, as #RGB, #RGBA, #RRGGBB, or #RRGGBBAA
- A color name, as is (think the color properties from the game)
- A palette index, as an x/y coordinate
If [0;36minactive[0m is set and the color isn't hexadecimal, the color will switch to its "inactive" form, which is the color an inactive text object would take on if it had that color in the game."""
        if _default_color:
            if tile.custom_color:
                return sprite
            color = tile.color
        else:
            tile.custom_color = True
        if type(color) is str:
            color = constants.COLOR_NAMES[color]
        if len(color) == 4:
            rgba = color
        else:
            if inactive is not None:
                color = constants.INACTIVE_COLORS[color]
            try:
                rgba = renderer.palette_cache[tile.palette].getpixel(color)
            except IndexError:
                raise errors.BadPaletteIndex(tile.name, color)
        return recolor(sprite, rgba)

    @add_variant("in")
    async def inactive(tile):
        """Applies the color that an inactive text of the tile's color would have.
    This does not work if a color is specified!"""
        tile.color = constants.INACTIVE_COLORS[tile.color]

    bayer_matrix = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ]) / 64

    @add_variant("grad")
    async def gradient(sprite, color: Color, angle: Optional[float] = 0.0, width: Optional[float] = 1.0,
                       offset: Optional[float] = 0, steps: Optional[int] = 0, raw: Optional[bool] = False,
                       extrapolate: Optional[bool] = False, dither: Optional[bool] = False, *, tile, wobble, renderer):
        """Applies a gradient to a tile.
Interpolates color through CIELUV color space by default. This can be toggled with [0;36mraw[0m.
If [0;36mextrapolate[0m is on, then colors outside the gradient will be extrapolated, as opposed to clamping from 0% to 100%.
[0;36Dither[0ming does nothing with [0;36steps[0m set to 0."""
        tile.custom_color = True
        src = Color.parse(tile, renderer.palette_cache)
        dst = Color.parse(tile, renderer.palette_cache, color=color)
        if not raw:
            src = np.hstack((cv2.cvtColor(np.array([[src[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], src[3]))
            dst = np.hstack((cv2.cvtColor(np.array([[dst[:3]]], dtype=np.uint8), cv2.COLOR_RGB2Luv)[0, 0], dst[3]))
        # thank you hutthutthutt#3295 you are a lifesaver
        scale = math.cos(math.radians(angle % 90)) + math.sin(math.radians(angle % 90))
        maxside = max(*sprite.shape[:2]) + 1
        grad = np.mgrid[offset:width + offset:maxside * 1j]
        grad = np.tile(grad[..., np.newaxis], (maxside, 1, 4))
        if not extrapolate:
            grad = np.clip(grad, 0, 1)
        grad_center = maxside // 2, maxside // 2
        rot_mat = cv2.getRotationMatrix2D(grad_center, angle, scale)
        warped_grad = cv2.warpAffine(grad, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_LINEAR)
        if steps:
            if dither:
                needed_size = np.ceil(np.array(warped_grad.shape) / 8).astype(int)
                image_matrix = np.tile(bayer_matrix, needed_size[:2])[:warped_grad.shape[0], :warped_grad.shape[1]]
                mod_warped_grad = warped_grad[:, :, 0]
                mod_warped_grad *= steps
                mod_warped_grad %= 1.0
                mod_warped_grad = (mod_warped_grad > image_matrix).astype(int)
                warped_grad = (np.floor(warped_grad[:, :, 1] * steps) + mod_warped_grad) / steps
                warped_grad = np.array((warped_grad.T, warped_grad.T, warped_grad.T, warped_grad.T)).T
            else:
                warped_grad = np.round(warped_grad * steps) / steps
        mult_grad = np.clip(((1 - warped_grad) * src + warped_grad * dst), 0, 255)
        if not raw:
            mult_grad[:, :, :3] = cv2.cvtColor(mult_grad[:, :, :3].astype(np.uint8), cv2.COLOR_Luv2RGB).astype(
                np.float64)
        mult_grad /= 255
        return (sprite * mult_grad).astype(np.uint8)

    @add_variant("overlay/", "o!", no_function_name=True)
    async def overlay(sprite, overlay: str, x: Optional[int] = 0, y: Optional[int] = 0, *, tile, wobble, renderer):
        """Applies an overlay to a sprite. X and Y can be given to offset the overlay."""
        tile.custom_color = True
        assert overlay in renderer.overlay_cache, f"`{overlay}` isn't a valid overlay!"
        overlay_image = renderer.overlay_cache[overlay]
        tile_amount = np.ceil(np.array(sprite.shape[:2]) / overlay_image.shape[:2]).astype(
            int)  # Convert sprite.shape to ndarray to allow vectorized math
        overlay_image = np.roll(overlay_image, (x, y), (0, 1))
        overlay_image = np.tile(overlay_image, (*tile_amount, 1))[:sprite.shape[0], :sprite.shape[1]].astype(float)
        return np.multiply(sprite, overlay_image / 255, casting="unsafe").astype(np.uint8)

    # --- TEXT MANIPULATION ---

    @add_variant("noun", "prop")
    async def property(sprite,
                       plate: Optional[Literal["blank", "left", "up", "right", "down", "turn", "deturn", "soft"]] = None, *,
                       tile, wobble, renderer):
        """Applies a property plate to a sprite."""
        if plate is None:
            plate = tile.frame if tile.altered_frame else None
        else:
            plate = {v: k for k, v in constants.DIRECTIONS.items()}[plate]
        sprite = sprite[:, :, 3] > 0
        plate, _ = renderer.bot.db.plate(plate, wobble)
        plate = np.array(plate)[..., 3] > 0
        size = tuple(max(a, b) for a, b in zip(sprite.shape[:2], plate.shape))
        dummy = np.zeros(size, dtype=bool)
        delta = ((plate.shape[0] - sprite.shape[0]) // 2,
                 (plate.shape[1] - sprite.shape[1]) // 2)
        p_delta = max(-delta[0], 0), max(-delta[1], 0)
        delta = max(delta[0], 0), max(delta[1], 0)
        dummy[p_delta[0]:p_delta[0] + plate.shape[0],
        p_delta[1]:p_delta[1] + plate.shape[1]] = plate
        dummy[delta[0]:delta[0] + sprite.shape[0],
        delta[1]:delta[1] + sprite.shape[1]] &= ~sprite
        return np.dstack([dummy[..., np.newaxis].astype(np.uint8) * 255] * 4)

    @add_variant()
    async def custom(tile):
        """Forces custom generation of the text."""
        tile.custom = True
        tile.style = "noun"

    @add_variant("let")
    async def letter(tile):
        """Makes custom words appear as letter groups."""
        tile.style = "letter"

    @add_variant("1line", "1l")
    async def oneline(tile):
        """Makes custom words appear in one line."""
        tile.style = "oneline"


    # --- FILTERS ---

    @add_variant()
    async def hide(sprite):
        """Hides the tile."""
        sprite[..., 3] = 0
        return sprite

    @add_variant("rot")
    async def rotate(sprite, angle: float, expand: Optional[bool] = False):
        """Rotates a sprite."""
        if expand:
            scale = math.cos(math.radians(-angle % 90)) + math.sin(math.radians(-angle % 90))
            padding = int(sprite.shape[0] * ((scale - 1) / 2)), int(sprite.shape[1] * ((scale - 1) / 2))
            dst_size = sprite.shape[0] + padding[0], sprite.shape[1] + padding[1]
            check_size(*dst_size)
            sprite = np.pad(sprite,
                            (padding,
                             padding,
                             (0, 0)))
        image_center = tuple(np.array(sprite.shape[1::-1]) / 2 - 0.5)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        return cv2.warpAffine(sprite, rot_mat, sprite.shape[1::-1], flags=cv2.INTER_NEAREST)

    @add_variant("rot3d")
    async def rotate3d(sprite, phi: float, theta: float, gamma: float):
        """Rotates a sprite in 3D space."""
        phi, theta, gamma = math.radians(phi), math.radians(theta), math.radians(gamma)
        d = np.sqrt(sprite.shape[1] ** 2 + sprite.shape[0] ** 2)
        f = d / (2 * math.sin(gamma) if math.sin(gamma) != 0 else 1)
        w, h = sprite.shape[1::-1]
        proj_23 = np.array([[1, 0, -w / 2],
                            [0, 1, -h / 2],
                            [0, 0, 1],
                            [0, 0, 1]])
        rot_mat = np.dot(np.dot(
            np.array([[1, 0, 0, 0],
                      [0, math.cos(theta), -math.sin(theta), 0],
                      [0, math.sin(theta), math.cos(theta), 0],
                      [0, 0, 0, 1]]),
            np.array([[math.cos(phi), 0, -math.sin(phi), 0],
                      [0, 1, 0, 0],
                      [np.sin(phi), 0, math.cos(phi), 0],
                      [0, 0, 0, 1]])),
            np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                      [math.sin(gamma), math.cos(gamma), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]))
        trans_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, f],
            [0, 0, 0, 1]])
        proj_32 = np.array([
            [f, 0, w / 2, 0],
            [0, f, h / 2, 0],
            [0, 0, 1, 0]
        ])
        final_matrix = np.dot(proj_32, np.dot(trans_mat, np.dot(rot_mat, proj_23)))
        return cv2.warpPerspective(sprite, final_matrix, sprite.shape[1::-1], flags=cv2.INTER_NEAREST)

    @add_variant()
    async def scale(sprite, w: float, h: Optional[float] = None, interpolation: Optional[Literal["nearest", "linear", "cubic", "area", "lanczos"]] = "nearest"):
        """Scales a sprite by the given multipliers."""
        if h is None:
            h = w
        dst_size = (int(w * sprite.shape[0]), int(h * sprite.shape[1]))
        if dst_size[0] <= 0 or dst_size[1] <= 0:
            raise AssertionError(
                f"Can't scale a tile to `{int(w * sprite.shape[0])}x{int(h * sprite.shape[1])}`, as it has a non-positive target area.")
        check_size(*dst_size)
        dim = sprite.shape[:2] * np.array((h, w))
        dim = dim.astype(int)
        return cv2.resize(sprite[:, ::-1], dim[::-1], interpolation={
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }[interpolation])[:, ::-1]

    @add_variant()
    async def pad(sprite, left: int, top: int, right: int, bottom: int):
        """Pads the sprite by the specified values."""
        check_size(sprite.shape[1] + max(left, 0) + max(right, 0), sprite.shape[0] + max(top, 0) + max(bottom, 0))
        return np.pad(sprite, ((top, bottom), (left, right), (0, 0)))

    @add_variant("px")
    async def pixelate(sprite, x: int, y: Optional[int] = None):
        """Pixelates the sprite."""
        if y is None:
            y = x
        return sprite[y - 1::y, x - 1::x].repeat(y, axis=0).repeat(x, axis=1)

    @add_variant()
    async def posterize(sprite, bands: int):
        """Posterizes the sprite."""
        return np.dstack([np.digitize(sprite[..., i], np.linspace(0, 255, bands)) * (255 / bands) for i in range(4)])

    @add_variant("m")
    async def meta(sprite, level: Optional[int] = 1, kernel: Optional[Literal["full", "edge"]] = "full", size: Optional[int] = 1):
        """Applies a meta filter to an image."""
        if level is None: level = 1
        if size is None: size = 1
        assert size > 0, f"The given meta size of {size} is too small!"
        assert size <= constants.MAX_META_SIZE, f"The given meta size of {size} is too large! Try something lower than `{constants.MAX_META_SIZE}`."
        assert abs(level) <= constants.MAX_META_DEPTH, f"Meta depth of {level} too large! Try something lower than `{constants.MAX_META_DEPTH}`."
        # Not padding at negative values is intentional
        padding = max(level*size, 0)
        orig = np.pad(sprite, ((padding, padding), (padding, padding), (0, 0)))
        check_size(*orig.shape[size::-1])
        base = orig[..., 3]
        if level < 0:
            base = 255 - base
        ksize = 2*size + 1
        ker = np.ones((ksize, ksize))
        if kernel == 'full':
            ker[size, size] = - ksize**2 + 1
        elif kernel == 'edge':
            ker[size, size] = - ksize**2 + 5
            ker[0,0] = 0
            ker[0,ksize-1] = 0
            ker[ksize-1,ksize-1] = 0
            ker[ksize-1,0] = 0
        for _ in range(abs(level)):
            base = cv2.filter2D(src=base, ddepth=-1, kernel=ker)
        base = np.dstack((base, base, base, base))
        mask = orig[..., 3] > 0
        if not (level % 2) and level > 0:
            base[mask, ...] = orig[mask, ...]
        else:
            base[mask ^ (level < 0), ...] = 0
        return base
    
    @add_variant(no_function_name=True)
    async def omni(sprite, type: Optional[Literal["pivot", "branching"]] = "branching", *, tile, wobble, renderer):
        """Gives the tile an overlay, like the omni text."""
        opvalue = [0xcb, 0xab, 0x8b][wobble]
        num = 3
        if type == "pivot":
            num = 1
        nsprite = await meta(sprite, num)
        sprite = await pad(sprite, num, num, num, num)
        for i in range(nsprite.shape[0]):
            for j in range(nsprite.shape[1]):
                if nsprite[i, j, 3] == 0:
                    try:
                        nsprite[i, j] = sprite[i, j]
                    except:
                        pass
                else:
                    nsprite[i, j, 3] = opvalue
        return nsprite

    @add_variant()
    async def land(sprite, direction: Optional[Literal["left", "top", "right", "bottom"]] = "bottom"):
        """Removes all space between the sprite and its bounding box on the specified side."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        left, right = np.where(cols)[0][[0, -1]]
        top, bottom = np.where(rows)[0][[0, -1]]
        displacement = {"left": left, "top": top, "right": right+1-sprite.shape[1], "bottom": bottom+1-sprite.shape[0]}[direction]
        index = {"left": 0, "top": 1, "right": 0, "bottom": 1}[direction]
        return await wrap(sprite, ((1 - index) * displacement), index * displacement)

    @add_variant()
    async def bbox(sprite):
        """Puts the sprite's bounding box behind it. Useful for debugging."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        try:
            left, right = np.where(cols)[0][[0, -1]]
            top, bottom = np.where(rows)[0][[0, -1]]
        except IndexError:
            return sprite
        out = np.zeros_like(sprite).astype(float)
        out[top:bottom,   left:right] = (0xFF, 0xFF, 0xFF, 0x80)
        out[top,          left:right] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[bottom,       left:right] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[top:bottom,   left      ] = (0xFF, 0xFF, 0xFF, 0xc0)
        out[top:bottom+1, right     ] = (0xFF, 0xFF, 0xFF, 0xc0)
        sprite = sprite.astype(float)
        mult = sprite[..., 3, np.newaxis] / 255
        sprite[..., :3] = (1 - mult) * out[..., :3] + mult * sprite[..., :3]
        sprite[...,  3] = (sprite[..., 3] + out[..., 3] * (1 - mult[..., 0]))
        return sprite.astype(np.uint8)

    @add_variant()
    async def warp(sprite, x1_y1: list[int, int], x2_y2: list[int, int], x3_y3: list[int, int], x4_y4: list[int, int]):
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
        check_size(*new_shape)
        final_arr = np.zeros((*new_shape, 4), dtype=np.uint8)
        for source, destination in zip(src, dst):  # Iterate through the four triangles
            clip = cv2.fillConvexPoly(np.zeros(new_shape, dtype=np.uint8), destination, 1).astype(bool)
            M = cv2.getAffineTransform(source.astype(np.float32), destination.astype(np.float32))
            warped_arr = cv2.warpAffine(sprite, M, new_shape[::-1], flags=cv2.INTER_NEAREST)
            final_arr[clip] = warped_arr[clip]
        return final_arr

    @add_variant("mm")
    async def matrix(sprite,
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
    async def neon(sprite, strength: Optional[float] = 0.714):
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
        sprite[:, :, 3] = np.multiply(sprite[:, :, 3], np.clip(final_mask, 0, 1), casting="unsafe")
        return sprite.astype(np.uint8)

    @add_variant()
    async def convolve(sprite, width: int, height: int, *cell: float):
        """Convolves the sprite with the given 2D convolution matrix. Information on these can be found at https://en.wikipedia.org/wiki/Kernel_(image_processing)"""
        assert width * height == len(cell), f"Can't fit {len(cell)} values into a matrix that's {width}x{height}!"
        kernel = np.array(cell).reshape((height, width))
        return cv2.filter2D(src=sprite, ddepth=-1, kernel=kernel)

    @add_variant()
    async def scan(sprite, axis: Literal["x", "y"], on: Optional[int] = 1, off: Optional[int] = 1,
                   offset: Optional[int] = 0):
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
        return np.dstack((sprite[:, :, :3], sprite[:, :, 3] * mask))

    @add_variant()
    async def flip(sprite, *axis: Literal["x", "y"]):
        """Flips the sprite along the specified axes."""
        for a in axis:
            if a == "x":
                sprite = sprite[:, ::-1, :]
            else:
                sprite = sprite[::-1, :, :]
        return sprite

    @add_variant()
    async def mirror(sprite, axis: Literal["x", "y"], half: Literal["back", "front"]):
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
    async def normalize(sprite):
        """Centers the sprite on its visual bounding box."""
        rows = np.any(sprite[:, :, 3], axis=1)
        cols = np.any(sprite[:, :, 3], axis=0)
        if not len(row_check := np.where(rows)[0]) or not len(col_check := np.where(cols)[0]):
            return sprite
        left, right = row_check[[0, -1]]
        top, bottom = col_check[[0, -1]]
        sprite_center = sprite.shape[0] // 2 - 1, sprite.shape[1] // 2 - 1
        center = int((top + bottom) // 2), int((left + right) // 2)
        displacement = np.array((sprite_center[0] - center[0], sprite_center[1] - center[1]))
        return np.roll(sprite, displacement, axis=(0, 1))

    @add_variant("disp")
    async def displace(post, x: int, y: int):
        """Displaces the tile by the specified coordinates."""
        post.displacement = [post.displacement[0] - x, post.displacement[1] - y]

    # Original code by Charlotte (CenTdemeern1)
    @add_variant("flood")
    async def floodfill(sprite, color: Color, inside: Optional[bool] = True, *, tile, wobble, renderer):
        """Floodfills either inside or outside a sprite with a given brightness value."""
        color = Color.parse(tile, renderer.palette_cache, color)
        sprite[sprite[:, :, 3] == 0] = 0  # Optimal
        sprite_alpha = sprite[:, :, 3]  # Stores the alpha channel separately
        sprite_alpha[sprite_alpha > 0] = -1  # Sets all nonzero numbers to a number that's neither 0 nor 255.
        # Pads the alpha channel by 1 on each side to allow flowing past
        # where the sprite touches the edge of the bounding box.
        sprite_alpha = np.pad(sprite_alpha, ((1, 1), (1, 1)))
        sprite_flooded = cv2.floodFill(
            image=sprite_alpha,
            mask=None,
            seedPoint=(0, 0),
            newVal=255
        )[1]
        mask = sprite_flooded != (inside * 255)
        sprite_flooded[mask] = ((not inside) * 255)
        mask = mask[1:-1, 1:-1]
        if inside:
            sprite_flooded = 255 - sprite_flooded
        # Crops the alpha channel back to the original size and positioning
        sprite[:, :, 3][mask] = sprite_flooded[1:-1, 1:-1][mask].astype(np.uint8)
        sprite[(sprite[:, :] == [0, 0, 0, 255]).all(2)] = color
        return sprite
    
    @add_variant("pf")
    async def pointfill(sprite, color: Color, x: int, y: int, *, tile, wobble, renderer):
        """Floodfills a sprite starting at a given point."""
        color = Color.parse(tile, renderer.palette_cache, color)
        assert x >= 0 and y >= 0 and y < sprite.shape[0] and x < sprite.shape[1], f"Target point `{x},{y}` must be inside the sprite!"
        target_color = sprite[y,x]
        sprite[sprite[:, :, 3] == 0] = 0  # Optimal
        sprite_alpha = sprite[:, :, :].copy()  # Stores the alpha channel separately
        not_color_mask = (sprite[:, :, 0] != target_color[0]) | (sprite[:, :, 1] != target_color[1]) | (sprite[:, :, 2] != target_color[2])
        color_mask = (sprite[:, :, 0] == target_color[0]) & (sprite[:, :, 1] == target_color[1]) & (sprite[:, :, 2] == target_color[2])
        sprite_alpha[not_color_mask] = 255
        sprite_alpha[color_mask] = 0 # and now to override it
        sprite_alpha = sprite_alpha[:, :, 3].copy() #???
        sprite_flooded = cv2.floodFill(
            image=sprite_alpha,
            mask=None,
            seedPoint=(x, y),
            newVal=100
        )[1]
        mask = sprite_flooded == 100
        sprite[mask] = color
        return sprite

    @add_variant("rm")
    async def remove(sprite, color: Color, invert: Optional[bool] = False, *, tile, wobble, renderer):
        """Removes a certain color from the sprite. If [36minvert[0m is on, then it removes all but that color."""
        color = Color.parse(tile, renderer.palette_cache, color)
        if invert:
            sprite[(sprite[:, :, 0] != color[0]) | (sprite[:, :, 1] != color[1]) | (sprite[:, :, 2] != color[2])] = 0
        else:
            sprite[(sprite[:, :, 0] == color[0]) & (sprite[:, :, 1] == color[1]) & (sprite[:, :, 2] == color[2])] = 0
        return sprite
    
    @add_variant("rp")
    async def replace(sprite, color1: Color, color2: Color, invert: Optional[bool] = False, *, tile, wobble, renderer):
        """Replaces a certain color with a different color. If [36minvert[0m is on, then it replaces all but that color."""
        color1 = Color.parse(tile, renderer.palette_cache, color1)
        color2 = Color.parse(tile, renderer.palette_cache, color2)
        if invert:
            sprite[(sprite[:, :, 0] != color1[0]) | (sprite[:, :, 1] != color1[1]) | (sprite[:, :, 2] != color1[2])] = color2
        else:
            sprite[(sprite[:, :, 0] == color1[0]) & (sprite[:, :, 1] == color1[1]) & (sprite[:, :, 2] == color1[2])] = color2
        return sprite

    @add_variant()
    async def clip(sprite, *, tile, wobble, renderer):
        """Crops the sprite to within its grid space."""
        width = sprite.shape[1]
        height = sprite.shape[0]
        left = (width - 24) // 2
        up = (height - 24) // 2
        right = (width + 24) // 2
        down = (height + 24) // 2
        return await crop(sprite, [left,up], [right,down], True)

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
    async def color_select(sprite, *index: int):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, list(index))

    @add_variant("cslice", "cs")
    async def color_slice(sprite, s: Slice):
        """Keeps only the slice of colors, indexed by their occurrence. This changes per-frame, not per-tile.
Slices are notated as [30m([36mstart[30m/[36mstop[30m/[36mstep[30m)[0m, with all values being omittable."""
        return slice_image(sprite, s.slice)

    @add_variant("cshift", "csh")
    async def color_shift(sprite, s: Slice):
        """Shifts the colors of a sprite around, by index of occurence.
Slices are notated as [30m([36mstart[30m/[36mstop[30m/[36mstep[30m)[0m, with all values being omittable."""
        unique_colors = liquify.get_colors(sprite)
        unique_colors = np.array(
            sorted(unique_colors, key=lambda color: liquify.count_instances_of_color(sprite, color), reverse=True))
        final_sprite = np.tile(sprite, (len(unique_colors), 1, 1, 1))
        mask = np.equal(final_sprite[:, :, :, :], unique_colors.reshape((-1, 1, 1, 4))).all(axis=3)
        out = np.zeros(sprite.shape)
        for i, color in enumerate(unique_colors[s.slice]):
            out += np.tile(mask[i].T, (4, 1, 1)).T * color
        return out.astype(np.uint8)

    @add_variant("abberate")  # misspelling alias because i misspell it all the time
    async def aberrate(sprite, x: Optional[int] = 1, y: Optional[int] = 0):
        """Abberates the colors of a sprite."""
        check_size(sprite.shape[0] + abs(x) * 2, sprite.shape[1] + abs(y) * 2)
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

    @add_variant("alpha", "op")
    async def opacity(sprite, amount: float):
        """Sets the opacity of the sprite, from 0 to 1."""
        sprite[:, :, 3] = np.multiply(sprite[:, :, 3], np.clip(amount, 0, 1), casting="unsafe")
        return sprite

    @add_variant("neg")
    async def negative(sprite, alpha: bool = False):
        """Inverts the sprite's RGB or RGBA values."""
        sl = slice(None, None if alpha else 3)
        sprite[..., sl] = 255 - sprite[..., sl]
        return sprite

    @add_variant()
    async def wrap(sprite, x: int, y: int):
        """Wraps the sprite around its image box."""
        return np.roll(sprite, (y, x), (0, 1))

    @add_variant()
    async def melt(sprite, side: Optional[Literal["left", "top", "right", "bottom"]] = "bottom"):
        """Removes transparent pixels from each row/column and shifts the remaining ones to the end."""
        is_vertical = side in ("top", "bottom")
        at_end = side in ("right", "bottom")
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        # NOTE: I couldn't find a way to do this without at least one Python loop :/
        for i in range(sprite.shape[0]):
            sprite_slice = sprite[i, sprite[i, :, 3] != 0]
            sprite[i] = np.pad(sprite_slice,
                               ((sprite[i].shape[0] - sprite_slice.shape[0], 0)[::2 * at_end - 1], (0, 0)))
        if is_vertical:
            sprite = np.swapaxes(sprite, 0, 1)
        return sprite

    @add_variant()
    async def bend(sprite, axis: Literal["x", "y"], amplitude: int, offset: float, frequency: float):
        """Displaces the sprite by a wave. Frequency is a percentage of the sprite's size along the axis."""
        if axis == "y":
            sprite = np.rot90(sprite)
        offset = ((np.sin(
            np.linspace(offset, np.pi * 2 * (frequency + offset), sprite.shape[0])) / 2) * amplitude).astype(
            int)
        # NOTE: np.roll can't be element wise :/
        sprite[:] = sprite[np.mod(np.arange(sprite.shape[0]) + offset, sprite.shape[1])]
        if axis == "y":
            sprite = np.rot90(sprite, -1)
        return sprite

    @add_variant()
    async def wave(sprite, axis: Literal["x", "y"], amplitude: int, offset: float, frequency: float):
        """Displaces the sprite per-slice by a wave. Frequency is a percentage of the sprite's size along the axis."""
        if axis == "y":
            sprite = np.rot90(sprite)
        offset = ((np.sin(
            np.linspace(offset, np.pi * 2 * (frequency + offset), sprite.shape[0])) / 2) * amplitude).astype(
            int)
        # NOTE: np.roll can't be element wise :/
        for row in range(sprite.shape[0]):
            sprite[row] = np.roll(sprite[row], offset[row], axis=0)
        if axis == "y":
            sprite = np.rot90(sprite, -1)
        return sprite

    @add_variant("hs")
    async def hueshift(sprite, angle: int):
        """Shifts the hue of the sprite. 0 to 360."""
        hsv = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + int(angle // 2), 180)
        sprite[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sprite

    @add_variant("gamma", "g")
    async def brightness(sprite, brightness: float):
        """Sets the brightness of the sprite."""
        sprite = sprite.astype(float)
        sprite[:, :, :3] *= brightness
        sprite = sprite.clip(-256.0, 255.0) % 256
        return sprite.astype(np.uint8)

    @add_variant("ps")
    async def palette_snap(sprite, *, tile, wobble, renderer):
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
        min_indexes = np.argmin(diff_matrix, 0).reshape(
            diff_matrix.shape[1:])
        result = np.full(sprite.shape, 0, dtype=np.uint8)
        for i, color in enumerate(palette_colors):
            result[:, :, :3][min_indexes == i] = color
        result[:, :, 3] = sprite[:, :, 3]
        return result

    @add_variant("sat", "grayscale", "gscale")
    async def saturation(sprite, saturation: Optional[float] = 0):
        """Saturates or desaturates a sprite."""
        gray_sprite = sprite.copy()
        gray_sprite[..., :3] = (sprite[..., 0] * 0.299 + sprite[..., 1] * 0.587 + sprite[..., 2] * 0.114)[..., np.newaxis]
        return composite(gray_sprite, sprite, saturation)

    @add_variant()
    async def blank(sprite):
        """Sets a sprite to pure white."""
        sprite[:, :, :3] = 255
        return sprite

    @add_variant("liquify", no_function_name=True)
    async def liquify_variant(sprite):
        """"Liquifies" the tile by melting every color except the main color and distributing the main color downwards."""
        return liquify.liquify(sprite)

    @add_variant()
    async def planet(sprite):
        """Turns the tile into a planet by melting every color except the main color and distributing the main color in a circle."""
        a = liquify.planet(sprite)
        return a

    @add_variant("nl")
    async def normalize_lightness(sprite):
        """Normalizes a sprite's HSL lightness, bringing the lightest value up to full brightness."""
        arr_hls = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_RGB2HLS).astype(
            np.float64)  # since WHEN was it HLS???? huh?????
        max_l = np.max(arr_hls[:, :, 1])
        arr_hls[:, :, 1] *= (255 / max_l)
        sprite[:, :, :3] = cv2.cvtColor(arr_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)  # my question still stands
        return sprite

    @add_variant("3oo", "skul", hidden=True)
    async def threeoo(sprite, scale: float):
        """Content-aware scales the sprite downwards."""
        assert 0, "Due to both extremely little use and extreme difficulty to program, 3oo isn't in the bot anymore. Sorry!"

    @add_variant()
    async def crop(sprite, x_y: list[int, int], u_v: list[int, int], change_bbox: Optional[bool] = False):
        """Crops the sprite to the specified bounding box.
    If the [36mchange_bbox[0m toggle is on, then the sprite's bounding box is altered, as opposed to removing pixels."""
        (x, y), (u, v) = x_y, u_v
        if change_bbox:
            return sprite[y:v, x:u]
        else:
            dummy = np.zeros_like(sprite)
            dummy[y:v, x:u] = sprite[y:v, x:u]
            return dummy

    @add_variant()
    async def snip(sprite, x_y: list[int, int], u_v: list[int, int]):
        """Snips the specified box out of the sprite."""
        (x, y), (u, v) = x_y, u_v
        sprite[y:v, x:u] = 0
        return sprite

    @add_variant()
    async def croppoly(sprite, *x_y: list[int]):
        """Crops the sprite to the specified polygon."""
        assert len(x_y) > 5, "Must have at least 3 points to define a polygon!"
        pts = np.array(x_y, dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[1::-1], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        return np.multiply(sprite, clip_poly, casting="unsafe").astype(np.uint8)

    @add_variant()
    async def snippoly(sprite, *x_y: list[int]):
        """Like croppoly, but also like snip. Snips the specified polygon out of the sprite."""
        assert len(x_y) > 5, "Must have at least 3 points to define a polygon!"
        pts = np.array(x_y, dtype=np.int32).reshape((1, -1, 2))[:, :, ::-1]
        clip_poly = cv2.fillPoly(np.zeros(sprite.shape[1::-1], dtype=np.float32), pts, 1)
        clip_poly = np.tile(clip_poly, (4, 1, 1)).T
        return np.multiply(sprite, 1 - clip_poly, casting="unsafe").astype(np.uint8)

    @add_variant("cvt")
    async def convert(sprite, direction: Literal["to", "from"],
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
    async def threshold(sprite, r: float, g: Optional[float] = None, b: Optional[float] = None,
                        a: Optional[float] = 0.0):
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
    async def blur(sprite, radius: int, gaussian: Optional[bool] = False):
        """Blurs a sprite. Uses box blur by default, though gaussian blur can be used with the boolean toggle."""
        check_size(sprite.shape[0] + radius * 2, sprite.shape[1] + radius * 2)
        arr = np.pad(sprite, ((radius, radius), (radius, radius), (0, 0)))
        assert radius > 0, f"Blur radius of {radius} is too small!"
        if gaussian:
            arr = cv2.GaussianBlur(arr, (radius * 2 + 1, radius * 2 + 1), 0)
        else:
            arr = cv2.boxFilter(arr, -1, (radius * 2 + 1, radius * 2 + 1))
        return arr

    @add_variant("fish")
    async def fisheye(sprite, strength: float):
        """Applies a fisheye effect."""
        size = np.array(sprite.shape[:2])
        filt = np.indices(sprite.shape[:2], dtype=np.float32) / size[:, np.newaxis, np.newaxis]
        filt = (2 * filt) - 1
        abs_filt = np.linalg.norm(filt, axis=0)
        filt /= (1 - (strength / 2) * (abs_filt[np.newaxis, ...]))
        filt += 1
        filt /= 2
        filt = filt * np.array(sprite.shape)[:2, np.newaxis, np.newaxis]
        filt = np.float32(filt)
        mapped = cv2.remap(sprite, filt[1], filt[0],
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0).astype(float)
        return np.uint8(mapped)

    @add_variant("filter", "fi!")
    async def filterimage(sprite, filter_url: str, absolute: Optional[bool] = None, *, tile, wobble, renderer):
        """Applies a filter image to a sprite. For information about filter images, look at the filterimage command."""
        frames, abs_db = await renderer.bot.db.get_filter(filter_url)
        try:
            filter = frames[wobble]
        except IndexError:
            filter = frames[0]
        filt = np.array(filter.convert("RGBA"))
        check_size(*filt.shape[:2])
        absolute = absolute if absolute is not None else \
            abs_db if abs_db is not None else False
        filt = np.float32(filt)
        filt[..., :2] -= 0x80
        if not absolute:
            filt[..., :2] += np.indices(filt.shape[:2]).T
        mapped = cv2.remap(sprite, filt[..., 0], filt[..., 1],
                           interpolation=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_WRAP).astype(float)
        filt /= 255
        mapped[..., :3] *= filt[..., 2, np.newaxis]
        mapped[..., 3] *= filt[..., 3]
        return np.uint8(mapped)

    @add_variant()
    async def glitch(sprite, distance: int, chance: Optional[float] = 1.0, seed: Optional[int] = None, *, tile, wobble,
                     renderer):
        """Randomly displaces a sprite's pixels. An RNG seed is created using the tile's attributes if not specified."""
        if seed is None:
            seed = abs(hash(tile))
        dst = np.indices(sprite.shape[:2], dtype=np.float32)
        rng = np.random.default_rng(seed * 3 + wobble)
        displacement = rng.uniform(-distance, distance, dst.shape)
        mask = rng.uniform(0, 1, dst.shape)
        displacement[mask > chance] = 0
        dst += displacement
        return cv2.remap(sprite, dst[1], dst[0],
                         interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_WRAP)

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
