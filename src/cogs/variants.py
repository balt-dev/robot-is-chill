import glob
import inspect
import re
import sys
import types
import typing
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageChops, Image, ImageOps, ImageFilter

from . import liquify
from .. import constants
from ..types import Variant, RegexDict, VaryingArgs

"""
TODO:
- def f(*args: type)
"""


def class_init(self, *args):
    self.args = args


def parse_signature(v: list[str], t: list[type | types.GenericAlias]) -> list[typing.Any]:
    out = []
    t = list(t).copy()
    v = list(v).copy()
    if v is None or (len(v) and v[0] is None):
        val = None
    while len(t) > 0:
        curr_type = t.pop(0)
        if typing.get_origin(curr_type) is typing.Union:
            curr_type = typing.get_args(curr_type)[0]
            if v[0] is None:
                continue
        if type(curr_type) is VaryingArgs:
            print(v)
            out.extend(parse_signature(v, [curr_type.type] * len(v)))
            print(out)
            return out
        elif isinstance(curr_type, list):  # tree branch
            num_values = len(curr_type)
            val = tuple(parse_signature(v[:num_values], curr_type))
            del v[:num_values]
        elif isinstance(curr_type, tuple):  # literal
            assert v[0] in curr_type, "Supplied value not in allowed values for variant!"
            val = v[0]
        elif isinstance(curr_type, bool):
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

    def generate_pattern(params: list[inspect.Parameter]):
        pattern = f""
        for i, p in enumerate(params):
            if typing.get_origin(p.annotation) == typing.Literal and len(typing.get_args(p.annotation)) > 2:
                pattern += f"/({'|'.join([str(arg) for arg in typing.get_args(p.annotation)])})"
            elif typing.get_origin(p.annotation) == list:
                pattern += rf"/\({generate_pattern([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno) for name, anno in zip(p.name.split('_'), typing.get_args(p.annotation))])}\)"
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                # You can't match a variable number of groups in RegEx.
                pattern += f"(?:{'/' if i - 1 else ''}{(generate_pattern([inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=p.annotation)]).replace('/', '', 1) + '/?')})?" * constants.VAR_POSITIONAL_MAX
            elif typing.get_origin(p.annotation) == typing.Union:
                pattern += f"(?:{'/' if i - 1 else ''}{generate_pattern([inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typing.get_args(p.annotation)[0])])})?"
            elif p.annotation == int:
                pattern += r"/(-?\d+)"
            elif p.annotation == float:
                pattern += r"/([+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+))"  # From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers/42629198#42629198
            elif p.annotation == str:
                pattern += r"/(.+?)"
            elif p.annotation == bool:
                pattern += r"/(true|false)"
            else:
                continue
        return pattern.lstrip("/")  # Remove starting /

    def generate_syntax(params):
        syntax = ""
        for name, param in dict(params).items():
            if param.annotation == inspect.Parameter.empty:
                continue
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                syntax += f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m/[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m/[0;30m..."
                break
            elif typing.get_origin(param.annotation) is typing.Union:
                syntax += f"[0;30m[[1;34m{typing.get_args(param.annotation)[0].__name__} [0;34m{name}[0;30m: [1;37m{repr(param.default)}[0;30m][0m"
            elif typing.get_origin(param.annotation) == list:
                syntax += f"""[0;34m({generate_syntax({
                    name: inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno)
                    for name, anno in zip(name.split('_'), typing.get_args(param.annotation))})}[0;34m)[0m"""
            elif typing.get_origin(param.annotation) == typing.Literal:
                syntax += f"""[0;30m<[32m{'[0;30m/[32m'.join([repr(arg) for arg in typing.get_args(param.annotation)])} [0;36m{name}[0;30m>[0m"""
            else:
                syntax += f"[0;30m<[1;36m{param.annotation.__name__} [0;36m{name}[0;30m>[0m"
            syntax += "/"
        return syntax.rstrip("[0m").rstrip("/")  # Remove ending /

    def get_type_tree(unparsed_tree):
        tree = []
        for i, p in enumerate(unparsed_tree):
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                tree.append(VaryingArgs(p.annotation))
                return tree
            elif typing.get_origin(p.annotation) == typing.Literal:
                tree.append(tuple(typing.get_args(p.annotation)))
            elif isinstance(p.annotation, typing.GenericAlias):
                tree.append(get_type_tree(
                    inspect.Parameter(p.name, p.kind, annotation=anno) for anno in typing.get_args(p.annotation)))
            else:
                tree.append(p.annotation)
        return tree

    def create_variant(func: typing.Callable, aliases: typing.Iterable[str], no_function_name=False):
        assert func.__doc__ is not None, f"Variant `{func.__name__}` is missing a docstring!"
        sig = inspect.signature(func)
        params = sig.parameters
        has_kwargs = any([p.kind == inspect.Parameter.KEYWORD_ONLY for p in params.values()])
        if not no_function_name:  # HACKY
            aliases = list(aliases)
            aliases.append(func.__name__)
            aliases = tuple(aliases)
        pattern = rf"(?:{'|'.join(aliases)}){generate_pattern(list(params.values()))}"
        print(pattern)
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
                    "apply": (lambda self, obj, **kwargs: func(obj, *self.args, **kwargs if has_kwargs else {})),
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
    def blending(tile, mode: typing.Literal["normal", "add", "subtract", "multiply", "cut", "mask", "xor"]):
        """Sets the blending mode for the tile."""
        tile.blending = {"normal": "NORMAL",
                         "add": "ADD",
                         "subtract": "SUB",
                         "multiply": "MUL",
                         "max": "MAX",
                         "min": "MIN",
                         "cut": "CUT",
                         "mask": "MASK",
                         "xor": "XOR"}[mode]

    # --- ANIMATION FRAMES ---

    @add_variant(no_function_name=True)
    def frame(tile, anim_frame: int):
        """Sets the animation frame of a sprite."""
        tile.altered_frame = True
        tile.frame = anim_frame
        tile.surrounding = 0

    @add_variant(no_function_name=True)
    def direction(tile, direction: typing.Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]):
        """Sets the direction of a tile."""
        tile.altered_frame = True
        tile.frame = constants.DIRECTION_VARIANTS[direction]

    @add_variant(no_function_name=True)
    def tiling(tile, tiling: typing.Literal[*tuple(constants.AUTO_VARIANTS.keys())]):
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
    def palette_color(tile, x: int, y: int):
        """Sets a color by palette index."""
        tile.color = (x, y)

    @add_variant("#", no_function_name=True)
    def hex_color(tile, color: str):
        """Sets a color by hexadecimal value. Can take a 3-long or 6-long hexadecimal color."""
        assert re.fullmatch(r'(?:[0-9A-Fa-f]{3}){1,2}', color) is not None, f"Invalid color for tile `{tile.name}`!"
        r, g, b = [int(n, base=16) for n in (list(color) if len(color) == 3 else re.findall("..", color))]
        if len(color) == 3:
            r, g, b = (r << 4) + r, (g << 4) + g, (b << 4) + b
        tile.color = (r, g, b)

    @add_variant(no_function_name=True)
    def named_color(tile, color: typing.Literal[*constants.COLOR_NAMES.keys()]):
        """Sets a color by name."""
        tile.color = constants.COLOR_NAMES[color]

    # --- TEXT MANIPULATION ---

    @add_variant("noun", "prop")
    def property(sprite, *, tile, wobble, renderer):
        """Applies a property plate to a sprite."""
        plate, box = renderer.bot.db.plate(tile.frame if tile.altered_frame else None, wobble)
        size = (max(sprite.width, plate.width), max(sprite.height, plate.height))
        dummy = Image.new("1", size)
        delta = (plate.size[0] - sprite.size[0]) // 2, \
                (plate.size[0] - sprite.size[1]) // 2
        plate_dummy = Image.new("1", size)
        plate_dummy.paste(
            plate.convert("1"),
            (max((sprite.width - plate.width) // 2, 0), max((sprite.height - plate.height) // 2, 0))
        )
        dummy.paste(
            Image.new("1", (sprite.width, sprite.height), 1),
            delta,
            sprite.getchannel("A").convert("1")
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

    @add_variant("m")
    def meta(sprite, level: typing.Optional[int] = 1) -> Image.Image:
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

    @add_variant("mm")
    def matrix(post,
               aa_ab_ac_ad: list[float, float, float, float],
               ba_bb_bc_bd: list[float, float, float, float],
               ca_cb_cc_cd: list[float, float, float, float],
               da_db_dc_dd: list[float, float, float, float]):
        """Multiplies the sprite by the given RGBA matrix."""
        matrix = np.array((aa_ab_ac_ad, ba_bb_bc_bd, ca_cb_cc_cd, da_db_dc_dd))
        img = np.array(post.convert('RGBA'), dtype=np.float64) / 255.0
        immul = img.reshape(-1, 4) @ matrix  # @ <== matmul
        immul = (np.clip(immul, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(immul.reshape(img.shape))

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

    # Original code by Charlotte (CenTdemeern1)
    @add_variant("flood")
    def floodfill(sprite, brightness: typing.Optional[float] = 1.0, inside: typing.Optional[bool] = False):
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
        im_flooded[im_flooded != (0 if inside else 255)] = (255 if inside else 0)
        if not inside:
            im_flooded = 255 - im_flooded
        # Crops the alpha channel back to the original size and positioning
        im[:, :, 3] = im_flooded[1:-1, 1:-1].astype(np.uint8)
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

        # This is indented because we don't need to convert back if
        # nothing changed
        return Image.fromarray(img)

    @add_variant("csel")
    def color_select(sprite, *index: int):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, list(index))

    @add_variant("cslice")
    def color_slice(sprite, start: typing.Optional[int] = None, stop: typing.Optional[int] = None,
                    step: typing.Optional[int] = None):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        return slice_image(sprite, slice(start, stop, step))

    @add_variant("abberate")  # misspelling alias because i misspell it all the time
    def aberrate(post, x: int, y: int):
        """Abberates the colors of a sprite."""
        arr = np.array(post)
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
        new_sprite = sprite.copy()
        im_alpha = new_sprite.getchannel("A")
        new_sprite.putalpha(im_alpha.point(lambda i: int(i * amount)))
        return new_sprite

    @add_variant("neg")
    def negative(post):
        """Inverts the sprite's RGB values."""
        arr = np.array(post)
        arr[:, :, :3] = ~arr[:, :, :3]
        return Image.fromarray(arr)

    @add_variant("hs")
    def hueshift(post, angle: int):
        """Shifts the hue of the sprite. 0 to 360."""
        arr = np.array(post)
        arr_rgb, arr_a = arr[:, :, :3], arr[:, :, 3]
        hsv = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + int(angle // 2), 180)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(np.dstack((rgb, arr_a)))

    @add_variant("gamma", "g")
    def brightness(post, brightness: float):
        """Sets the brightness of the sprite. Can go above 1.0, does nothing below 0.0."""
        arr = np.array(post, dtype=np.float64)
        arr[:, :, :3] *= brightness
        arr = arr.clip(0.0, 255.0)
        return Image.fromarray(arr.astype(np.uint8))

    @add_variant("ps")
    def palette_snap(post, *, tile, palette_cache):
        """Snaps all the colors in the tile to the specified palette."""
        palette_colors = np.array(palette_cache[tile.palette].convert("RGB")).reshape(-1, 3)
        im = np.array(post)
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
    def saturation(post, saturation: float):
        """Saturates or desaturates a tile."""
        return Image.blend(post.convert("RGBA"), post.convert("LA").convert("RGBA"), 1.0 - saturation)

    @add_variant("nl")
    def normalize_lightness(post):
        """Normalizes a sprite's HSL lightness, bringing the lightest value up to full brightness."""
        arr = np.array(post)
        arr_rgb, sprite_a = arr[:, :, :3], arr[:, :, 3]
        arr_hls = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HLS).astype(np.float64)  # since WHEN was it HLS???? huh?????
        max_l = np.max(arr_hls[:, :, 1])
        arr_hls[:, :, 1] *= (255 / max_l)
        sprite_rgb = cv2.cvtColor(arr_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)  # my question still stands
        return Image.fromarray(np.dstack((sprite_rgb, sprite_a)))

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
