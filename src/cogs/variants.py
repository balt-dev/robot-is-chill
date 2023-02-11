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
from ..types import Variant, RegexDict

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
    while len(t) > 0:
        curr_type = t.pop(0)
        if typing.get_origin(curr_type) is typing.Union:
            curr_type = typing.get_args(curr_type)[0]
            if v[0] is None:
                continue
        if isinstance(curr_type, list):  # tree branch
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
            val = curr_type(v.pop(0))
        out.append(val)
    return out


async def setup(bot):
    """Get the variants."""
    bot.variants = []

    def generate_pattern(sign):
        pattern = f""
        for t in sign:
            if typing.get_origin(t) == typing.Union:
                pattern += f"(?:/{generate_pattern(typing.get_args(t))})?"
            elif isinstance(t, list) and len(t) > 2:
                pattern += rf"/\({generate_pattern(t)}\)"
            elif type(t) is not type:
                pattern += f"/({'|'.join([str(arg) for arg in t])})"
            elif t == int:
                pattern += r"/(-?\d+)"
            elif t == float:
                pattern += r"/([+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+))"  # From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers/42629198#42629198
            elif t == str:
                pattern += r"/(.+?)"
            elif t == bool:
                pattern += r"/(true|false)"
            else:
                continue
        return pattern.lstrip("/")  # Remove starting /

    def generate_syntax(params):
        syntax = ""
        for name, param in dict(params).items():
            if param.annotation == inspect.Parameter.empty:
                continue
            elif isinstance(param.annotation, list):
                syntax += f"""({generate_syntax({
                    name: inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno)
                    for name, anno in zip(name.split('_'), typing.get_args(param.annotation))})})"""
            elif typing.get_origin(param.annotation) is typing.Literal:
                syntax += f"""<{'/'.join([repr(arg) for arg in typing.get_args(param.annotation)])} {name}>"""
            else:
                syntax += f"<{param.annotation.__name__} {name}>"
            syntax += "/"
        return syntax[:-1]  # Remove ending /

    def get_type_tree(unparsed_tree):
        tree = []
        for t in unparsed_tree:
            if isinstance(t, typing.GenericAlias):
                tree.append(get_type_tree(typing.get_args(t)))
            elif typing.get_origin(t) is typing.Literal:
                tree.append(tuple(typing.get_args(t)))
            else:
                tree.append(t)
        return tree

    def create_variant(func: typing.Callable, aliases: typing.Iterable[str], no_function_name=False):
        assert func.__doc__ is not None, f"Variant `{func.__name__}` is missing a docstring!"
        sig = inspect.signature(func)
        params = sig.parameters
        type_tree = get_type_tree(p.annotation for p in params.values() if p.kind != inspect.Parameter.KEYWORD_ONLY)[1:]
        has_kwargs = any([p.kind == inspect.Parameter.KEYWORD_ONLY for p in params.values()])
        pattern = rf"(?:{'|'.join(aliases)}{'' if no_function_name else f'|{func.__name__}'}){generate_pattern(type_tree)}"
        print(pattern)
        syntax = generate_syntax(params)
        class_name = func.__name__.replace("_", " ").title().replace(" ", "") + "Variant"
        variant_type = tuple(params.keys())[0]
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

    @add_variant("noop", "")
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
    def direction(tile, d: typing.Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]):
        """Sets the direction of a tile."""
        tile.altered_frame = True
        tile.frame = constants.DIRECTION_VARIANTS[d]

    @add_variant(no_function_name=True)
    def tiling(tile, d: typing.Literal[*tuple(constants.AUTO_VARIANTS.keys())]):
        """Alters the tiling of a tile. Only works on tiles that tile."""
        tile.altered_frame = True
        tile.surrounding |= constants.AUTO_VARIANTS[d]

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
    def set_palette(tile, palette: str):
        """Sets the tile's palette."""
        assert palette in palette_names, f"Palette `{palette}` was not found!"
        tile.palette = palette

    @add_variant(no_function_name=True)
    def palette_color(post, x: int, y: int):
        """Sets a color by palette index."""
        post.color = (x, y)

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

    @add_variant("noun")
    def prop(sprite: Image, *, tile, wobble: int, renderer):
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
    def matmul(post,
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

    # Original code by Charlotte
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

    @add_variant("csel")
    # Original code by Charlotte
    def colselect(sprite, *indices: int):
        """Keeps only the selected colors, indexed by their occurrence. This changes per-frame, not per-tile."""
        sprite = sprite.convert("RGBA")

        # Get a dictionary of colors mapped to their frequencies,
        # and said dictionary's keys sorted by its values to create a histogram
        color_dict = dict(map(reversed, sprite.getcolors(maxcolors=0xFFFFFF)))
        colors = np.array(sorted(color_dict, key=color_dict.get, reverse=True))

        indices = [index % len(colors) for index in indices]
        colors_to_delete = np.delete(colors[colors[:, 3] != 0], indices, axis=0)

        # NOTE: This could be optimized with np.isin or similar.
        # However, it's 1 AM as I am typing this, and I really do not care.
        processable_sprite = np.array(sprite)
        for color in colors_to_delete:
            liquify.remove_instances_of_color(processable_sprite, color)
        return Image.fromarray(processable_sprite)

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
