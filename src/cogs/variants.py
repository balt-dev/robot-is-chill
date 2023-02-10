import glob
import inspect
import re
import types
import typing
from pathlib import Path

from PIL import ImageChops, Image, ImageOps, ImageFilter

from .. import constants
from ..types import Variant, RegexDict

"""
TODO:
- def f(*args: type)
- Directions
- States
- Tiling
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
            val = type(curr_type)(parse_signature(v[:num_values], curr_type))
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
                pattern += f"/(?:{generate_pattern(typing.get_args(t))})?"
            elif type(t) is not type:
                pattern += f"/({'|'.join([str(arg) for arg in t])})"
            elif isinstance(t, typing.Iterable):
                pattern += rf"/\({generate_pattern(t)}\)"
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
        return pattern[1:]  # Remove starting /

    def generate_syntax(params):
        syntax = ""
        for name, param in dict(params).items():
            if param.annotation == inspect.Parameter.empty:
                continue
            elif isinstance(param.annotation, typing.Iterable):
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
        sig = inspect.signature(func)
        params = sig.parameters
        type_tree = get_type_tree(p.annotation for p in params.values() if p.kind != inspect.Parameter.KEYWORD_ONLY)[1:]
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
                    "apply": (lambda self, obj, **kwargs: func(obj, *self.args, **kwargs)),
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

    @add_variant("noop", "")
    def nothing(tile):
        """Does nothing. Useful for resetting persistent variants."""
        pass

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
    def palette_color(tile, palette: str):
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
        dummy = Image.new("1", (plate.width, plate.height))
        delta = (plate.size[0] - sprite.size[0]) // 2, \
            (plate.size[0] - sprite.size[1]) // 2
        dummy.paste(
            Image.new("1", (sprite.width, sprite.height), 1),
            delta,
            sprite.getchannel("A").convert("1")
        )
        band = ImageChops.logical_xor(dummy, plate.convert("1")).convert("L")
        return Image.merge("RGBA", (band, band, band, band))

    @add_variant()
    def custom(tile):
        """Forces custom generation of the text."""
        tile.custom = True
        tile.style = "noun"

    @add_variant()
    def letter(tile):
        """Makes 1 or 2 letter custom words appear as letter groups."""
        tile.style = "letter"

    @add_variant()
    def hide(tile):
        """Hides the tile."""
        tile.empty = True

    # --- FILTERS ---

    @add_variant("m")
    def meta(sprite, level: typing.Optional[int] = 1, *, tile, wobble: int, renderer) -> Image.Image:
        """Applies a meta filter to an image."""
        assert abs(level) <= constants.MAX_META_DEPTH, f"Meta depth of {level} too large!"
        if tile.style == "property":
            prop_text = prop(sprite, tile=tile, wobble=wobble, renderer=renderer)
            orig, _ = renderer.bot.db.plate(tile.frame if tile.altered_frame else None, wobble)
            orig = orig.copy()
            base = orig.getchannel("A")
        else:
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
        if tile.style == "property":
            dummy = Image.new("1", (base.width, base.height))
            dummy.paste(
                Image.new("1", (sprite.width, sprite.height), 1),
                (level, level),
                prop_text.getchannel("A").convert("1")
            )
            band = ImageChops.logical_xor(dummy, base.convert("1")).convert("L")
            base = Image.merge("RGBA", (band, band, band, band))
        return base

    # --- ADD TO BOT ---

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])