import inspect
import re
import types
import typing

from PIL import ImageChops

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
        if isinstance(curr_type, list):  # tree branch
            num_values = len(curr_type)
            val = type(curr_type)(parse_signature(v[:num_values], curr_type))
            del v[:num_values]
        elif isinstance(curr_type, tuple):  # literal
            assert v[0] in curr_type, "Supplied value not in allowed values for variant!"
            val = v[0]
        elif curr_type == bool:
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
            if type(t) is not type:
                pattern += f"/({'|'.join([str(arg) for arg in t])})"
            elif typing.get_origin(t) == typing.Union:
                pattern += f"(?:/{generate_pattern(typing.get_args(t))})?"
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
        type_tree = get_type_tree(p.annotation for p in params.values())[1:]
        pattern = rf"(?:{'|'.join(aliases)}{'' if no_function_name else f'|{func.__name__}'}){generate_pattern(type_tree)}"
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
                    "__repr__": (lambda self: f"{self.__class__.__name__}({self.args})"),
                    "apply": (lambda self, obj: func(obj, *self.args)),
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

    # --- ANIMATION FRAMES ---
    @add_variant(no_function_name=True)
    def frame(tile, anim_frame: int):
        """Sets the animation frame of a sprite."""
        tile.frame = anim_frame
        tile.surrounding = 0

    @add_variant(no_function_name=True)
    def direction(tile, d: typing.Literal[*tuple(constants.DIRECTION_VARIANTS.keys())]):
        tile.frame = constants.DIRECTION_VARIANTS[d]

    @add_variant(no_function_name=True)
    def tiling(tile, d: typing.Literal[*tuple(constants.AUTO_VARIANTS.keys())]):
        tile.surrounding |= constants.AUTO_VARIANTS[d]

    @add_variant("a", no_function_name=True)
    def animation_frame(tile, a_frame: int):
        tile.frame += a_frame

    @add_variant("s", "sleep", no_function_name=True)
    def sleep(tile):
        tile.frame = (tile.frame - 1) % 32

    # --- COLORING ---

    @add_variant(no_function_name=True)
    def palette_color(tile, x: int, y: int):
        tile.color = (x, y)

    @add_variant("#", no_function_name=True)
    def hex_color(tile, color: str):
        assert re.match(r'(?:[0-9A-Fa-f]{3}){1,2}', color), f"Invalid color for tile {tile.name}!"
        r, g, b = [int(n, base=16) for n in (list(color) if len(color) == 3 else re.findall("..", color))]
        if len(color) == 3:
            r, g, b = (r << 4) + r, (g << 4) + g, (b << 4) + b
        tile.color = (r, g, b)

    @add_variant(no_function_name=True)
    def named_color(tile, color: typing.Literal[*constants.COLOR_NAMES.keys()]):
        tile.color = constants.COLOR_NAMES[color]

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
