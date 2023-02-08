import inspect
import types
import typing

from PIL import ImageChops

from ..types import Variant, RegexDict


"""
TODO:
- def f(*args: type)
- def f(a: Literal["foo","bar"])
- Directions
- States
- Tiling
"""

def parse_signature(v: list[str], t: list[type | types.GenericAlias]) -> list[typing.Any]:
    out = []
    t = list(t).copy()
    v = list(v).copy()
    while len(t) > 0:
        curr_type = t.pop(0)
        if isinstance(curr_type, typing.Iterable):
            num_values = len(curr_type)
            val = type(curr_type)(parse_signature(v[:num_values], curr_type))
            del v[:num_values]
        elif curr_type == bool:
            g = v.pop(0)
            val = g == "true"
        else:
            val = curr_type(v.pop(0))
        out.append(val)
    return out


def generate_pattern(sign):
    pattern = f""
    for t in sign:
        if isinstance(t, typing.Iterable):
            pattern += f"\({generate_pattern(t)}\)"
        elif t == int:
            pattern += r"(-?\d+)"
        elif t == float:
            pattern += r"([+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+))"  # From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers/42629198#42629198
        elif t == str:
            pattern += r"(.+?)"
        elif t == bool:
            pattern += r"(true|false)"
        else:
            continue
        pattern += "/"
    return pattern[:-1]  # Remove ending /


def generate_syntax(params):
    syntax = ""
    for name, param in dict(params).items():
        if param.annotation == inspect.Parameter.empty:
            continue
        elif isinstance(param.annotation, typing.Iterable):
            syntax += f"""({generate_syntax({
                name: inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno)
                for name, anno in zip(name.split('_'), typing.get_args(param.annotation))})})"""
        else:
            syntax += f"<{param.annotation.__name__} {name}>"
        syntax += "/"
    return syntax[:-1]  # Remove ending /


def get_type_tree(types):
    return [get_type_tree([ti for ti in typing.get_args(t)]) if isinstance(t, typing.GenericAlias) else t for t in
            types]


def class_init(self, *args):
    self.args = args


async def setup(bot):
    """Get the variants."""
    bot.variants = []

    def create_variant(func: typing.Callable, aliases: typing.Iterable[str]):
        sig = inspect.signature(func)
        params = sig.parameters
        types = get_type_tree(p.annotation for p in params.values())[1:]
        pattern = rf"(?:{'|'.join(aliases)}|{func.__name__}){generate_pattern(types)}"
        syntax = generate_syntax(params)
        class_name = func.__name__.title() + "Variant"

        def apply(self, obj):
            print(self.args)
            return func(obj, *self.args)

        def class_init(self, *args):
            self.args = args

        def class_repr(self):
            return f"{self.__class__.__name__}({self.args})"

        variant_type = tuple(params.keys())[0]
        bot.variants.append(
            type(
                class_name,
                (Variant,),
                {
                    "__init__": class_init,
                    "__doc__": func.__doc__,
                    "__repr__": class_repr,
                    "apply": apply,
                    "pattern": pattern,
                    "signature": types,
                    "syntax": syntax,
                    "type": variant_type,
                }
            )
        )


    def add_variant(*aliases):
        def wrapper(func):
            create_variant(func, aliases)
            return func

        return wrapper

    @add_variant("inv", "i")
    def invert(sprite):
        """Inverts the colors of a sprite."""
        return ImageChops.invert(sprite)

    @add_variant("disp")
    def displace(skeleton, x: int, y: int):
        """Displaces a tile by the given coordinates."""
        skeleton.displacement = (skeleton.displacement[0] + x, skeleton.displacement[1] + y)

    bot.variants = RegexDict([(variant.pattern, variant) for variant in bot.variants])
