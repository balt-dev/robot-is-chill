from __future__ import annotations

from dataclasses import dataclass, field

from typing import Literal, Optional
import re
import numpy as np

from . import errors, constants
from .cogs.variants import parse_signature
from .db import TileData
from .types import Variant, Context, RegexDict

#vvv used for the <> below vvv
def mult(a,b):
    try:
        return a*b
    except:
        return "don't."
def add(a,b):
    try:
        return a+b
    except:
        return "don't."
def sub(a,b):
    try:
        return a-b
    except:
        return "don't."
def div(a,b):
    try:
        return a/b
    except:
        return "don't."
def exp(a,b):
    try:
        return a**b
    except:
        return "don't."

# thanks to https://stackoverflow.com/questions/13055884/parsing-math-expression-in-python-and-solving-to-find-an-answer for this
def parse_expression(x):
    try:
        operators = set('+-*/^')
        op_out = [] 
        num_out = []
        buff = []
        for c in x:
            if c in operators:  
                num_out.append(''.join(buff))
                buff = []
                op_out.append(c)
            else:
                buff.append(c)
        num_out.append(''.join(buff))
        operator_order = ('^','*/','+-')
        
        num_out2 = []
        for num in num_out:
            num2 = num
            if len(num) > 0:
                if num[0] == "!":
                    num2 = "-" + num[1:]
            try:
                float(num2)
            except:
                return "*Didn't work!*"
            num_out2.append(num2)
        
        num_out = num_out2

        op_dict = {'*':mult,
                '/':div,
                '+':add,
                '-':sub,
                '^':exp}
        value = None
        for op in operator_order:                   
            while any(o in op_out for o in op):        
                idx,oo = next((i,o) for i,o in enumerate(op_out) if o in op)
                op_out.pop(idx)
                value1 = float(num_out[idx])
                value2 = float(num_out[idx+1])
                value = op_dict[oo](value1, value2)
                if value == "don't.":
                    return "*Didn't work!*"
                num_out[idx:idx+2] = [value]

        num = num_out[0]
        if num%1 == 0:
            num = int(num)
        
        return str(num)
    except:
        return "*Didn't work!*"

def parse_variants(possible_variants: RegexDict[Variant], raw_variants: list[str],
                   name=None, possible_variant_names=[], macros={}):
    macro_count = 0
    out = {}
    i = 0
    while i < len(raw_variants):
        raw_variant = raw_variants[i]
        if raw_variant.startswith("m!") and raw_variant[2:].split("/", 1)[0] in macros:
            assert macro_count < 50, "Too many macros in one sprite! Are some recursing infinitely?"
            macro_count += 1
            raw_macro, *macro_args = raw_variant[2:].split("/")
            macro = macros[raw_macro].value
            for j, arg in enumerate(macro_args):
                macro = macro.replace(f"${j + 1}", arg)
            # parsing <>s
            k = 0
            checking = False
            start = 0
            while k < len(macro):
                char = macro[k]
                if char == "<":
                    oper = False
                    start = k
                    current = ""
                    checking = True
                elif char == ">" and checking:
                    num = parse_expression(current)
                    if num != "*Didn't work!*":
                        macro = macro[:start] + num + macro[k+1:]
                        checking = False
                        k = -1
                elif char == ":":
                    checking = False
                elif checking:
                    if oper:
                        if char == "-":
                            char = "!"
                        oper = False
                    else:
                        if char in "+-*/":
                            oper = True
                    current += char
                k += 1
            del raw_variants[i]
            raw_variants[i:i] = macro.split(":")  # Extend at index i # wait you can do that? -gabeyk9
            continue
        try:
            final_variant = possible_variants[raw_variant]
            var_type = final_variant.type
            variant_args = [g for g in re.fullmatch(final_variant.pattern, raw_variant).groups() if g is not None]
            final_args = parse_signature(variant_args, final_variant.signature)
            out[var_type] = out.get(var_type, [])
            out[var_type].append(final_variant(*final_args))
        except KeyError:
            for variant_name in possible_variant_names:
                if raw_variant.startswith(variant_name) and len(variant_name):
                    raise errors.BadVariant(name, raw_variant, variant_name)
            raise errors.UnknownVariant(name, raw_variant)
        i += 1
    return out


@dataclass
class TileSkeleton:
    """A tile that hasn't been assigned a sprite yet."""
    name: str = "<empty tile>"
    raw_string: str = ""
    variants: dict[str: Variant, str: Variant, str: Variant] = field(
        default_factory=lambda: {"sprite": [], "tile": [], "post": []})
    palette: str = "default"
    empty: bool = True
    easter_egg: bool = False

    @classmethod
    async def parse(cls, possible_variants, string: str, rule: bool = True, glyph: bool = False, palette: str = "default",
                    bot=None, global_variant="", possible_variant_names=[], macros={}):
        out = cls()
        if string == "-":
            return out
        if rule:
            if string[:5] == "tile_":
                string = string[5:]
            elif string[0] == "$":
                string = string[1:]
                if string[0] == "#":
                    string = "glyph_" + string[1:]
            else:
                string = "text_" + string
        elif glyph:
            if string[:5] == "tile_":
                string = string[5:]
            elif string[0] == "#":
                string = string[1:]
                if string[0] == "$":
                    string = "text_" + string[1:]
            else:
                string = "glyph_" + string
        elif string[0] == "$":
            string = "text_" + string[1:]
        elif string[0] == "#":
            string = "glyph_" + string[1:]
        out.empty = False
        out.raw_string = string
        out.palette = palette
        raw_variants = re.split(r"[;:]", string)
        out.name = raw_variants.pop(0)
        raw_variants[0:0] = global_variant.split(":")
        if out.name in ["-1", "0", "1", "2", "3", "4"] and bot is not None:
            num = out.name
            # Easter egg!
            out.easter_egg = True
            exc = "SELECT DISTINCT name FROM tiles WHERE tiling LIKE " + num
            if num == "2":
                exc += " AND name NOT LIKE 'text_anni'"
                # NOTE: text_anni should be tiling -1, but Hempuli messed it up I guess
            exc += " ORDER BY RANDOM() LIMIT 1"
            async with bot.db.conn.cursor() as cur:
                await cur.execute(exc)
                out.name = (await cur.fetchall())[0][0]
            raw_variants.insert(0, "m!" + num + "ify")
        out.variants |= parse_variants(possible_variants, raw_variants, name=out.name,
                                       possible_variant_names=possible_variant_names, macros=macros)
        return out


def is_adjacent(pos, tile, grid, tile_borders=False) -> bool:
    """Tile is next to a joining tile."""
    w, x, y, z = pos
    joining_tiles = (tile.name, "level", "border")
    if x < 0 or y < 0 or \
            y >= grid.shape[2] or x >= grid.shape[3]:
        return tile_borders
    return grid[w, z, y, x].name in joining_tiles


def get_bitfield(*arr: bool):
    return sum(b << a for a, b in enumerate(list(arr)[::-1]))


def handle_tiling(tile: Tile, grid, pos, tile_borders=False):
    w, z, y, x = pos
    adj_r = is_adjacent((w, x + 1, y, z), tile, grid, tile_borders)
    adj_u = is_adjacent((w, x, y - 1, z), tile, grid, tile_borders)
    adj_l = is_adjacent((w, x - 1, y, z), tile, grid, tile_borders)
    adj_d = is_adjacent((w, x, y + 1, z), tile, grid, tile_borders)
    fallback = constants.TILING_VARIANTS[get_bitfield(adj_r, adj_u, adj_l, adj_d, False, False, False, False)]
    # Variant with diagonal tiles as well, not guaranteed to exist
    # The renderer falls back to the simple variant if it doesn't
    adj_ru = adj_r and adj_u and is_adjacent(
        (w, x + 1, y - 1, z), tile, grid, tile_borders)
    adj_lu = adj_u and adj_l and is_adjacent(
        (w, x - 1, y - 1, z), tile, grid, tile_borders)
    adj_ld = adj_l and adj_d and is_adjacent(
        (w, x - 1, y + 1, z), tile, grid, tile_borders)
    adj_rd = adj_d and adj_r and is_adjacent(
        (w, x + 1, y + 1, z), tile, grid, tile_borders)
    tile.frame = constants.TILING_VARIANTS.get(get_bitfield(adj_r, adj_u, adj_l, adj_d, adj_ru, adj_lu, adj_ld, adj_rd),
                                               fallback)
    tile.fallback_frame = fallback


@dataclass
class Tile:
    """A tile that's ready for processing."""
    name: str = "Undefined (if this is showing, something's gone horribly wrong)"
    sprite: tuple[str, str] | np.ndarray = (constants.BABA_WORLD, "error")
    tiling: int = constants.TILING_NONE
    surrounding: int = 0b00000000  # RULDEQZC
    frame: int = 0
    fallback_frame: int = 0
    wobble_frames: tuple[int] | None = None
    custom_color: bool = False
    color: tuple[int, int] = (0, 3)
    empty: bool = True
    custom: bool = False
    style: Literal["noun", "property", "letter"] = "noun"
    palette: str = None
    overlay: str | None = None
    hue: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    filterimage: str | None = None
    palette_snapping: bool = False
    normalize_gamma: bool = False
    variants: dict[str, list] = field(default_factory=lambda: {
        "sprite": [],
        "tile": [],
        "post": []
    })
    altered_frame: bool = False

    def __hash__(self):
        return hash((self.name, self.sprite if type(self.sprite) is tuple else 0, self.frame, self.fallback_frame,
                     self.empty, self.custom, self.color,
                     self.style, self.palette, self.overlay, self.hue,
                     self.gamma, self.saturation, self.filterimage,
                     self.palette_snapping, self.normalize_gamma, self.altered_frame,
                     hash(tuple(self.variants["sprite"])),
                     hash(tuple(var for var in self.variants["tile"] if var.hashed)),
                     self.custom_color, self.palette))

    @classmethod
    async def prepare(cls, possible_variants, tile: TileSkeleton, tile_data_cache: dict[str, TileData], grid,
                      position: tuple[int, int, int, int], tile_borders: bool = False, ctx: Context = None):
        if tile.empty:
            return cls(name="<empty>")
        name = tile.name
        try:
            metadata = tile_data_cache[name]
            style = constants.TEXT_TYPES[metadata.text_type]
            value = cls(name=tile.name, sprite=(metadata.source, metadata.sprite), tiling=metadata.tiling,
                        color=metadata.active_color, variants=tile.variants, empty=False, style=style,
                        palette=tile.palette)
            if metadata.tiling == constants.TILING_TILE:
                handle_tiling(value, grid, position, tile_borders=tile_borders)
        except KeyError:
            if name[:5] == "text_":
                value = cls(name=name, tiling=constants.TILING_NONE, variants=tile.variants, empty=False, custom=True,
                            palette=tile.palette)
            elif name[:5] == "char_" and ctx is not None:  # allow external calling for potential future things?
                seed = int(name[5:]) if re.fullmatch(r'-?\d+', name[5:]) else name[5:]
                character = ctx.bot.generator.generate(seed=seed)
                color = character[1]["color"]
                value = cls(name=name, tiling=constants.TILING_CHAR, variants=tile.variants, empty=False, custom=True,
                            sprite=character[0], color=color, palette=tile.palette)
            elif name[:6] == "cchar_" and ctx is not None:  # allow external calling for potential future things? again?
                customid = int(name[6:]) if re.fullmatch(r'-?\d+', name[6:]) else name[6:]
                character = ctx.bot.generator.generate(customid=customid)
                color = character[1]["color"]
                value = cls(name=name, tiling=constants.TILING_CHAR, variants=tile.variants, empty=False, custom=True,
                            sprite=character[0], color=color, palette=tile.palette)
            else:
                raise errors.TileNotFound(name)
        for variant in value.variants["tile"]:
            await variant.apply(value)
            if value.surrounding != 0:
                value.frame = constants.TILING_VARIANTS[value.surrounding]
                value.fallback_frame = constants.TILING_VARIANTS[value.surrounding & 0b11110000]
        value.variants["sprite"].append(
            possible_variants["0/3"](value.color, _default_color=True)
        )
        return value


@dataclass
class ProcessedTile:
    """A tile that's been processed, and is ready to render."""
    empty: bool = True
    name: str = "?"
    wobble_frames: tuple[int] | None = None
    frames: list[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = field(
        default_factory=lambda: [None, None, None], repr=False)
    blending: Literal[*tuple(constants.BLENDING_MODES.keys())] = "normal"
    displacement: list[int, int] = field(default_factory=lambda: [0, 0])
    keep_alpha: bool = True

    def copy(self):
        return ProcessedTile(self.empty, self.name, self.wobble, self.frames, self.blending, self.displacement,
                             self.keep_alpha)
