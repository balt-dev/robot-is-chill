from __future__ import annotations

from dataclasses import dataclass, field

from typing import Literal
from PIL import Image
import re
import numpy as np
from . import errors, constants
from .cogs.variants import parse_signature
from .db import TileData
from .types import Variant, Context


@dataclass
class TileSkeleton:
    """A tile that hasn't been assigned a sprite yet."""
    name: str = "<empty tile>"
    raw_string: str = ""
    variants: dict[str: Variant, str: Variant, str: Variant] = field(
        default_factory=lambda: {"sprite": [], "tile": [], "post": []})
    palette: str = "default"
    empty: bool = True

    @classmethod
    def parse(cls, possible_variants, string: str, rule: bool = True, macros=None, palette: str = "default"):
        if macros is None:
            macros = {}
        out = cls()
        if string == "-":
            return out
        if rule:
            if string[:5] == "tile_":
                string = string[5:]
            else:
                string = "text_" + string
        out.empty = False
        out.raw_string = string
        out.palette = palette
        raw_variants = re.split(r"[;:]", string)
        out.name = raw_variants.pop(0)
        macro_count = 0
        for raw_variant in raw_variants:
            if raw_variant in macros:
                assert macro_count < 50, "Too many macros! Did you recurse them somewhere?"
                macro_count += 1
                raw_variants.extend(macros[raw_variant].split(":"))
                continue
            try:
                final_variant = possible_variants[raw_variant]
                variant_args = re.fullmatch(final_variant.pattern, raw_variant).groups()
                final_args = parse_signature(variant_args, final_variant.signature)
                out.variants[final_variant.type].append(final_variant(*final_args))
            except KeyError:
                raise errors.UnknownVariant(out.name, raw_variant)
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
    custom_color: bool = False
    color: tuple[int, int] = (0, 3)
    empty: bool = True
    blending: Literal["NORMAL", "ADD", "SUB", "MULT", "CUT", "MASK", "XOR"] = "NORMAL"
    custom: bool = False
    style: Literal["noun", "property", "letter"] = "noun"
    palette: str = None
    overlay: str | None = None
    hue: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    filterimage: str | None = None
    displacement: tuple[int, int] = (0, 0)
    palette_snapping: bool = False
    normalize_gamma: bool = False
    variants: dict[str, list] = None
    altered_frame: bool = False

    def __hash__(self):
        return hash((self.name, self.sprite if type(self.sprite) is tuple else 0, self.frame, self.fallback_frame,
                     self.color, self.empty, self.blending, self.custom,
                     self.style, self.palette, self.overlay, self.hue,
                     self.gamma, self.saturation, self.filterimage, self.displacement,
                     self.palette_snapping, self.normalize_gamma,
                     hash(tuple(self.variants["sprite"])),
                     hash(tuple(self.variants["tile"])),
                     hash(tuple(self.variants["post"])), self.custom_color, self.palette))

    @classmethod
    def prepare(cls, tile: TileSkeleton, tile_data_cache: dict[str, TileData], grid,
                position: tuple[int, int, int, int], tile_borders: bool = False, ctx: Context = None):
        if tile.empty:
            return cls(name="<empty>")
        name = tile.name
        try:
            metadata = tile_data_cache[name]
            style = constants.TEXT_TYPES[metadata.text_type]
            value = cls(name=name, sprite=(metadata.source, metadata.sprite), tiling=metadata.tiling,
                        color=metadata.active_color, variants=tile.variants, empty=False, style=style,
                        palette=tile.palette)
            if metadata.tiling == constants.TILING_TILE:
                handle_tiling(value, grid, position, tile_borders=tile_borders)
        except KeyError:
            if name[:5] == "text_":
                value = cls(name=name, tiling=constants.TILING_NONE, variants=tile.variants, empty=False, custom=True,
                            palette=tile.palette)
            elif name[:5] == "char_" and ctx is not None:  # allow external calling for potential future things?
                seed = int(name[5:]) if re.match(r'-?\d+', name[5:]) else name[5:]
                character = ctx.bot.generator.generate(True, seed=seed)
                color = character[1]["color"][1]
                value = cls(name=name, tiling=constants.TILING_CHAR, variants=tile.variants, empty=False, custom=True,
                            sprite=character[0], color=color, palette=tile.palette)
            else:
                raise errors.TileNotFound(name)
        for variant in value.variants["tile"]:
            variant.apply(value)
            if value.surrounding != 0:
                value.frame = constants.TILING_VARIANTS[value.surrounding]
                value.fallback_frame = constants.TILING_VARIANTS[value.surrounding & 0b11110000]
        return value


@dataclass
class ProcessedTile:
    """A tile that's been processed, and is ready to render."""
    frames: tuple[Image, Image, Image] | None = None
    blending: Literal["NORMAL", "ADD", "SUB", "MULT", "CUT", "MASK"] = "NORMAL"
    displacement: tuple[int, int] = (0, 0)
    delta: float = 0
