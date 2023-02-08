from __future__ import annotations

from dataclasses import dataclass, field

from src.constants import BABA_WORLD
from typing import TYPE_CHECKING, Literal, TypedDict
from PIL import Image
import re
import numpy as np
from . import errors, constants
from .cogs.variants import parse_signature
from .db import TileData
from .types import Variant


@dataclass
class TileSkeleton:
    """A tile that hasn't been assigned a sprite yet."""
    name: str
    variants: dict[str: Variant, str: Variant, str: Variant]

    @classmethod
    def parse(cls, ctx, string: str):
        possible_variants = ctx.bot.variants
        raw_variants = re.split(r";|:", string)
        name = raw_variants.pop(0)
        variants = {"skeleton": [], "sprite": [], "tile": []}
        for raw_variant in raw_variants:
            try:
                final_variant = possible_variants[raw_variant]
                variant_args = re.fullmatch(final_variant.pattern, raw_variant).groups()
                final_args = parse_signature(variant_args, final_variant.signature)
                variants[final_variant.type].append(final_variant(*final_args))
            except KeyError:
                raise errors.UnknownVariant(name, raw_variant)
        return cls(name, variants)


@dataclass
class Tile:
    """A tile that's ready for processing."""
    name: str = "Undefined (if this is showing, something's gone horribly wrong)"
    sprite: tuple[str, str] | np.ndarray = (constants.BABA_WORLD, "error")
    frame: tuple[int, int] = 0  # number, fallback
    color: tuple[int, int] | tuple[int, int, int] = (0, 3)  # x, y / r, g, b
    empty: bool = True
    blending: Literal["NORMAL", "ADD", "SUB", "MULT", "CUT", "MASK"] = "NORMAL"
    custom: bool = False
    custom_style: Literal["noun", "property", "letter"] = "noun"
    palette: str = "default"
    overlay: str | None = None
    hue: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    filterimage: str | None = None
    displacement: tuple[int, int] = (0, 0)
    channel_matrix: np.ndarray = field(default_factory=(lambda: np.identity(4)))
    palette_snapping: bool = False
    normalize_gamma: bool = False
    variants: dict[str: Variant, str: Variant] = None

    @classmethod
    def prepare(cls, tile_data_cache: dict[str, TileData], tile: TileSkeleton):
        name = tile.name
        if not len(name):
            return cls(name="<empty>")
        try:
            metadata = tile_data_cache[name]
        except KeyError:
            raise errors.TileNotFound(name)
        print(metadata.active_color)
        value = cls(name=name, sprite=(metadata.source, metadata.sprite), color=metadata.active_color,
                    variants=tile.variants, empty=False)
        for variant in value.variants["skeleton"]:
            variant.apply(value)
        return value


@dataclass
class ProcessedTile:
    """A tile that's been processed, and is ready to render."""
    frames: tuple[Image, Image, Image] | None = None
    blending: Literal["NORMAL", "ADD", "SUB", "MULT", "CUT", "MASK"] = "NORMAL"
    displacement: tuple[int, int] = (0, 0)
    delta: float = 0
