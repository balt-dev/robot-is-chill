from __future__ import annotations

from dataclasses import dataclass, field
from src.constants import BABA_WORLD
from typing import TYPE_CHECKING, Literal, TypedDict

from PIL import Image
import re

from . import errors

if TYPE_CHECKING:
    # @ps-ignore
    RawGrid = list[list[list['RawTile']]]
    FullGrid = list[list[list['FullTile']]]
    GridIndex = tuple[int, int, int]

@dataclass
class RawTile:
    '''Raw tile given from initial pass of +rule and +tile command parsing'''
    name: str
    variants: list[str]
    
    def __repr__(self) -> str:
        return self.name
    
    @classmethod
    def from_str(cls, string: str) -> RawTile:
        '''Parse from user input'''
        parts = re.split('[\;,\:]', string)
        if any(len(part) == 0 for part in parts):
            raise errors.EmptyVariant(parts[0])
        return RawTile(parts[0], parts[1:])
    
    @property
    def is_text(self) -> bool:
        '''Text is special'''
        return self.name.startswith("text_") or self.name.startswith("rule_")

class TileFields(TypedDict, total=False):
    sprite: tuple[str, str]
    variant_number: int
    color_index: tuple[int, int]
    color_rgb: tuple[int, int, int]
    empty: bool
    cut_alpha: bool
    mask_alpha: bool
    meta_level: int
    custom_direction: int
    custom_style: Literal["noun", "property", "letter"]
    custom: bool
    style_flip: bool
    angle: float
    blur_radius: int
    glitch: int
    filters: list[str]
    displace: tuple[int,int]
    scale: tuple[float,float]
    warp: tuple[tuple[float,float],tuple[float,float],tuple[float,float],tuple[float,float]]
    neon: float
    opacity: float
    pixelate: int
    freeze: bool
    negative: bool
    palette: str
    overlay: str
    brightness: float

@dataclass
class FullTile:
    '''A tile ready to be rendered'''
    name: str
    sprite: tuple[str, str] = BABA_WORLD, "error"
    variant_number: int = 0
    color_index: tuple[int, int] = (0, 3)
    color_rgb: tuple[int, int, int] | None = None
    custom: bool = False
    cut_alpha: bool = False
    mask_alpha: bool = False
    style_flip: bool = False
    empty: bool = False
    displace: tuple[int,int] = (0,0)
    scale: tuple[float,float] = (1,1)
    meta_level: int = 0
    custom_direction: int | None = None
    custom_style: Literal["noun", "property", "letter"] | None = None
    filters: list[str] = field(default_factory=list)
    angle: float = 0
    blur_radius: int = 0
    glitch: int = 0
    warp: tuple[tuple[float,float],tuple[float,float],tuple[float,float],tuple[float,float]] = ((0,0),(0,0),(0,0),(0,0))
    neon: float = 1
    opacity: float = 1
    pixelate: int = 0
    freeze: bool = False
    negative: bool = False
    palette: str = ""
    overlay: str = ""
    brightness: float = 1
    
    @classmethod
    def from_tile_fields(cls, tile: RawTile, fields: TileFields) -> FullTile:
        '''Create a FullTile from a RawTile and TileFields'''
        return FullTile(
            name=tile.name,
            **fields
        )

@dataclass
class ReadyTile:
    '''Tile that's about to be rendered, and already has a prerendered sprite.'''
    frames: tuple[Image.Image, Image.Image, Image.Image] | None
    cut_alpha: bool = False
    mask_alpha: bool = False
    displace: tuple[int,int] = (0,0)
    scale: tuple[int,int] = (1,1)