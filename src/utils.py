from __future__ import annotations
from src.constants import BABA_WORLD

from typing import Callable, List, Optional, Tuple, TypeVar
from PIL import Image
import numpy as np


def recolor(sprite: Image.Image, rgba: tuple[int, int, int, int]) -> Image.Image:
    """Apply rgba color multiplication (0-255)"""
    print(rgba)
    return Image.fromarray(np.multiply(sprite, np.array(rgba) / 255, casting="unsafe").astype(np.uint8))


class Tile:
    """Represents a tile object, ready to be rendered."""

    def __init__(
            self,
            *,
            name: Optional[str] = None,
            variant: Optional[int] = None,
            color: Optional[Tuple[int, int]] = None,
            source: str = BABA_WORLD,
            meta_level: int = 0,
            style: Optional[str] = None,
            custom: bool = False,
            images: Optional[List[Image.Image]] = None
    ):
        self.name = name
        self.variant = variant
        self.color = color
        self.source = source
        self.style = style
        self.meta_level = meta_level
        self.custom = custom
        self.images = images or []

    def __repr__(self) -> str:
        if self.custom:
            return f"<Custom tile {self.name}>"
        return f"<Tile {self.name} : {self.variant} with {self.color} from {self.source}>"


T = TypeVar("T")


def cached_open(path, *, cache: dict[str, T],
                fn: Callable[[str], T] = open) -> T:
    """Checks whether a path is in the cache, and if so, returns that element.

    Otherwise calls the function on the path.
    """
    if path in cache:
        return cache[path]
    cache[path] = result = fn(path)
    return result
