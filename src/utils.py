from __future__ import annotations

import cv2
from discord.ext import menus
from discord.ext.menus.views import ViewMenuPages


from typing import Callable, List, Optional, Tuple, TypeVar
from PIL import Image
import numpy as np

def recolor(sprite: Image.Image | np.ndarray, rgba: tuple[int, int, int, int]) -> Image.Image:
    """Apply rgba color multiplication (0-255)"""
    arr = np.multiply(sprite, np.array(rgba) / 255, casting="unsafe").astype(np.uint8)
    if isinstance(sprite, np.ndarray):
        return arr
    return Image.fromarray(arr)

def composite(a, b, t):
    return (1.0 - t) * a + t * b

class Tile:
    """Represents a tile object, ready to be rendered."""

    def __init__(
            self,
            *,
            name: Optional[str] = None,
            variant: Optional[int] = None,
            color: Optional[Tuple[int, int]] = None,
            source: str = "vanilla",
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
        return f"<Tile {self.name} : {self.variant} with {self.color} from {'[generated]' if self.custom else self.source}>"


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

class ButtonPages(ViewMenuPages,inherit_buttons=False):
    @menus.button('⏮', position=menus.First())
    async def go_to_first_page(self, payload):
        await self.show_page(0)

    @menus.button('◀', position=menus.First(1))
    async def go_to_previous_page(self, payload):
        await self.show_checked_page(self.current_page - 1)

    @menus.button('▶', position=menus.Last(1))
    async def go_to_next_page(self, payload):
        await self.show_checked_page(self.current_page + 1)

    @menus.button('⏭', position=menus.Last(2))
    async def go_to_last_page(self, payload):
        max_pages = self._source.get_max_pages()
        last_page = max(max_pages - 1, 0)
        await self.show_page(last_page)

    @menus.button('⏹', position=menus.Last())
    async def stop_pages(self, payload):
        self.stop()