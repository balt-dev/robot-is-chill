from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable

from src.db import TileData

from .. import constants, errors
from ..tile import FullTile, RawTile, TileFields

import random as rand

if TYPE_CHECKING:
    from ...ROBOT import Bot
    from ..tile import FullGrid, GridIndex, RawGrid

HandlerFn = Callable[['HandlerContext'], TileFields]
DefaultFn = Callable[['DefaultContext'], TileFields]

class ContextBase:
    '''The context that the (something) was invoked in.'''
    def __init__(self, *, bot: Bot, tile: RawTile, grid: RawGrid, index: GridIndex, tile_data_cache: dict[str, TileData], flags: dict[str, Any]) -> None:
        self.bot = bot
        self.tile = tile
        self.grid = grid
        self.index = index
        self.flags = flags
        self.tile_data_cache = tile_data_cache

    @property
    def tile_data(self) -> TileData | None:
        '''Associated tile data'''
        return self.tile_data_cache.get(self.tile.name)
    
    def is_adjacent(self, coordinate: tuple[int, int, int],) -> bool:
        '''Tile is next to a joining tile'''
        l, x, y = coordinate
        joining_tiles = (self.tile.name, "level")
        if x < 0 or y < 0 or y >= len(self.grid[l]) or x >= len(self.grid[l][0]):
            return bool(self.flags.get("tile_borders"))
        return self.grid[l][y][x].name in joining_tiles

class HandlerContext(ContextBase):
    '''The context that the handler was invoked in.'''
    def __init__(self, *, 
        fields: TileFields,
        variant: str,
        groups: tuple[str, ...],
        extras: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.fields = fields
        self.variant = variant
        self.groups = groups
        self.extras = extras
        super().__init__(**kwargs)

class DefaultContext(ContextBase):
    '''The context that a default factory was invoked in.'''

class VariantHandlers:
    def __init__(self, bot: Bot) -> None:
        self.handlers: list[Handler] = []
        self.bot = bot
        self.default_fields: DefaultFn = lambda ctx: {}
        self.tile_data_cache: dict[str, TileData] = {}

    def handler(
        self, 
        *, 
        pattern: str,
        variant_hints: dict[str, str],
        variant_group: str = "Other",
        order: int | None = None
    ) -> Callable[[HandlerFn], Handler]:
        '''Registers a variant handler.
        
        The decorated function should take one argument (`HandlerContext`) and return `TileFields`.

        The handler is invoked when the variant matches `pattern`. If the pattern includes any 
        capturing groups, they are accessible at `HandlerContext.groups`.

        `variant_hints` is a list of (variant, user-friendly representation) pairs.
        Each variant should be valid for the handler. The representation should typically 
        be related to the variant provided, as it will be passed to the user. 
        
        `variant_group` is a key used to group variant handlers together. It should
        be a user-friendly string.

        The lower `order` is, the earlier the handler is prioritized (loosely).
        If `order` is `None` or not given, the handler is given the least priority (loosely).
        '''
        def deco(fn: HandlerFn) -> Handler:
            handler = Handler(pattern, fn, variant_hints, variant_group)
            if order is None:
                self.handlers.append(handler)
            else:
                self.handlers.insert(order, handler)
            return handler
        return deco
    
    def default(self, fn: DefaultFn):
        '''Registers a default field factory.
        
        There can only be one factory.
        
        The function should take no arguments and return a `TileFields`.
        Successive calls to this decorator will replace previous factories.
        '''
        self.default_fields = fn
    
    def finalize(self, fn: Callable[[FullTile], None]):
        '''Registers a finalizer.
        
        There can only be one.
        '''
        self.finalizer = fn

    def all_variants(self) -> list[str]:
        '''All the possible variants
        
        tuples: (real string, representation string)
        '''
        return [
            repr
            for handler in self.handlers
            for repr in handler.hints.values()
        ]

    def valid_variants(self, tile: RawTile, tile_data_cache: dict[str, TileData]) -> dict[str, list[str]]:
        '''Returns the variants that are valid for a given tile.
        This data is pulled from the handler's `hints` attribute.
        
        The output is grouped by the variant group of the handler.
        '''
        out: dict[str, list[str]] = {}
        for handler in self.handlers:
            for variant, repr in handler.hints.items():
                try:
                    groups = handler.match(variant)
                    if groups is not None:
                        mock_ctx = HandlerContext(
                            bot=self.bot,
                            fields={},
                            groups=groups,
                            variant=variant,
                            tile=tile,
                            grid=[[[tile]]],
                            index=(0, 0),
                            extras={},
                            tile_data_cache=tile_data_cache,
                            flags=dict(disallow_custom_directions=True)
                        )
                        handler.handle(mock_ctx)
                except errors.VariantError:
                    pass # Variant not possible
                else:
                    out.setdefault(handler.group, []).append(repr)
        return out

    def handle_tile(self, tile: RawTile, grid: RawGrid, index: GridIndex, tile_data_cache: dict[str, TileData], **flags: Any) -> FullTile:
        '''Take a RawTile and apply its variants to it'''
        default_ctx = DefaultContext(
            bot=self.bot,
            tile=tile,
            grid=grid,
            index=index,
            tile_data_cache=tile_data_cache,
            flags=flags
        )
        fields: TileFields = self.default_fields(default_ctx)
        extras = {}
        for variant in tile.variants:
            failed = True
            for handler in reversed(self.handlers):
                groups = handler.match(variant)
                if groups is not None:
                    failed = False
                    ctx = HandlerContext(
                        bot=self.bot,
                        fields=fields,
                        groups=groups,
                        variant=variant,
                        tile=tile,
                        grid=grid,
                        index=index,
                        extras=extras,
                        tile_data_cache=tile_data_cache,
                        flags=flags
                    )
                    fields.update(handler.handle(ctx))
            if failed:
                raise errors.UnknownVariant(tile, variant)
        full = FullTile.from_tile_fields(tile, fields)
        self.finalizer(full, **flags)
        return full

    async def handle_grid(self, grid: RawGrid, **flags: Any) -> FullGrid:
        '''Apply variants to a full grid of raw tiles'''
        tile_data_cache = {
            data.name: data async for data in self.bot.db.tiles(
                {
                    tile.name for row in grid for stack in row for tile in stack
                },
                maximum_version = flags.get("ignore_editor_overrides", 1000)
            )
        }
        return [
            [
                [
                    self.handle_tile(tile, grid, (x, y, z), tile_data_cache, **flags)
                    for z, tile in enumerate(stack)
                ]
                for x, stack in enumerate(row)
            ]
            for y, row in enumerate(grid)
        ]

class Handler:
    '''Handles a single variant'''
    def __init__(
        self,
        pattern: str,
        fn: HandlerFn,
        hints: dict[str, str],
        group: str
    ):
        self.pattern = pattern
        self.fn = fn
        self.hints = hints
        self.group = group

    def match(self, variant: str) -> tuple[str, ...] | None:
        '''Can this handler take the variant?
        
        Returns the matched groups if possible, else returns `None`.
        '''
        matches = re.fullmatch(self.pattern, variant)
        if matches is not None:
            return matches.groups()
    
    def handle(self, ctx: HandlerContext) -> TileFields:
        '''Handle the variant'''
        return self.fn(ctx)

def split_variant(variant: int | None) -> tuple[int, int]:
    '''The sleeping animation is slightly inconvenient'''
    if variant is None:
        return 0, 0
    dir, anim = divmod(variant, 8)
    if anim == 7:
        dir = (dir + 1) % 4
        anim = -1
    return dir * 8, anim

def join_variant(dir: int, anim: int) -> int:
    '''The sleeping animation is slightly inconvenient'''
    return (dir + anim) % 32

def setup(bot: Bot):
    '''Get the variant handler instance'''
    handlers = VariantHandlers(bot)
    bot.handlers = handlers

    @handlers.default
    def default(ctx: DefaultContext) -> TileFields:
        '''Handles default colors, facing right, and auto-tiling'''
        if ctx.tile.name == "-":
            return {
                "empty": True
            }
        tile_data = ctx.tile_data
        color = (0, 3)
        variant = 0
        if tile_data is not None:
            color = tile_data.active_color
            if tile_data.tiling in constants.AUTO_TILINGS:
                y, l, x = ctx.index
                variant = (
                    + 1 * ctx.is_adjacent((l, x + 1, y))
                    + 2 * ctx.is_adjacent((l, x, y - 1))
                    + 4 * ctx.is_adjacent((l, x - 1, y))
                    + 8 * ctx.is_adjacent((l, x, y + 1))
                )
            if ctx.flags.get("raw_output"):
                color = (0, 3)
            return {
                "variant_number": variant,
                "color_index": color,
                "meta_level": 0,
                "sprite": (tile_data.source, tile_data.sprite),
            }
        if not ctx.tile.is_text:
            raise errors.TileNotFound(ctx.tile)
        return {
            "custom": True,
            "variant_number": variant,
            "color_index": color,
            "meta_level": 0,
        }

    @handlers.finalize
    def finalize(tile: FullTile, **flags) -> None:
        if flags.get("extra_names") is not None:
            if flags["extra_names"]:
                flags["extra_names"][0] = "render"
            else:
                name = tile.name.replace("/", "")
                variant = tile.variant_number
                meta_level = tile.meta_level
                flags["extra_names"].append(
                    meta_level * "meta_" + f"{name}_{variant}"
                )
        if tile.custom and tile.custom_style is None:
            if len(tile.name[5:]) == 2 and flags.get("default_to_letters"):
                tile.custom_style = "letter"
            else:
                tile.custom_style = "noun"

    @handlers.handler(
        pattern=r"|".join(constants.DIRECTION_VARIANTS),
        variant_hints=constants.DIRECTION_REPRESENTATION_VARIANTS,
        variant_group="Alternate sprites"
    )
    def directions(ctx: HandlerContext) -> TileFields:
        dir = constants.DIRECTION_VARIANTS[ctx.variant]
        _, anim = split_variant(ctx.fields.get("variant_number"))
        tile_data = ctx.tile_data
        if tile_data is not None and tile_data.tiling in constants.DIRECTION_TILINGS:
            return {
                "variant_number": join_variant(dir, anim),
                "custom_direction": dir
            }
        elif ctx.flags.get("ignore_bad_directions"):
            return {}
        else:
            if ctx.flags.get("disallow_custom_directions") and not ctx.tile.is_text:
                raise errors.BadTilingVariant(ctx.tile, ctx.variant, "<missing>")
            return {
                "custom_direction": dir
            }

    @handlers.handler(
        pattern=r"|".join(constants.ANIMATION_VARIANTS),
        variant_hints=constants.ANIMATION_REPRESENTATION_VARIANTS,
        variant_group="Alternate sprites"
    )
    def animations(ctx: HandlerContext) -> TileFields:
        anim = constants.ANIMATION_VARIANTS[ctx.variant]
        dir, _ = split_variant(ctx.fields.get("variant_number"))
        tile_data = ctx.tile_data
        tiling = None
        if tile_data is not None:
            tiling = tile_data.tiling
            if tiling in constants.ANIMATION_TILINGS:
                return {
                    "variant_number": join_variant(dir, anim)
                }
        raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)
    
    @handlers.handler(
        pattern=r"|".join(constants.SLEEP_VARIANTS),
        variant_hints=constants.SLEEP_REPRESENTATION_VARIANTS,
        variant_group="Alternate sprites"
    )
    def sleep(ctx: HandlerContext) -> TileFields:
        anim = constants.SLEEP_VARIANTS[ctx.variant]
        dir, _ = split_variant(ctx.fields.get("variant_number"))
        tile_data = ctx.tile_data
        if tile_data is not None:
            if tile_data.tiling in constants.SLEEP_TILINGS:
                return {
                    "variant_number": join_variant(dir, anim)
                }
            raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tile_data.tiling)
        raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, "<missing>")

    @handlers.handler(
        pattern=r"|".join(constants.AUTO_VARIANTS),
        variant_hints=constants.AUTO_REPRESENTATION_VARIANTS,
        variant_group="Alternate sprites"
    )
    def auto(ctx: HandlerContext) -> TileFields:
        tile_data = ctx.tile_data
        tiling = None
        if tile_data is not None:
            tiling = tile_data.tiling
            if tiling in constants.AUTO_TILINGS:
                if ctx.extras.get("auto_override", False):
                    num = ctx.fields.get("variant_number") or 0
                    return {
                        "variant_number": num | constants.AUTO_VARIANTS[ctx.variant]
                    }
                else:
                    ctx.extras["auto_override"] = True
                    return {
                        "variant_number": constants.AUTO_VARIANTS[ctx.variant]
                    }
        raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)

    @handlers.handler(
        pattern=r"\d{1,2}",
        variant_hints={"0": "`raw variant number` (e.g. `8`, `17`, `31`)"},
        variant_group="Alternate sprites"
    )
    def raw_variant(ctx: HandlerContext) -> TileFields:
        variant = int(ctx.variant)
        tile_data = ctx.tile_data
        if tile_data is None:
            raise errors.VariantError("what tile is that even")
        tiling = tile_data.tiling
        try:
            if tiling in constants.AUTO_TILINGS:
                if variant >= 16:
                    raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)
            else:
                dir, anim = split_variant(variant)
                if dir != 0:
                    if tiling not in constants.DIRECTION_TILINGS or dir not in constants.DIRECTIONS:
                        raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)
                if anim != 0:
                    if anim in constants.SLEEP_VARIANTS.values():
                        if tiling not in constants.SLEEP_TILINGS:
                            raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)
                    else:
                        if tiling not in constants.ANIMATION_TILINGS or anim not in constants.ANIMATION_VARIANTS.values():
                            raise errors.BadTilingVariant(ctx.tile.name, ctx.variant, tiling)
            return {
                "variant_number": variant
            }
        except:
            return {
                "variant_number": -1
            }

    @handlers.handler(
        pattern=r"|".join(constants.COLOR_NAMES),
        variant_hints=constants.COLOR_REPRESENTATION_VARIANTS,
        variant_group="Colors"
    )
    def color_name(ctx: HandlerContext) -> TileFields:
        return {
            "color_index": constants.COLOR_NAMES[ctx.variant]
        }

    @handlers.handler(
        pattern=r"(\d)/(\d)",
        variant_hints={"0/0": "`palette_x/palette_y` (Color palette index, e.g. `0/3`)"},
        variant_group="Colors"
    )
    def color_index(ctx: HandlerContext) -> TileFields:
        x, y = int(ctx.groups[0]), int(ctx.groups[1])
        if x > 6 or y > 4:
            raise errors.BadPaletteIndex(ctx.tile.name, ctx.variant)
        return {
            "color_index": (int(ctx.groups[0]), int(ctx.groups[1]))
        }
    
    @handlers.handler(
        pattern=r"#([0-9a-fA-F]{1,6})",
        variant_hints={"#ffffff": "`#hex_code` (Color hex code, e.g. `#f055ee`)"},
        variant_group="Colors"
    )
    def color_rgb(ctx: HandlerContext) -> TileFields:
        color = int(ctx.groups[0], base=16)
        red = color >> 16
        green = (color & 0x00ff00) >> 8
        blue = color & 0x0000ff
        return {
            "color_rgb": (red, green, blue)
        }
        
    @handlers.handler(
        pattern='|'.join(constants.CUSTOM_COLOR_NAMES),
        variant_hints=constants.CUSTOM_COLOR_REPRESENTATION_VARIANTS,
        variant_group="Custom Colors"
    )
    def color_rgb(ctx: HandlerContext) -> TileFields:
        return {
            "color_rgb": constants.CUSTOM_COLOR_NAMES[ctx.variant]
        }
    
    @handlers.handler(
        pattern=r"random",
        variant_hints={"random": "`random` (Recolors the sprite to a random color.)"},
        variant_group="Custom Colors"
    )
    def random(ctx: HandlerContext) -> TileFields:
        return {
            "color_rgb": [rand.randint(0,255) for _ in range(3)]
        }
        
    @handlers.handler(
        pattern=r"inactive|in",
        variant_hints={"in": "`inactive` / `in` (Inactive text color)"},
        variant_group="Colors"
    )
    def inactive(ctx: HandlerContext) -> TileFields:
        color = ctx.fields.get("color_index", (0, 3))
        tile_data = ctx.tile_data
        if tile_data is not None and ctx.tile.is_text:
            # only the first `:inactive` should pick the provided color
            if color == tile_data.active_color:
                return {
                    "color_index": tile_data.inactive_color
                }
        return {
            "color_index": constants.INACTIVE_COLORS[color]
        }

    @handlers.handler(
        pattern=r"hide",
        variant_hints={"hide": "`hide` (It's a mystery)"},
        variant_group="Filters"
    )
    def hide(ctx: HandlerContext) -> TileFields:
        return {
            "empty": True
        }

    @handlers.handler(
        pattern=r"meta|m(?!\d)",
        variant_hints={"m": "`meta` / `m` (+1 meta layer)"},
        variant_group="Filters"
    )
    def meta(ctx: HandlerContext) -> TileFields:
        level = ctx.fields.get("meta_level", 0)
        if abs(level) >= constants.MAX_META_DEPTH:
            raise errors.BadMetaVariant(ctx.tile.name, ctx.variant, level)
        return {
            "meta_level": level + 1
        }
    
    @handlers.handler(
        pattern=r"m(-{0,1}\d+)",
        variant_hints={"m1": "`mX` (A specific meta depth, e.g. `m1`, `m3`)"},
        variant_group="Filters"
    )
    def meta_absolute(ctx: HandlerContext) -> TileFields:
        level = int(ctx.groups[0])
        if abs(level) > constants.MAX_META_DEPTH:
            raise errors.BadMetaVariant(ctx.tile.name, ctx.variant, level)
        return {
            "meta_level": level
        }

    @handlers.handler(
        pattern=r"noun",
        variant_hints={"noun": "`noun` (Noun-style text)"},
        variant_group="Custom text"
    )
    def noun(ctx: HandlerContext) -> TileFields:
        if not ctx.tile.is_text:
            raise errors.TileNotText(ctx.tile.name, "noun")
        tile_data = ctx.tile_data
        if tile_data is not None:
            if constants.TEXT_TYPES[tile_data.text_type] == "property":
                return {
                    "style_flip": True,
                    "custom_style": "noun"
                }
        return {
            "custom": True,
            "custom_style": "noun"
        }
    
    @handlers.handler(
        pattern=r"letter|let",
        variant_hints={"let": "`letter` / `let` (Letter-style text)"},
        variant_group="Custom text"
    )
    def letter(ctx: HandlerContext) -> TileFields:
        if not ctx.tile.is_text:
            raise errors.TileNotText(ctx.tile.name, "letter")
        if len(ctx.tile.name[5:]) > 2:
            raise errors.BadLetterVariant(ctx.tile.name, "letter")
        return {
            "custom": True,
            "custom_style": "letter"
        }
    
    @handlers.handler(
        pattern=r"property|prop",
        variant_hints={"prop": "`property` / `prop` (Property-style text)"},
        variant_group="Custom text"
    )
    def property(ctx: HandlerContext) -> TileFields:
        tile_data = ctx.tile_data
        if not ctx.tile.is_text:
            if tile_data is not None:
                # this will be funny
                return {
                    "style_flip": True,
                    "custom_style": "property"
                }
            else:
                raise errors.VariantError("yet again (but this time on a technicality)")
        if tile_data is not None:
            if constants.TEXT_TYPES[tile_data.text_type] == "noun":
                return {
                    "style_flip": True,
                    "custom_style": "property"
                }
        return {
            "custom_style": "property",
            "custom": True,
        }

    @handlers.handler(
        pattern=r"mask",
        variant_hints={"mask": "`mask` (Tiles below get cut to this)"},
        variant_group="Filters"
    )
    def mask(ctx: HandlerContext) -> TileFields:
        return{
            "mask_alpha": True
        }
    
    @handlers.handler(
        pattern=r"cut",
        variant_hints={"cut": "`cut` (Tiles below get this cut from them)"},
        variant_group="Filters"
    )
    def cut(ctx: HandlerContext) -> TileFields:
        return{
            "cut_alpha": True
        }
        
    @handlers.handler(
        pattern=r"neon(?:(-?\d+(?:\.\d+)?))?",
        variant_hints={"neon": "`neon[float]` (Pixels surrounded by identical pixels get their alpha divided by n. If not specified, n is 1.4.)"},
        variant_group="Filters"
    )
    def neon(ctx: HandlerContext) -> TileFields:
        neon = ctx.groups[0] or 1.4
        return {
            "neon": float(neon)
        }
    
    @handlers.handler(
        pattern=r"pixelate(?:([\d]+))?",
        variant_hints={"pixelate": "`pixelate<int>` (Pixelates the sprite with a radius of n.)"},
        variant_group="Filters"
    )
    def neon(ctx: HandlerContext) -> TileFields:
        pixelate = ctx.groups[0]
        return {
            "pixelate": int(pixelate)
        }
    
    @handlers.handler(
        pattern=r"opacity(?:([\d\.]+))?",
        variant_hints={"opacity": "`opacity<float>` (The image gets less opaque by n.)"},
        variant_group="Filters"
    )
    def neon(ctx: HandlerContext) -> TileFields:
        opacity = ctx.groups[0] or 1
        return {
            "opacity": float(opacity)
        }

    @handlers.handler(
        pattern=r"blank",
        variant_hints={"blank": "`blank` (Makes all of the sprite its palette-defined color.)"},
        variant_group="Filters"
    )
    def blank(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["blank"]
            }
        except:
            return{
                "filters": ["blank"]
            }
            
    @handlers.handler(
        pattern=r"face|eyes",
        variant_hints={"face": "`face` (Tries to extract the face of a sprite by removing all but the least used color.)"},
        variant_group="Filters"
    )
    def face(ctx: HandlerContext) -> TileFields:
        return{
            "colslice": [-1]
        }
          
    @handlers.handler(
        pattern=r"main",
        variant_hints={"main": "`main` (Removes all but the most used color.)"},
        variant_group="Filters"
    )
    def main(ctx: HandlerContext) -> TileFields:
        return{
            "colslice": [0]
        }
            
    @handlers.handler(
        pattern=r"land",
        variant_hints={"land": "`land` (Displaces the sprite to the floor.)"},
        variant_group="Filters"
    )
    def main(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["land"]
            }
        except:
            return{
                "filters": ["land"]
            }
    
    @handlers.handler(
        pattern=r"flipx",
        variant_hints={"flipx": "`flipx` (Flips sprite horizontally.)"},
        variant_group="Filters"
    )
    def flipx(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["flipx"]
            }
        except:
            return{
                "filters": ["flipx"]
            }

    @handlers.handler(
        pattern=r"reverse",
        variant_hints={"reverse|rev": "`reverse` (Swaps a sprite's colors based off of frequency.)"},
        variant_group="Filters"
    )
    def flipx(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["reverse"]
            }
        except:
            return{
                "filters": ["reverse"]
            }
            
    @handlers.handler(
        pattern=r"flipy",
        variant_hints={"flipy": "`flipy` (Flips sprite vertically.)"},
        variant_group="Filters"
    )
    def flipy(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["flipy"]
            }
        except:
            return{
                "filters": ["flipy"]
            }
    
    @handlers.handler(
        pattern=r"scanx",
        variant_hints={"scanx": "`scanx` (Applies a horizonal scanline effect.)"},
        variant_group="Filters"
    )
    def scanx(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["scanx"]
            }
        except:
            return{
                "filters": ["scanx"]
            }
    
    @handlers.handler(
        pattern=r"scany",
        variant_hints={"scany": "`scany` (Applies a vertical scanline effect.)"},
        variant_group="Filters"
    )
    def scany(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["scany"]
            }
        except:
            return{
                "filters": ["scany"]
            }
    
    @handlers.handler(
        pattern=r"invert|inv",
        variant_hints={"invert": "`invert` (Inverts sprite color.)"},
        variant_group="Filters"
    )
    def invert(ctx: HandlerContext) -> TileFields:
        try:
            return{
                "filters": ctx.fields.get("filters") + ["invert"]
            }
        except:
            return{
                "filters": ["invert"]
            }
            
    @handlers.handler(
        pattern=r"(?:floodfill|flood|fill)([01]\.\d+)?",
        variant_hints={"floodfill": "`floodfill` (Fills in all open pockets in the sprite.)"},
        variant_group="Filters"
    )
    def floodfill(ctx: HandlerContext) -> TileFields:
        return{
            "floodfill": float(ctx.groups[0] or 0)
        }

    @handlers.handler(
        pattern=r"fisheye(-?\d+(?:\.\d+)?)?",
        variant_hints={"fisheye": "`fisheye[n]` (Applies fisheye effect. n is intensity, defaulting to 0.5.)"},
        variant_group="Filters"
    )
    def fisheye(ctx: HandlerContext) -> TileFields:
        fish = ctx.groups[0] or 0.5
        return {
            "fisheye": float(fish) 
        }
        
    @handlers.handler(
        pattern=r"(?:glitch|g)(\d+(?:\.\d+)?)(?:\/(\d+(?:\.\d+)?))?",
        variant_hints={"glitch": "`glitch<float>[/float]` (Displaces some pixels. With 123/.456, 123 is the max displacement distance, with a 45.6% chance of displacing a pixel.)"},
        variant_group="Filters"
    )
    def glitch(ctx: HandlerContext) -> TileFields:
        intensity = ctx.groups[0] or 0
        chance = ctx.groups[1] or 100
        return {
            "glitch": (float(intensity),float(chance))
        }
    
    @handlers.handler(
        pattern=r"blur(\d)",
        variant_hints={"blur": "`blur<int>` (Gaussian blurs the sprite with a radius of n.)"},
        variant_group="Filters"
    )
    def blur_radius(ctx: HandlerContext) -> TileFields:
        radius = ctx.groups[0] or 0
        return {
            "blur_radius": int(radius) 
        }
        
    @handlers.handler(
        pattern=r"rotate(-{0,1}[\d\.]+)",
        variant_hints={"rotate": "`rotate<int>` (Rotates the sprite n degrees counterclockwise)"},
        variant_group="Filters"
    )
    def rotate(ctx: HandlerContext) -> TileFields:
        angle = ctx.groups[0] or 0
        return {
            "angle": float(angle) 
        }

    @handlers.handler(
        pattern=r"scale([\d\.]+)(?:\/([\d\.]+))?",
        variant_hints={"scale": "`scale<int>/[int]` (Scales the sprite by n1 on the x axis and n2 on the y axis, or n1 if n2 isn't specified.)"},
        variant_group="Filters"
    )
    def scale(ctx: HandlerContext) -> TileFields:
        n = float(ctx.groups[1]) if ctx.groups[1] else float(ctx.groups[0])
        return {
            "scale": (max(min(float(ctx.groups[0]),48),0.01),max(min(n,48),0.01))
        }

    @handlers.handler(
        pattern=r"add",
        variant_hints={"add": "`add` (Makes the tile's RGB add to the tiles below.)"},
        variant_group="Filters"
    )
    def add(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'add'
        }
        
    @handlers.handler(
        pattern=r"xor",
        variant_hints={"xor": "`xor` (Makes the tile's RGB XOR with the tiles below.)"},
        variant_group="Filters"
    )
    def xor(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'xora'
        }
        
    @handlers.handler(
        pattern=r"xora",
        variant_hints={"xora": "`xora` (Makes the tile's RGBA XOR with the tiles below.)"},
        variant_group="Filters"
    )
    def xora(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'xor'
        }
    
    @handlers.handler(
        pattern=r"subtract",
        variant_hints={"subtract": "`subtract` (Makes the tile's RGB subtract from the tiles below.)"},
        variant_group="Filters"
    )
    def subtract(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'subtract'
        }
    
    @handlers.handler(
        pattern=r"maximum",
        variant_hints={"maximum": "`maximum` (Compares the tile's RGB from the tiles below, and keeps the max for each channel.)"},
        variant_group="Filters"
    )
    def maximum(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'maximum'
        }

    @handlers.handler(
        pattern=r"minimum",
        variant_hints={"minimum": "`minimum` (Compares the tile's RGB from the tiles below, and keeps the minimum for each channel.)"},
        variant_group="Filters"
    )
    def minimum(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'minimum'
        }

    @handlers.handler(
        pattern=r"multiply",
        variant_hints={"multiply": "`multiply` (Makes the tile's RGB multiply with the tiles below.)"},
        variant_group="Filters"
    )
    def subtract(ctx: HandlerContext) -> TileFields:
        return {
            "blending": 'multiply'
        }

    @handlers.handler(
        pattern=r"displace(\-?\d{1,3})\/(\-?\d{1,3})",
        variant_hints={"displace": "`displace<int>/<int>` (Displaces the sprite by x pixels to the right and y pixels downwards.)"},
        variant_group="Filters"
    )
    def displace(ctx: HandlerContext) -> TileFields:
        return {
            "displace": (0-int(ctx.groups[0]),0-int(ctx.groups[1]))
        }

    @handlers.handler(
        pattern=r"warp\((\-?[\d\.]+)\/(\-?[\d\.]+)\)\((\-?[\d\.]+)\/(\-?[\d\.]+)\)\((\-?[\d\.]+)\/(\-?[\d\.]+)\)\((\-?[\d\.]+)\/(\-?[\d\.]+)\)",
        variant_hints={"warp": "`warp(<int>/<int>)(<int>/<int>)(<int>/<int>)(<int>/<int>)` \n Transforms the corners of the image.\n Order goes top left, top right, bottom right, bottom left. \n Values are the offset of the point, as (right/down)."},
        variant_group="Filters"
    )
    def warp(ctx: HandlerContext) -> TileFields:
        return {
            "warp": ((float(ctx.groups[0]),float(ctx.groups[1])),(float(ctx.groups[2]),float(ctx.groups[3])),(float(ctx.groups[4]),float(ctx.groups[5])),(float(ctx.groups[6]),float(ctx.groups[7])))
        }
    
    @handlers.handler(
        pattern=r"freeze",
        variant_hints={"freeze": "`freeze` (Freezes the first wobble frame of the tile.)"},
        variant_group="Filters"
    )
    def freeze(ctx: HandlerContext) -> TileFields:
        return{
            "freeze": True
        }

    @handlers.handler(
        pattern=r"negative|neg",
        variant_hints={"negative": "`negative` (RGB color inversion.)"},
        variant_group="Filters"
    )
    def negative(ctx: HandlerContext) -> TileFields:
        return{
            "negative": True
        }
        
    @handlers.handler(
        pattern=r"complement|comp",
        variant_hints={"complement": "`complement` (HSL hue inversion.)"},
        variant_group="Filters"
    )
    def complement(ctx: HandlerContext) -> TileFields:
        return{
            "hueshift": 180
        }
        
    @handlers.handler(
        pattern=r"(?:hueshift|hs)(-?[\d\.]+)",
        variant_hints={"hueshift": "`hueshift<float>` (HSL hue shift.)"},
        variant_group="Filters"
    )
    def hueshift(ctx: HandlerContext) -> TileFields:
        return{
            "hueshift": float(ctx.groups[0]) if ctx.groups[0] else 0.0
        }
        
    @handlers.handler(
        pattern=r"(?:palette\/|p\!)(\w+)",
        variant_hints={"palette": "`palette/<palettename>` (Applies a different color palette to the tile.)"},
        variant_group="Filters"
    )
    def palette(ctx: HandlerContext) -> TileFields:
        return{
            "palette": ctx.groups[0]
        }

    @handlers.handler(
        pattern=r"(?:overlay\/|o\!)(\w+)",
        variant_hints={"overlay": "`(o!|overlay/)<overlayname>` (Applies an overlay on the tile.)"},
        variant_group="Filters"
    )
    def overlay(ctx: HandlerContext) -> TileFields:
        return{
            "overlay": ctx.groups[0]
        }
    
    @handlers.handler(
        pattern=r"(?:brightness|bright)([\d\.]*)",
        variant_hints={"brightness": "`brightness<factor>` (Darkens or brightens the tile by multiplying it by factor.)"},
        variant_group="Filters"
    )
    def brightness(ctx: HandlerContext) -> TileFields:
        return{
            "brightness": float(ctx.groups[0])
        }

    @handlers.handler(
        pattern=r"wavex([\d\.]*)\/([\d\.]*)\/([\d\.]*)",
        variant_hints={"wavex": "`wavex<offset>/<amplitude>/<speed>` (Creates a wave of horizonal lines going in order from top to bottom.)"},
        variant_group="Filters"
    )
    def wavex(ctx: HandlerContext) -> TileFields:
        return{
            "wavex": (float(ctx.groups[0]),float(ctx.groups[1]),float(ctx.groups[2]))
        }

    @handlers.handler(
        pattern=r"wavey([\d\.]*)\/([\d\.]*)\/([\d\.]*)",
        variant_hints={"wavey": "`wavey<offset>/<amplitude>/<speed>` (Creates a wave of vertical lines going in order from left to right.)"},
        variant_group="Filters"
    )
    def wavey(ctx: HandlerContext) -> TileFields:
        return{
            "wavey": (float(ctx.groups[0]),float(ctx.groups[1]),float(ctx.groups[2]))
        }

    @handlers.handler(
        pattern=r"gradientx([\d\.]*)\/([\d\.]*)\/([\d\.]*)\/([\d\.]*)",
        variant_hints={"gradientx": "`gradientx<start>/<end>/<startvalue>/<endvalue>` (Creates a horizonal gradient on the tile going from left to right.)"},
        variant_group="Filters"
    )
    def gradientx(ctx: HandlerContext) -> TileFields:
        return{
            "gradientx": (float(ctx.groups[0]),float(ctx.groups[1]),float(ctx.groups[2]),float(ctx.groups[3]))
        }

    @handlers.handler(
        pattern=r"gradienty([\d\.]*)\/([\d\.]*)\/([\d\.]*)\/([\d\.]*)",
        variant_hints={"gradienty": "`gradienty<start>/<end>/<startvalue>/<endvalue>` (Creates a vertical gradient on the tile going from top to bottom.)"},
        variant_group="Filters"
    )
    def gradienty(ctx: HandlerContext) -> TileFields:
        return{
            "gradienty": (float(ctx.groups[0]),float(ctx.groups[1]),float(ctx.groups[2]),float(ctx.groups[3]))
        }

    @handlers.handler(
        pattern=r"(abs){0,1}(?:filterimage\/|fi!|filterimage=|fi=)(.+)",
        variant_hints={"filterimage": "`[abs]filterimage/<url>` `[abs]filterimage=<url>` `[abs]fi!<url>` `[abs]fi=<url>` applies a filter image.\nWarning: big images may take a while to render.\nImages bigger than 64 pixels not recommended.\nTip: Remove the http(s):// part from the URLs!\nUse variant with `abs` in front of the name to use absolute positions!\nTo use an image from the database, use `db!<name>` as the url!"},
        variant_group="Filters"
    )
    def filterimage(ctx: HandlerContext) -> TileFields:
        a = ctx.groups[0] if ctx.groups[0] else ""
        url = ctx.groups[1]
        if url.startswith("db!"):
            return {
                "filterimage": a+url
            }
        else:
            url=url.replace("localhost","")
            if url[:url.find("/")].replace(".","").isnumeric():
                url=""
            return {
                "filterimage": a+"https://"+url
            }

    @handlers.handler(
        pattern=r"crop\((-?\d+?)\/(-?\d+?)\/(-?\d+?)\/(-?\d+?)\)",
        variant_hints={"crop": "`crop(<x>/<y>/<width>/<height>)` (Crops the sprite to the rectange defined as n3 as width, n4 as height, with the point at n1/n2 being its top-left corner)"},
        variant_group="Filters"
    )
    def crop(ctx: HandlerContext) -> TileFields:
        return{
            "crop": (int(ctx.groups[0]),int(ctx.groups[1]),int(ctx.groups[2]),int(ctx.groups[3]))
        }
        
    @handlers.handler(
        pattern=r"pad\((\d+?)\/(\d+?)\/(\d+?)\/(\d+?)\)",
        variant_hints={"pad": "`pad(<left>/<top>/<right>/<bottom>)` (Pads the sprite with transparency on each of its sides.)"},
        variant_group="Filters"
    )
    def pad(ctx: HandlerContext) -> TileFields:
        return{
            "pad": (int(ctx.groups[0]),int(ctx.groups[1]),int(ctx.groups[2]),int(ctx.groups[3]))
        }
        
    @handlers.handler(
        pattern=r"nothing|none|n|-",
        variant_hints={"nothing": "`nothing` (Literally does nothing.)"},
        variant_group="Filters"
    )
    def nothing(ctx: HandlerContext) -> TileFields:
        return{}
    
    @handlers.handler(
        pattern=r"(?:color|col|c)(-?\d+)(?:\/(-?\d+))?",
        variant_hints={"color": "`color<n>[/<n>]` (Cuts all but the specified color or range of colors from the image. First number is inclusive, second is exclusive.)"},
        variant_group="Filters"
    )
    def color(ctx: HandlerContext) -> TileFields:
        return{
            "colslice": tuple([int(n) for n in ctx.groups if type(n) != type(None)]) 
            if len(ctx.groups) == 2 
            else int(ctx.groups[0])
        }
        
    return handlers