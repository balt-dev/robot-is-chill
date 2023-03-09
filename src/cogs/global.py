from __future__ import annotations

import asyncio
import collections
import glob
import os
import random
import signal
import time
import traceback
from dataclasses import dataclass
from functools import partial
from multiprocessing import Process
from pathlib import Path, PurePath
from urllib.parse import urlparse

import requests

import math
from PIL import Image
import re
from datetime import datetime
from io import BytesIO
from json import load
from time import perf_counter
from typing import Any, OrderedDict, Coroutine, Literal, Optional

import numpy as np
import emoji
from charset_normalizer import from_bytes

import aiohttp
import discord
from discord.ext import commands, menus

from src.cogs.variants import parse_signature
from src.utils import ButtonPages
from ..tile import Tile, TileSkeleton, parse_variants

from .. import constants, errors
from ..db import CustomLevelData, LevelData
from ..types import Bot, Context, Color, RegexDict

from .errorhandler import CommandErrorHandler

@dataclass
class SignText:
    time_start: int = 0
    time_end: int = 0
    x: int = 0
    y: int = 0
    text: str = "null"
    size: float = 1.0
    xo: int = 0
    yo: int = 0
    color: tuple[int, int, int, int] = (255, 255, 255, 255)
    font: Optional[str] = None
    alignment: Optional[str] = None
    stroke: tuple[tuple[int, int, int, int], int] = (0, 0, 0, 0), 0

def try_index(string: str, value: str) -> int:
    """Returns the index of a substring within a string.

    Returns -1 if not found.
    """
    index = -1
    try:
        index = string.index(value)
    except BaseException:
        pass
    return index


# Splits the "text_x,y,z..." shortcuts into "text_x", "text_y", ...
def split_commas(grid: list[list[str]], prefix: str):
    for row in grid:
        to_add = []
        for i, word in enumerate(row):
            if "," in word:
                if word.startswith(prefix):
                    each = re.split(r'(?<!\\),', word)
                    expanded = [each[0]]
                    expanded.extend([prefix + segment for segment in each[1:]])
                    to_add.append((i, expanded))
                else:
                    pass
        for change in reversed(to_add):
            row[change[0]:change[0] + 1] = change[1]
    return grid


async def warn_dangermode(ctx: Context):
    warning_embed = discord.Embed(
        title="Warning: Danger Mode",
        color=discord.Color(16711680),
        description="Danger Mode has been enabled by the developer.\nOutput may not be reliable or may break entirely.\nProceed at your own risk.")
    await ctx.send(embed=warning_embed, delete_after=5)


async def coro_part(func, *args, **kwargs):
    async def wrapper():
        result = func(*args, **kwargs)
        return await result

    return wrapper


class FilterQuerySource(menus.ListPageSource):
    def __init__(
            self, data: list[str]):
        super().__init__(data, per_page=45)

    async def format_page(self, menu: menus.Menu, entries: list[str]) -> discord.Embed:
        embed = discord.Embed(
            title=f"{menu.current_page + 1}/{self.get_max_pages()}",
            color=menu.bot.embed_color
        ).set_footer(
            text="Filters by CenTdemeern1",
            icon_url="https://sno.mba/assets/filter_icon.png"
        )
        while len(entries) > 0:
            field = ""
            for entry in entries[:15]:
                field += f"{entry}\n"
            embed.add_field(
                name="",
                value=field,
                inline=True
            )
            del entries[:15]
        return embed


class GlobalCog(commands.Cog, name="Baba Is You"):
    def __init__(self, bot: Bot):
        self.bot = bot
        with open("config/leveltileoverride.json") as f:
            j = load(f)
            self.level_tile_override = j

    # Check if the bot is loading
    async def cog_check(self, ctx):
        """Only if the bot is not loading assets."""
        return not self.bot.loading

    async def start_timeout(self, ctx, *args, timeout_multiplier: float = 1.0, **kwargs):
        def handler(_signum, _frame):
            asyncio.ensure_future(CommandErrorHandler(ctx.bot).on_command_error(ctx, AssertionError(
                "The command took too long and was timed out.")))

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(constants.TIMEOUT_DURATION * timeout_multiplier))
        await self.render_tiles(ctx, *args, **kwargs)

    async def handle_variant_errors(self, ctx: Context, err: errors.VariantError):
        """Handle errors raised in a command context by variant handlers."""
        try:
            word, variant, *rest = err.args
        except ValueError:
            word, *rest = err.args
            variant = '(Unspecified in error)'
        msg = f"The variant `{variant}` for `{word}` is invalid"
        if isinstance(err, errors.BadTilingVariant):
            tiling = rest[0]
            return await ctx.error(
                f"{msg}, since it can't be applied to tiles with tiling type `{tiling}`."
            )
        elif isinstance(err, errors.TileNotText):
            return await ctx.error(
                f"{msg}, since the tile is not text."
            )
        elif isinstance(err, errors.BadPaletteIndex):
            return await ctx.error(
                f"{msg}, since the color is outside the palette."
            )
        elif isinstance(err, errors.BadLetterVariant):
            return await ctx.error(
                f"{msg}, since letter-style text can only be 1 or 2 letters wide."
            )
        elif isinstance(err, errors.BadMetaVariant):
            depth = rest[0]
            return await ctx.error(
                f"{msg}. `abs({depth})` is greater than the maximum meta depth, which is `{constants.MAX_META_DEPTH}`."
            )
        elif isinstance(err, errors.UnknownVariant):
            return await ctx.error(
                f"There isn't a variant called `{variant}`."
            )
        else:
            return await ctx.error(f"{msg}.")

    async def handle_custom_text_errors(self, ctx: Context, err: errors.TextGenerationError):
        """Handle errors raised in a command context by variant handlers."""
        text, *rest = err.args
        msg = f"The text {text} couldn't be generated automatically"
        if isinstance(err, errors.BadLetterStyle):
            return await ctx.error(
                f"{msg}, since letter style can only applied to a single row of text."
            )
        elif isinstance(err, errors.TooManyLines):
            return await ctx.error(
                f"{msg}, since it has too many lines."
            )
        elif isinstance(err, errors.LeadingTrailingLineBreaks):
            return await ctx.error(
                f"{msg}, since there's `/` characters at the start or end of the text."
            )
        elif isinstance(err, errors.BadCharacter):
            mode, char = rest
            return await ctx.error(
                f"{msg}, since the letter {char} doesn't exist in '{mode}' mode."
            )
        elif isinstance(err, errors.CustomTextTooLong):
            return await ctx.error(
                f"{msg}, since it's too long ({len(text)})."
            )
        else:
            return await ctx.error(f"{msg}.")

    async def handle_grid(
            self, ctx, grid, possible_variants, tile_borders=False):
        """Parses a TileSkeleton array into a Tile grid."""
        tile_data_cache = {
            data.name: data async for data in self.bot.db.tiles(
                {
                    tile.name for tile in grid.flatten()
                }
            )
        }
        return [
            [
                [
                    [
                        # grid gets passed by reference, as it is mutable
                        await Tile.prepare(possible_variants, tile, tile_data_cache, grid, (w, z, y, x), tile_borders, ctx)
                        for x, tile in enumerate(row)
                    ]
                    for y, row in enumerate(layer)
                ]
                for z, layer in enumerate(timestep)
            ]
            for w, timestep in enumerate(grid)
        ]

    async def render_tiles(self, ctx: Context, *, objects: str, rule: bool):
        """Performs the bulk work for both `tile` and `rule` commands."""
        try:
            await ctx.typing()
            ctx.silent = ctx.message.flags.silent
            tiles = emoji.demojize(objects.strip(), language='alias').replace(":hearts:","♥")  # keep the heart, for the people
            tiles = re.sub(r'<a?(:.+?:)\d+?>', r'\1', tiles)
            tiles = re.sub(r"\\(?=[:<])", "", tiles)

            replace_list = [
                ['а', 'a'],
                ['в', 'b'],
                ['е', 'e'],
                ['з', '3'],
                ['к', 'k'],
                ['м', 'm'],
                ['н', 'h'],
                ['о', 'o'],
                ['р', 'p'],
                ['с', 'c'],
                ['т', 't'],
                ['х', 'x'],
                ['ⓜ', ':m:']
            ]
            for src, dst in replace_list:
                tiles = tiles.replace(src, dst)

            # Determines if this should be a spoiler
            spoiler = "||" in tiles
            tiles = tiles.replace("||", "")

            # Check for empty input
            if not tiles:
                return await ctx.error("Input cannot be blank.")

            old_tiles = tiles

            offset = 0
            for match in re.finditer(r"(?<!\\)\"(.*?)(?<!\\)\"", tiles, flags=re.RegexFlag.DOTALL):
                a, b = match.span()
                text = match.group(1)
                prefix = "tile_" if rule else "text_"
                sliced = re.split("([\n ]|$)", text)
                zipped = zip(sliced[1::2], sliced[:-1:2])
                text = "".join(f"{prefix}{t}{joiner}" if t != "-" else f"-{joiner}" for joiner, t in zipped)
                tiles = tiles[:a - offset] + text + tiles[b - offset:]
                offset += (b - a) - len(text)

            # Split input into lines
            word_rows = tiles.splitlines()

            # Split each row into words
            word_grid = [re.split(r"(?<!\\) ", row) for row in word_rows]

            parsing_overhead = time.perf_counter()
            # Check flags
            potential_flags = filter(
                lambda i: i[0].startswith("-"),
                [(word, x, y)
                 for y, row in enumerate(word_grid)
                 for x, word in enumerate(row)]
            )

            kwargs = {}
            to_delete = []
            for potential_flag, x, y in potential_flags:
                for flag in self.bot.flags.list:
                    to_delete, kwargs = await flag.match(ctx, potential_flag, x, y, kwargs, to_delete)
            raw_output = kwargs.get("raw_output", False)
            macros = kwargs.get("macro", {})
            image_format = kwargs.get('image_format', 'gif')
            do_embed = kwargs.get('do_embed', False)
            global_variant = kwargs.get('global_variant', "")
            word_grid = split_commas(word_grid, "char_")
            for x, y in reversed(to_delete):
                del word_grid[y][x]
            try:
                if rule:
                    comma_grid = split_commas(word_grid, "tile_")
                else:
                    comma_grid = split_commas(word_grid, "text_")
            except errors.SplittingException as e:
                cause = e.args[0]
                return await ctx.error(f"I couldn't split the following input into separate objects: \"{cause}\".")

            sign_texts = []

            tilecount = 0
            maxstack = 1
            maxdelta = 1
            try:
                for row in comma_grid:
                    for stack in row:
                        maxstack = max(maxstack, len(re.split(r'(?<!\\)&', stack)))
                        for timeline in re.split(r'(?<!\\)&', stack):
                            maxdelta = max(maxdelta, len(re.split(r'(?<!\\)>', timeline)))
                w, h, d, t = max([len(comma_grid[n]) for n in range(len(comma_grid))]), len(
                    comma_grid), maxstack, maxdelta  # width, height, depth, time
                layer_grid = np.full((t, d, h, w), TileSkeleton(), dtype=object)
                if maxstack > constants.MAX_STACK and ctx.author.id != self.bot.owner_id:
                    return await ctx.error(
                        f"Stack too high ({maxstack}).\nYou may only stack up to {constants.MAX_STACK} tiles on one space.")

                possible_variants = RegexDict([(variant.pattern, variant) for variant in ctx.bot.variants._values if variant.type != "sign"])
                font_variants = RegexDict([(variant.pattern, variant) for variant in ctx.bot.variants._values if variant.type == "sign"])

                def catch(f, *args, **kwargs):
                    try:
                        return f(*args, **kwargs)
                    except:
                        return None

                pal = kwargs.get("palette", "default")
                for y, row in enumerate(comma_grid):
                    for x, stack in enumerate(row):
                        for l, timeline in enumerate(re.split(r'(?<!\\)&', stack)):
                            for d, tile in enumerate(timeline_split := re.split(r'(?<!\\)>', timeline)):
                                if len(tile):
                                    if (match := re.fullmatch(r"\{(.*)}(.*)", tile)) is not None:
                                        sign_text = SignText(text=match.group(1), x=x, y=y, time_start=d)
                                        variants = [variant for variant in match.group(2).split(":") if len(variant)]
                                        variants = parse_variants(font_variants, variants, self.bot).get("sign", [])
                                        for variant in variants:
                                            await variant.apply(sign_text, bot=self.bot, palette=pal)
                                        layer_grid[d:, l, y, x] = TileSkeleton()
                                        for o in range(1, maxdelta - d):
                                            try:
                                                text = timeline_split[d+o]
                                                if len(text):
                                                    print(text, o)
                                                    break
                                            except IndexError:
                                                continue
                                        else:
                                            o = maxdelta - d
                                        sign_text.time_end = d + o
                                        # Sign texts sadly cannot respect layers.
                                        sign_texts.append(sign_text)
                                        print(d, d+o)
                                        continue
                                    tile = re.sub(r"\\(.)", r"\1", tile)
                                    assert not len(tile.split(':', 1)) - 1 or not tile.split(':', 1)[1].count(
                                        ';'), 'Error! Persistent variants (`;`) can\'t come after ephemeral ones (`:`).'
                                    if catch(tile.index, ":") or catch(tile.index, ";") \
                                            or ":" not in tile and ";" not in tile:
                                        tilecount += 1
                                        print(range(layer_grid.shape[0] - d))
                                        # This is done to prevent setting everything to one instance of an object.
                                        layer_grid[d:, l, y, x] = [
                                            await TileSkeleton.parse(
                                                possible_variants, tile, rule,
                                                palette=pal, bot=self.bot,
                                                global_variant=global_variant
                                            )
                                            for _ in range(layer_grid.shape[0] - d)
                                        ]
                                    else:
                                        layer_grid[d:, l, y, x] = await TileSkeleton.parse(
                                            possible_variants,
                                            layer_grid[d - 1, l, y, x].raw_string.split(";" if ";" in tile else ":", 1)[
                                                0] + tile,
                                            rule, macros, bot=self.bot)

                # Get the dimensions of the grid
                grid_shape = layer_grid.shape
                # Don't proceed if the request is too large.
                # (It shouldn't be that long to begin with because of Discord's 2000-character limit)
                if tilecount > constants.MAX_TILES and not (
                        ctx.author.id in [self.bot.owner_id, 280756504674566144]):
                    return await ctx.error(
                        f"Too many tiles ({tilecount}). You may only render up to {constants.MAX_TILES} tiles at once, including empty tiles.")
                # Handles variants based on `:` affixes
                buffer = BytesIO()
                extra_buffer = BytesIO() if raw_output else None
                full_grid = await self.handle_grid(ctx, layer_grid, possible_variants, kwargs.get("tileborder", False))
                parsing_overhead = time.perf_counter() - parsing_overhead
                full_tiles, unique_tiles, rendered_frames, render_overhead = await self.bot.renderer.render_full_tiles(
                    full_grid,
                    random_animations=kwargs.get("random_animations", True),
                    gscale=kwargs.get("gscale", 1),
                    frames=kwargs.get("frames", (1, 2, 3))
                )
                composite_overhead, saving_overhead = await self.bot.renderer.render(
                    full_tiles,
                    out=buffer,
                    extra_out=extra_buffer,
                    sign_texts=sign_texts,
                    **kwargs
                )
            except errors.TileNotFound as e:
                word = e.args[0]
                if word.startswith("tile_") and await self.bot.db.tile(word[5:]) is not None:
                    return await ctx.error(f"The tile `{word}` could not be found. Perhaps you meant `{word[5:]}`?")
                if await self.bot.db.tile("text_" + word) is not None:
                    return await ctx.error(
                        f"The tile `{word}` could not be found. Perhaps you meant `{'text_' + word}`?")
                return await ctx.error(f"The tile `{word}` could not be found.")
            except errors.BadTileProperty as e:
                traceback.print_exc()
                return await ctx.error(f"Error! `{e.args[1]}`")
            except errors.EmptyVariant as e:
                word = e.args[0]
                return await ctx.error(
                    f"You provided an empty variant for `{word}`."
                )
            except errors.VariantError as e:
                return await self.handle_variant_errors(ctx, e)
            except errors.TextGenerationError as e:
                return await self.handle_custom_text_errors(ctx, e)

            filename = datetime.utcnow().strftime(
                f"render_%Y-%m-%d_%H.%M.%S.{image_format}")
            image = discord.File(buffer, filename=filename, spoiler=spoiler)
            description = f"{'||' if spoiler else ''}`{ctx.message.content.replace('||', '').replace('`', '')}`{'||' if spoiler else ''}"
            if do_embed:
                embed = discord.Embed(color=self.bot.embed_color)

                def rendertime(v):
                    v *= 1000
                    nice = False
                    if math.ceil(v) == 69:
                        nice = True
                    if objects == "lag":
                        v *= 100000
                    return f'{v:.4f}' + ("(nice)" if nice else "")

                stats = f'''
    Response time: {rendertime(parsing_overhead + render_overhead + composite_overhead + saving_overhead)} ms
    - Parsing overhead: {rendertime(parsing_overhead)} ms
    - Rendering overhead: {rendertime(render_overhead)} ms
    - Compositing overhead: {rendertime(composite_overhead)} ms
    - Saving overhead: {rendertime(saving_overhead)} ms
    Tiles rendered: {unique_tiles}
    Frames rendered: {rendered_frames}
    Tile matrix shape: {'x'.join(str(n) for n in grid_shape)}
    '''

                embed.add_field(name="Render statistics", value=stats)
            else:
                embed = None
            if extra_buffer is not None:
                extra_buffer.seek(0)
                await ctx.reply(description[:2000], embed=embed,
                                files=[discord.File(extra_buffer, filename=f"raw.zip"), image])
            else:
                await ctx.reply(description[:2000], embed=embed, file=image)
        finally:
            signal.alarm(0)

    @commands.command()
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def tile(self, ctx: Context, *, objects: str = ""):
        """Renders the tiles provided.

        **Flags**
        * See the `flags` commands for all the valid flags.

        **Variants**
        * `:variant`: Append `:variant` to a tile to change different attributes of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile.
        * `&` : Stacks tiles on top of each other. Tiles are rendered in stack order, so in `=rule baba&cursor me`, Baba and Me would be rendered below Cursor.
        * `tile_` : `tile_object` renders regular objects.
        * `,` : `tile_x,y,...` is expanded into `tile_x tile_y ...`
        * `||` : Marks the output gif as a spoiler.

        **Example commands:**
        `tile baba - keke`
        `tile --palette=marshmallow keke:d baba:s`
        `tile text_baba,is,you`
        `tile baba&flag ||cake||`
        `tile -P=mountain -B baba bird:l`
        """
        if self.bot.config['danger_mode']:
            await warn_dangermode(ctx)
        await self.start_timeout(
            ctx,
            objects=objects,
            rule=False)

    @commands.command(aliases=["text"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def rule(self, ctx: Context, *, objects: str = ""):
        """Renders the text tiles provided.

        If not found, the bot tries to auto-generate them!

        **Flags**
        * See the `flags` commands for all the valid flags.

        **Variants**
        * `:variant`: Append `:variant` to a tile to change different attributes of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile.
        * `&` : Stacks tiles on top of each other. Tiles are rendered in stack order, so in `=rule baba&cursor me`, Baba and Me would be rendered below Cursor.
        * `tile_` : `tile_object` renders regular objects.
        * `||` : Marks the output gif as a spoiler.

        **Example commands:**
        `rule baba is you`
        `rule -b rock is ||push||`
        `rule -p=test tile_baba on baba is word`
        `rule baba eat baba - tile_baba tile_baba:l`
        """
        if self.bot.config['danger_mode']:
            await warn_dangermode(ctx)
        await self.start_timeout(
            ctx,
            objects=objects,
            rule=True)

    # Generates tiles from a text file.
    @commands.command()
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def file(self, ctx: Context, rule: str = ''):
        """Renders the text from a file attatchment.

        Add -r, --rule, -rule, -t, --text, or -text to render as text.
        """
        try:
            objects = str(from_bytes((await ctx.message.attachments[0].read())).best())
            await self.start_timeout(
                ctx,
                objects=objects,
                rule=rule in [
                    '-r',
                    '--rule',
                    '-rule',
                    '-t',
                    '--text',
                    '-text'], timeout_multiplier=1.5)
        except IndexError:
            await ctx.error('You forgot to attach a file.')

    async def search_levels(self, query: str, **flags: Any) -> OrderedDict[tuple[str, str], LevelData]:
        """Finds levels by query.

        Flags:
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        """
        levels: OrderedDict[tuple[str, str],
        LevelData] = collections.OrderedDict()
        f_map = flags.get("map")
        f_world = flags.get("world")
        async with self.bot.db.conn.cursor() as cur:
            # [world]/[levelid]
            parts = query.split("/", 1)
            if len(parts) == 2:
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE
						world == :world AND
						id == :id AND (
							:f_map IS NULL OR map_id == :f_map
						) AND (
							:f_world IS NULL OR world == :f_world
						);
					''',
                    dict(
                        world=parts[0],
                        id=parts[1],
                        f_map=f_map,
                        f_world=f_world)
                )
                row = await cur.fetchone()
                if row is not None:
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

            maybe_parts = query.split(" ", 1)
            if len(maybe_parts) == 2:
                maps_queries = [
                    (maybe_parts[0], maybe_parts[1]),
                    (f_world, query)
                ]
            else:
                maps_queries = [
                    (f_world, query)
                ]

            for f_world, query in maps_queries:
                # someworld/[levelid]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE id == :id AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN :default
						THEN NULL
						ELSE world
					END ASC;
					''',
                    dict(
                        id=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [parent]-[map_id]
                segments = query.split("-")
                if len(segments) == 2:
                    await cur.execute(
                        '''
						SELECT * FROM levels
						WHERE parent == :parent AND (
							UNLIKELY(map_id == :map_id) OR (
								style == 0 AND
								CAST(number AS TEXT) == :map_id
							) OR (
								style == 1 AND
								LENGTH(:map_id) == 1 AND
								number == UNICODE(:map_id) - UNICODE("a")
							) OR (
								style == 2 AND
								SUBSTR(:map_id, 1, 5) == "extra" AND
								number == CAST(TRIM(SUBSTR(:map_id, 6)) AS INTEGER) - 1
							)
						) AND (
							:f_map IS NULL OR map_id == :f_map
						) AND (
							:f_world IS NULL OR world == :f_world
						) ORDER BY CASE world
							WHEN :default
							THEN NULL
							ELSE world
						END ASC;
						''',
                        dict(parent=segments[0], map_id=segments[1], f_map=f_map, f_world=f_world,
                             default=constants.BABA_WORLD)
                    )
                    for row in await cur.fetchall():
                        data = LevelData.from_row(row)
                        levels[data.world, data.id] = data

                # [name]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE name == :name AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN :default
						THEN NULL
						ELSE world
					END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [name-ish]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE INSTR(name, :name) AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN :default
						THEN NULL
						ELSE world
					END ASC, number DESC;
					''',
                    dict(
                        name=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

                # [map_id]
                await cur.execute(
                    '''
					SELECT * FROM levels
					WHERE map_id == :map AND parent IS NULL AND (
						:f_map IS NULL OR map_id == :f_map
					) AND (
						:f_world IS NULL OR world == :f_world
					)
					ORDER BY CASE world
						WHEN :default
						THEN NULL
						ELSE world
					END ASC;
					''',
                    dict(
                        map=query,
                        f_map=f_map,
                        f_world=f_world,
                        default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data

        return levels

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @commands.group(name="level", invoke_without_command=True)
    async def level_command(self, ctx: Context, *, query: str):
        """Renders the Baba Is You level from a search term.

        Levels are searched for in the following order:
        * Custom level code (e.g. "1234-ABCD")
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        """
        await self.perform_level_command(ctx, query, mobile=False)

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @level_command.command()
    async def mobile(self, ctx: Context, *, query: str):
        """Renders the mobile Baba Is You level from a search term.

        Levels are searched for in the following order:
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        """
        await self.perform_level_command(ctx, query, mobile=True)

    async def perform_level_command(self, ctx: Context, query: str, *, mobile: bool):
        # User feedback
        await ctx.typing()

        custom_level: CustomLevelData | None = None

        spoiler = query.count("||") >= 2
        fine_query = query.strip().replace("|", "")

        # [abcd-0123]
        if re.match(r"^[A-Za-z\d]{4}-[A-Za-z\d]{4}$", fine_query) and not mobile:
            row = await self.bot.db.conn.fetchone(
                '''
				SELECT * FROM custom_levels WHERE code == ?;
				''',
                fine_query
            )
            if row is not None:
                custom_level = CustomLevelData.from_row(row)
            else:
                # Expensive operation
                await ctx.reply("Searching for custom level... this might take a while", mention_author=False,
                                delete_after=10)
                await ctx.typing()
                async with aiohttp.request("GET",
                                           f"https://baba-is-bookmark.herokuapp.com/api/level/exists?code={fine_query.upper()}") as resp:
                    if resp.status in (200, 304):
                        data = await resp.json()
                        if data["data"]["exists"]:
                            try:
                                custom_level = await self.bot.get_cog("Reader").render_custom_level(fine_query)
                            except ValueError as e:
                                return await ctx.error(
                                    f"The level code is valid, but the level's {e.args[1]} is too big to fit in a GIF. ({e.args[0] * 48} > 65535)")
                            except aiohttp.ClientResponseError:
                                return await ctx.error(
                                    f"The Baba Is Bookmark site returned a bad response. Try again later.")
        if custom_level is None:
            levels = await self.search_levels(fine_query)
            try:
                _, level = levels.popitem(last=False)
            except KeyError:
                return await ctx.error("A level could not be found with that query.")
        else:
            levels = {}
            level = custom_level

        if isinstance(level, LevelData):
            path = level.unique()
            display = level.display()
            rows = [
                f"Name: ||{display}||" if spoiler else f"Name: {display}",
                f"ID: {path}",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: {level.subtitle}"
                )
            mobile_exists = os.path.exists(
                f"target/renders/{level.world}_m/{level.id}.gif")

            if not mobile and mobile_exists:
                rows.append(
                    f"*This level is also on mobile, see `+level mobile {level.unique()}`*"
                )
            elif mobile and mobile_exists:
                rows.append(
                    f"*This is the mobile version. For others, see `+level {level.unique()}`*"
                )

            if mobile and mobile_exists:
                gif = discord.File(
                    f"target/renders/{level.world}_m/{level.id}.gif",
                    filename=level.world + '_m_' + level.id + '.gif',
                    spoiler=spoiler)
            else:
                if mobile and not mobile_exists:
                    rows.append(
                        "*This level doesn't have a mobile version. Using the normal gif instead...*")
                gif = discord.File(
                    f"target/renders/{level.world}/{level.id}.gif",
                    filename=level.world + '_' + level.id + '.gif',
                    spoiler=spoiler)
        else:
            try:
                gif = discord.File(
                    f"target/renders/levels/{level.code}.gif",
                    filename=level.code + '.gif',
                    spoiler=spoiler)
            except FileNotFoundError:
                await self.bot.get_cog("Reader").render_custom_level(fine_query)
                gif = discord.File(
                    f"target/renders/levels/{level.code}.gif",
                    filename=level.code + '.gif',
                    spoiler=spoiler)
            path = level.unique()
            display = level.name
            rows = [
                f"Name: ||{display}|| (by {level.author})"
                if spoiler else f"Name: {display} (by {level.author})",
                f"ID: {path}",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: {level.subtitle}"
                )

        if len(levels) > 0:
            extras = [level.unique() for level in levels.values()]
            if len(levels) > constants.OTHER_LEVELS_CUTOFF:
                extras = extras[:constants.OTHER_LEVELS_CUTOFF]
            paths = ", ".join(f"{extra}" for extra in extras)
            plural = "result" if len(extras) == 1 else "results"
            suffix = ", ..." if len(
                levels) > constants.OTHER_LEVELS_CUTOFF else ""
            rows.append(
                f"*Found {len(levels)} other {plural}: {paths}{suffix}*"
            )

        formatted = "\n".join(rows)

        # Only the author should be mentioned
        mentions = discord.AllowedMentions(
            everyone=False, users=[
                ctx.author], roles=False)

        gif.spoiler = True

        # Send the result
        await ctx.reply(formatted, file=gif, allowed_mentions=mentions)

    @commands.group(aliases=["fi", "filter"], pass_context=True, invoke_without_command=True)
    async def filterimage(self, ctx: Context):
        """Performs filterimage-related actions like template creation,
        conversion and accessing the database."""
        await ctx.error("Invalid subcommand specified! Use `commands filterimage` to see what subcommmands there are.")

    @filterimage.command(aliases=["cvt"])
    async def convert(self, ctx: Context, target_mode: Literal["abs", "absolute", "rel", "relative"]):
        """Converts a filter to its opposing mode. An attachment with the filter is required."""
        # Get the attached image, or throw an error
        try:
            filter_url = ctx.message.attachments[0].url
        except IndexError:
            return await ctx.error("The filter to be converted wasn't attached.")
        filter_headers = requests.head(filter_url, timeout=3).headers
        assert int(filter_headers.get("content-length", 0)) < constants.FILTER_MAX_SIZE, f"Filter is too big!"
        with Image.open(requests.get(filter_url, stream=True).raw) as im:
            fil = np.array(im.convert("RGBA"))
        fil[..., :2] += np.indices(fil.shape[:2], dtype=np.uint8).T * (1 if target_mode.startswith("abs") else -1)
        out = BytesIO()
        Image.fromarray(fil).save(out, format="png", optimize=False)
        out.seek(0)
        filename = f"{Path(ctx.message.attachments[0].filename).stem}-{target_mode}.png"
        file = discord.File(out, filename=filename)
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Converted!",
            description=f'Converted filterimage to {target_mode}.'
        ).set_footer(
            text="Filters by CenTdemeern1",
            icon_url="https://sno.mba/assets/filter_icon.png"
        ).set_image(url=f"attachment://{filename}")
        await ctx.reply(embed=emb, file=file)

    @filterimage.command(aliases=["make", "mk"])
    async def create(self, ctx: Context, target_mode: Literal["abs", "absolute", "rel", "relative"], width: int,
                     height: int):
        """Creates a template filter."""
        size = (width, height)
        fil = np.ones((*size[::-1], 4), dtype=np.uint8) * 0xFF
        fil[..., :2] -= 0x7F
        fil[..., :2] += np.indices(fil.shape[:2], dtype=np.uint8).T * target_mode.startswith("abs")
        out = BytesIO()
        Image.fromarray(fil).save(out, format="png", optimize=False)
        out.seek(0)
        filename = f"filter-{size[0]}x{size[1]}-{target_mode}.png"
        file = discord.File(out, filename=filename)
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Created!",
            description=f'Created filterimage template of size {size} in mode {target_mode}.'
        ).set_footer(
            text="Filters by CenTdemeern1",
            icon_url="https://sno.mba/assets/filter_icon.png"
        ).set_image(url=f"attachment://{filename}")
        await ctx.reply(embed=emb, file=file)

    @filterimage.command(aliases=["reg"])
    async def register(self, ctx: Context, name: str, target_mode: Literal["abs", "absolute", "rel", "relative"]):
        """Adds a filter to the database! Requires an attachment.
    Keep in mind that if the message sent to create this filter is deleted, it will no longer work."""
        assert len(ctx.message.attachments), "An image to be registered has to be supplied!"
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM filterimages WHERE name like ?", name)
            dname = await cursor.fetchone()
            if dname is not None:
                return await ctx.error(f"Filter of name `{name}` already exists in the database!")
            command = "INSERT INTO filterimages VALUES (?, ?, ?, ?);"
            args = (name, target_mode.startswith("abs"), ctx.message.attachments[0].url, ctx.author.id)
            await cursor.execute(command, args)
            emb = discord.Embed(
                color=ctx.bot.embed_color,
                title="Registered!",
                description=f'Registered filter `{name}` in the filterimage database!'
            ).set_footer(
                text="Filters by CenTdemeern1",
                icon_url="https://sno.mba/assets/filter_icon.png"
            )
            await ctx.reply(embed=emb)

    @filterimage.command()
    async def get(self, ctx: Context, name: str):
        """Gets information about a filter."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT * FROM filterimages WHERE name == ?;", name)
            attrs = await cursor.fetchone()
            if attrs is None:
                return await ctx.error(f"Filter of name `{name}` isn't in the filterimage!")
            name, mode, url, author = attrs
            mode = "absolute" if mode else "relative"
            emb = discord.Embed(
                color=ctx.bot.embed_color,
                title=name,
                description=f"Mode: `{mode}`"
            ).set_footer(
                text="Filters by CenTdemeern1",
                icon_url="https://sno.mba/assets/filter_icon.png"
            ).set_image(url=url)
            user = await ctx.bot.fetch_user(author)
            emb.set_author(name=f"{user.name}#{user.discriminator}",
                           icon_url=user.avatar.url if user.avatar is not None else
                           f"https://cdn.discordapp.com/embed/avatars/{int(user.discriminator) % 5}.png")
            await ctx.reply(embed=emb)

    @filterimage.command(aliases=["del", "remove", "rm"])
    async def delete(self, ctx: Context, name: str):
        """Removes a filter from the database. You must have made it to do this."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute(
                f"DELETE FROM filterimages WHERE name == ?{'' if ctx.author.id == ctx.bot.owner_id else ' AND creator == ?'};",
                (name,) if ctx.author.id == ctx.bot.owner_id else (name, ctx.author.id))
            emb = discord.Embed(
                color=ctx.bot.embed_color,
                title="Deleted!",
                description=f"Removed the filter {name} from the database (if it existed in the first place)."
            ).set_footer(
                text="Filters by CenTdemeern1",
                icon_url="https://sno.mba/assets/filter_icon.png"
            )
            await ctx.reply(embed=emb)

    @filterimage.command(aliases=["?", "query", "find", "list"])
    async def search(self, ctx: Context, pattern: str = ".*"):
        """Lists filters that match a regular expression."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT name FROM filterimages WHERE name REGEXP ?", pattern)
            names = [row[0] for row in await cursor.fetchall()]
        return await ButtonPages(FilterQuerySource(sorted(names))).start(ctx)

    @filterimage.command(aliases=["#"])
    async def count(self, ctx: Context):
        """Gets the amount of filters in the database."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) FROM filterimages;")
            count = (await cursor.fetchone())[0]
            await cursor.execute("SELECT COUNT(*) FROM filterimages WHERE absolute == 1;")
            count_abs = (await cursor.fetchone())[0]
        emb = discord.Embed(
            color=ctx.bot.embed_color,
            title="Stats",
            description=f"There are {count} filters in the database, {count_abs} absolute and {count - count_abs} relative."
        ).set_footer(
            text="Filters by CenTdemeern1",
            icon_url="https://sno.mba/assets/filter_icon.png"
        )
        await ctx.reply(embed=emb)

async def setup(bot: Bot):
    await bot.add_cog(GlobalCog(bot))
