from __future__ import annotations
import collections
import os
import requests

from PIL import Image
import re
from datetime import datetime
from io import BytesIO
from json import load
from os import listdir
from time import time
from typing import Any, OrderedDict, TYPE_CHECKING

import aiohttp
import discord
from discord.ext import commands

if TYPE_CHECKING:
    from ..tile import RawGrid

from .. import constants, errors
from ..db import CustomLevelData, LevelData
from ..tile import RawTile
from ..types import Bot, Context


def try_index(string: str, value: str) -> int:
    '''Returns the index of a substring within a string.
    Returns -1 if not found.
    '''
    index = -1
    try:
        index = string.index(value)
    except:
        pass
    return index

# Splits the "text_x,y,z..." shortcuts into "text_x", "text_y", ...
def split_commas(grid: list[list[str]], prefix: str):
    for row in grid:
        to_add = []
        for i, word in enumerate(row):
            if "," in word:
                if word.startswith(prefix):
                    each = word.split(",")
                    expanded = [each[0]]
                    expanded.extend([prefix + segment for segment in each[1:]])
                    to_add.append((i, expanded))
                else:
                    raise errors.SplittingException(word)
        for change in reversed(to_add):
            row[change[0]:change[0] + 1] = change[1]
    return grid

class GlobalCog(commands.Cog, name="Baba Is You"):
    def __init__(self, bot: Bot):
        self.bot = bot
        with open("config/leveltileoverride.json") as f:
            j = load(f)
            self.level_tile_override = j

    # Check if the bot is loading
    async def cog_check(self, ctx):
        '''Only if the bot is not loading assets'''
        return not self.bot.loading

    async def handle_variant_errors(self, ctx: Context, err: errors.VariantError):
        '''Handle errors raised in a command context by variant handlers'''
        word, variant, *rest = err.args
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
                f"{msg}. `{depth}` is greater than the maximum meta depth, which is `{constants.MAX_META_DEPTH}`."
            )
        elif isinstance(err, errors.UnknownVariant):
            return await ctx.error(
                f"The variant `{variant}` is not valid."
            )
        else:
            return await ctx.error(f"{msg}.")

    async def handle_custom_text_errors(self, ctx: Context, err: errors.TextGenerationError):
        '''Handle errors raised in a command context by variant handlers'''
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

    def parse_raw(self, grid: list[list[list[str]]], *, rule: bool) -> RawGrid:
        '''Parses a string grid into a RawTile grid'''
        return [
            [
                [
                    RawTile.from_str(
                        "-" if tile == "-" else tile[5:] if tile.startswith("tile_") else f"text_{tile}"
                    ) if rule else RawTile.from_str(
                        "-" if tile == "text_-" else tile
                    )
                    for tile in stack
                ]
                for stack in row
            ]
            for row in grid
        ]

    async def render_tiles(self, ctx: Context, *, objects: str, rule: bool):
        '''Performs the bulk work for both `tile` and `rule` commands.'''
        await ctx.trigger_typing()
        start = time()
        tiles = objects.lower().strip().replace("\\", "")

        # Determines if this should be a spoiler
        spoiler = "|" in tiles
        tiles = tiles.replace("|", "")
        
        # Check for empty input
        if not tiles:
            return await ctx.error("Input cannot be blank.")

        # Split input into lines
        word_rows = tiles.splitlines()
        
        # Split each row into words
        word_grid = [row.split() for row in word_rows]

        # Check flags
        potential_flags = []
        potential_count = 0
        try:
            for y, row in enumerate(word_grid):
                for x, word in enumerate(row):
                    if potential_count == 6:
                        raise Exception
                    potential_flags.append((word, x, y))
                    potential_count += 1
        except Exception: pass
        background = None
        palette = "default"
        to_delete = []
        raw_output = False
        default_to_letters = False
        frames = [1,2,3]
        for flag, x, y in potential_flags:
            bg_match = re.fullmatch(r"(--background|-b)(=(\d)/(\d))?", flag)
            if bg_match:
                if bg_match.group(3) is not None:
                    tx, ty = int(bg_match.group(3)), int(bg_match.group(4))
                    if not (0 <= tx <= 7 and 0 <= ty <= 5):
                        return await ctx.error("The provided background color is invalid.")
                    background = tx, ty
                else:
                    background = (0, 4)
                to_delete.append((x, y))
                continue
            flag_match = re.fullmatch(r"(--palette=|-p=|palette:)(\w+)", flag)
            if flag_match:
                palette = flag_match.group(2)
                if palette + ".png" not in listdir("data/palettes"):
                    return await ctx.error(f"Could not find a palette with name \"{palette}\".")
                to_delete.append((x, y))
            raw_match = re.fullmatch(r"--raw|-r", flag)
            if raw_match:
                raw_output = True
                to_delete.append((x, y))
            letter_match = re.fullmatch(r"--letter|-l", flag)
            if letter_match:
                default_to_letters = True
                to_delete.append((x, y))
            frames_match = re.fullmatch(r"-frames=\[([1,2,3]),?([1,2,3])?,?([1,2,3])?\]", flag)
            if frames_match and frames_match.group(1):
                frames = [int(frames_match.group(1))]
                if frames_match.group(2):
                    frames.append(int(frames_match.group(2)))
                if frames_match.group(3):
                    frames.append(int(frames_match.group(3)))
                to_delete.append((x, y))
            combine_match = re.fullmatch(r"-combine", flag)
            if combine_match:
                async for m in ctx.channel.history(limit=100):
                    print("iterating")
                    if m.attachments:
                        before_image = Image.open(requests.get(m.attachments[0].url, stream=True).raw)
                        break
                to_delete.append((x, y))
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

        # Splits "&"-joined words into stacks
        stacked_grid: list[list[list[str]]] = []
        for row in comma_grid:
            stacked_row: list[list[str]] = []
            for stack in row:
                split = stack.split("&")
                stacked_row.append(split)
                #if len(split) > constants.MAX_STACK and ctx.author.id != self.bot.owner_id:
                #    return await ctx.error(f"Stack too high ({len(split)}).\nYou may only stack up to {constants.MAX_STACK} tiles on one space.")
            stacked_grid.append(stacked_row)


        # Get the dimensions of the grid
        width = max(len(row) for row in stacked_grid)
        height = len(stacked_grid)

        # Don't proceed if the request is too large.
        # (It shouldn't be that long to begin with because of Discord's 2000 character limit)
        area = width * height
        if area > constants.MAX_TILES and ctx.author.id != self.bot.owner_id:
            return await ctx.error(f"Too many tiles ({area}). You may only render up to {constants.MAX_TILES} tiles at once, including empty tiles.")
        elif area == 0:
            return await ctx.error(f"Can't render nothing.")

        # Pad the word rows from the end to fit the dimensions
        for row in stacked_grid:
            row.extend([["-"]] * (width - len(row)))
        try:
            grid = self.parse_raw(stacked_grid, rule=rule)
            # Handles variants based on `:` affixes
            buffer = BytesIO()
            extra_buffer = BytesIO() if raw_output else None
            extra_names = [] if raw_output else None
            full_grid = await self.bot.handlers.handle_grid(grid, raw_output=raw_output, extra_names=extra_names, default_to_letters=default_to_letters)
            try:
                before_image=before_image
            except:
                before_image=None
            await self.bot.renderer.render(
                await self.bot.renderer.render_full_tiles(
                    full_grid,
                    palette=palette,
                    random_animations=True
                ),
                before_image=before_image,
                palette=palette,
                background=background, 
                out=buffer,
                upscale=not raw_output,
                extra_out=extra_buffer,
                extra_name=extra_names[0] if raw_output else None, # type: ignore
                frames=frames
            )
        except errors.TileNotFound as e:
            word = e.args[0]
            if word.name.startswith("tile_") and await self.bot.db.tile(word.name[5:]) is not None:
                return await ctx.error(f"The tile `{word}` could not be found. Perhaps you meant `{word.name[5:]}`?")
            if await self.bot.db.tile("text_" + word.name) is not None:
                return await ctx.error(f"The tile `{word}` could not be found. Perhaps you meant `{'text_' + word.name}`?")
            return await ctx.error(f"The tile `{word}` could not be found.")
        except errors.BadTileProperty as e:
            word, (w, h) = e.args
            return await ctx.error(f"The tile `{word}` could not be made into a property, because it's too big (`{w} by {h}`).")
        except errors.EmptyVariant as e:
            word = e.args[0]
            return await ctx.error(
                f"You provided an empty variant for `{word}`."
            )
        except errors.VariantError as e:
            return await self.handle_variant_errors(ctx, e)
        except errors.TextGenerationError as e:
            return await self.handle_custom_text_errors(ctx, e)

        filename = datetime.utcnow().strftime(r"render_%Y-%m-%d_%H.%M.%S.gif")
        delta = time() - start
        msg = f"`{ctx.message.content}`\n*Rendered in {delta:.2f} s*"
        await ctx.reply(content=msg, file=discord.File(buffer, filename=filename, spoiler=spoiler))
        if extra_buffer is not None and extra_names is not None:
            extra_buffer.seek(0)
            await ctx.send("*Raw files:*", file=discord.File(extra_buffer, filename=f"{extra_names[0]}_raw.zip"))
        

    @commands.command(aliases=["text"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def rule(self, ctx: Context, *, objects: str = ""):
        '''Renders the text tiles provided. 
        
        If not found, the bot tries to auto-generate them! (See the `make` command for more.)

        **Flags**
        * `--palette=<...>` (`-P=<...>`): Recolors the output gif. See `search type:palettes` command for palettes.
        * `--background=[...]` (`-B=[...]`): Enables background color. If no argument is given, defaults to black. The argument must be a palette index ("x/y").
        * `--raw` (`-R`): Enables raw mode. The sprites are sent in a ZIP file as well as normally. By default, sprites have no color.
        * `--letter` (`-L`): Enables letter mode. Custom text that has 2 letters in it will be rendered in "letter" mode.
        
        **Variants**
        * `:variant`: Append `:variant` to a tile to change color or sprite of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile. 
        * `&` : Stacks tiles on top of each other.
        * `tile_` : `tile_object` renders regular objects.
        * `,` : `tile_x,y,...` is expanded into `tile_x tile_y ...`
        * `||` : Marks the output gif as a spoiler. 
        
        **Example commands:**
        `rule baba is you`
        `rule -B rock is ||push||`
        `rule -P=test tile_baba on baba is word`
        `rule baba eat baba - tile_baba tile_baba:l`
        '''
        await self.render_tiles(ctx, objects=objects, rule=True)

    # Generates an animated gif of the tiles provided, using the default palette
    @commands.command()
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def tile(self, ctx: Context, *, objects: str = ""):
        '''Renders the tiles provided.

       **Flags**
        * `--palette=<...>` (`-P=<...>`): Recolors the output gif. See `search type:palettes` command for palettes.
        * `--background=[...]` (`-B=[...]`): Enables background color. If no argument is given, defaults to black. The argument must be a palette index ("x/y").
        * `--raw` (`-R`): Enables raw mode. The sprites are sent in a ZIP file as well as normally. By default, sprites have no color.
        * `--letter` (`-L`): Enables letter mode. Custom text that has 2 letters in it will be rendered in "letter" mode.

        **Variants**
        * `:variant`: Append `:variant` to a tile to change color or sprite of a tile. See the `variants` command for more.

        **Useful tips:**
        * `-` : Shortcut for an empty tile. 
        * `&` : Stacks tiles on top of each other.
        * `text_` : `text_object` renders text objects.
        * `,` : `text_x,y,...` is expanded into `text_x text_y...`
        * `||` : Marks the output gif as a spoiler. 
        
        **Example commands:**
        `tile baba - keke`
        `tile --palette=marshmallow keke:d baba:s`
        `tile text_baba,is,you`
        `tile baba&flag ||cake||`
        `tile -P=mountain -B baba bird:l`
        '''
        await self.render_tiles(ctx, objects=objects, rule=False)

    async def search_levels(self, query: str, **flags: Any) -> OrderedDict[tuple[str, str], LevelData]:
        '''Finds levels by query.
        
        Flags:
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        '''
        levels: OrderedDict[tuple[str, str], LevelData] = collections.OrderedDict()
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
                    dict(world=parts[0], id=parts[1], f_map=f_map, f_world=f_world)
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
                    dict(id=query, f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
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
                        dict(parent=segments[0], map_id=segments[1], f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
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
                    dict(name=query, f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
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
                    dict(name=query, f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
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
                    dict(map=query, f_map=f_map, f_world=f_world, default=constants.BABA_WORLD)
                )
                for row in await cur.fetchall():
                    data = LevelData.from_row(row)
                    levels[data.world, data.id] = data
        
        return levels

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @commands.group(name="level", invoke_without_command=True)
    async def level_command(self, ctx: Context, *, query: str):
        '''Renders the Baba Is You level from a search term.

        Levels are searched for in the following order:
        * Custom level code (e.g. "1234-ABCD")
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        '''
        await self.perform_level_command(ctx, query, mobile=False)

    @commands.cooldown(5, 8, commands.BucketType.channel)
    @level_command.command()
    async def mobile(self, ctx: Context, *, query: str):
        '''Renders the mobile Baba Is You level from a search term.

        Levels are searched for in the following order:
        * World & level ID (e.g. "baba/20level")
        * Level ID (e.g. "16level")
        * Level number (e.g. "space-3" or "lake-extra 1")
        * Level name (e.g. "further fields")
        * The map ID of a world (e.g. "cavern", or "lake")
        '''
        await self.perform_level_command(ctx, query, mobile=True)
    
    async def perform_level_command(self, ctx: Context, query: str, *, mobile: bool):
        # User feedback
        await ctx.trigger_typing()

        custom_level: CustomLevelData | None = None
        
        spoiler = query.count("||") >= 2
        fine_query = query.lower().strip().replace("|", "")
        
        # [abcd-0123]
        if re.match(r"^[a-z0-9]{4}\-[a-z0-9]{4}$", fine_query) and not mobile:
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
                await ctx.reply("Searching for custom level... this might take a while", mention_author=False)
                await ctx.trigger_typing()
                async with aiohttp.request("GET", f"https://baba-is-bookmark.herokuapp.com/api/level/exists?code={fine_query.upper()}") as resp:
                    if resp.status in (200, 304):
                        data = await resp.json()
                        if data["data"]["exists"]:
                            try:
                                custom_level = await self.bot.get_cog("Reader").render_custom_level(fine_query)
                            except ValueError as e:
                                size = e.args[0]
                                return await ctx.error(f"The level code is valid, but the level's width, height or area is way too big ({size})!")
                            except aiohttp.ClientResponseError as e:
                                return await ctx.error(f"The Baba Is Bookmark site returned a bad response. Try again later.")
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
                f"Name: ||`{display}`||" if spoiler else f"Name: `{display}`",
                f"ID: `{path}`",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: `{level.subtitle}`"
                )
            mobile_exists = os.path.exists(f"target/renders/{level.world}_m/{level.id}.gif")
            
            if not mobile and mobile_exists:
                rows.append(
                    f"*This level is also on mobile, see `+level mobile {level.unique()}`*"
                )
            elif mobile and mobile_exists:
                rows.append(
                    f"*This is the mobile version. For others, see `+level {level.unique()}`*"
                )

            if mobile and mobile_exists:
                gif = discord.File(f"target/renders/{level.world}_m/{level.id}.gif", spoiler=True)
            elif mobile and not mobile_exists:
                rows.append("*This level doesn't have a mobile version. Using the normal gif instead...*")
                gif = discord.File(f"target/renders/{level.world}/{level.id}.gif", spoiler=True)
            else:
                gif = discord.File(f"target/renders/{level.world}/{level.id}.gif", spoiler=True)
        else:
            gif = discord.File(f"target/renders/levels/{level.code}.gif", spoiler=True)
            path = level.unique()
            display = level.name
            rows = [
                f"Name: ||`{display}`|| (by `{level.author}`)" 
                    if spoiler else f"Name: `{display}` (by `{level.author}`)",
                f"Level code: `{path}`",
            ]
            if level.subtitle:
                rows.append(
                    f"Subtitle: `{level.subtitle}`"
                )

        if len(levels) > 0:
            extras = [level.unique() for level in levels.values()]
            if len(levels) > constants.OTHER_LEVELS_CUTOFF:
                extras = extras[:constants.OTHER_LEVELS_CUTOFF]
            paths = ", ".join(f"`{extra}`" for extra in extras)
            plural = "result" if len(extras) == 1 else "results"
            suffix = ", `...`" if len(levels) > constants.OTHER_LEVELS_CUTOFF else ""
            rows.append(
                f"*Found {len(levels)} other {plural}: {paths}{suffix}*"
            )

        formatted = "\n".join(rows)

        # Only the author should be mentioned
        mentions = discord.AllowedMentions(everyone=False, users=[ctx.author], roles=False)

        # Send the result
        await ctx.reply(formatted, file=gif, allowed_mentions=mentions)

def setup(bot: Bot):
    bot.add_cog(GlobalCog(bot))
