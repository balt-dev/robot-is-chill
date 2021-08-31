from __future__ import annotations
from io import BytesIO
from pathlib import Path

import re
from os import listdir
from typing import Any, Sequence

from src.tile import RawTile
from src.db import CustomLevelData, LevelData, TileData

import discord
from discord.ext import commands, menus
from PIL import Image

from .. import constants
from ..types import Bot, Context

class SearchPageSource(menus.ListPageSource):
    def __init__(self, data: Sequence[Any], query: str):
        self.query = query
        super().__init__(data, per_page=constants.SEARCH_RESULT_UNITS_PER_PAGE)
    
    async def format_page(self, menu: menus.Menu, entries: Sequence[Any]) -> discord.Embed:
        target = f" for `{self.query}`" if self.query else ""
        out = discord.Embed(
            color=menu.bot.embed_color,
            title=f"Search results{target} (Page {menu.current_page + 1}/{self.get_max_pages()})"
        )
        out.set_footer(text="Note: Some custom levels may not show up here.")
        lines = ["```"]
        for (type, short), long in entries:
            if isinstance(long, TileData):
                lines.append(f"({type}) {short} sprite: {long.sprite} source: {long.source}\n")
                lines.append(f"    color: {long.inactive_color}, active color: {long.active_color} tiling: {long.tiling}\n")
                lines.append(f"    tags: {','.join(long.tags)}")
            elif isinstance(long, LevelData):
                lines.append(f"({type}) {short} {long.display()}")
            elif isinstance(long, CustomLevelData):
                lines.append(f"({type}) {short} {long.name} (by {long.author})")
            elif long is None:
                continue
            else:
                lines.append(f"({type}) {short}")
            lines.append("\n\n")
        
        if len(lines) > 1:
            lines[-1] = "```"
            out.description="".join(lines)
        else:
            out.title = f"No results found{target}"
        return out

class HintPageSource(menus.ListPageSource):
    def __init__(self, data: Sequence[tuple[str, dict[str, str]]], level: LevelData, others: int):
        self.level = level
        self.others = others
        super().__init__(data, per_page=1)
    
    async def format_page(self, menu: menus.Menu, entries: tuple[str, dict[str, str]]) -> discord.Embed:
        group, hints = entries
        embed = discord.Embed(
            color=menu.bot.embed_color,
            title=f"Hints for `{self.level.display()}` -- `{group}` ({menu.current_page + 1}/{self.get_max_pages()} endings)",
        )
        if self.others > 0:
            embed.set_footer(text=f"Found {self.others} other levels. Please change your search term if you meant any of those.")
        
        rows = ["*Click on the spoilers to view each hint*"]
        for kind, hint in hints.items():
            rows.append(f"__{kind}__: ||{hint}||")
        
        embed.description = "\n\n".join(rows)
        return embed

class UtilityCommandsCog(commands.Cog, name="Utility Commands"):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.command()
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def search(self, ctx: Context, *, query: str):
        '''Searches through bot data based on a query.

        This can return tiles, levels, palettes, variants, and sprite mods.

        **Tiles** can be filtered with the flags:
        * `sprite`: Will return only tiles that use that sprite.
        * `text`: Whether to only return text tiles (either `true` or `false`).
        * `source`: The source of the sprite. This should be a sprite mod.
        * `modded`: Whether to only return modded tiles (either `true` or `false`).
        * `color`: The color of the sprite. This can be a color name (`red`) or a palette (`0/3`).
        * `tiling`: The tiling type of the object. This must be one of `-1`, `0`, `1`, `2`, `3` or `4`.
        * `tag`: A tile tag, e.g. `animal` or `common`.

        **Levels** can be filtered with the flags:
        * `custom`: Whether to only return custom levels (either `true` or `false`).
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        * `author`: For custom levels, filters by the author.

        You can also filter by the result type:
        * `type`: What results to return. This can be `tile`, `level`, `palette`, `variant`, or `mod`.

        **Example commands:**
        `search baba`
        `search text:false source:vanilla sta`
        `search source:modded sort:color page:4`
        `search text:true color:0,3 reverse:true`
        '''
        # Pattern to match flags in the format (flag):(value)
        flag_pattern = r"([\d\w_/]+):([\d\w\-_/]+)"
        match = re.search(flag_pattern, query)
        plain_query = query.lower()

        # Whether or not to use simple string matching
        has_flags = bool(match)
        
        # Determine which flags to filter with
        flags = {}
        if has_flags:
            if match:
                flags = dict(re.findall(flag_pattern, query)) # Returns "flag":"value" pairs
            # Nasty regex to match words that are not flags
            non_flag_pattern = r"(?<![:\w\d,\-/])([\w\d,_/]+)(?![:\d\w,\-/])"
            plain_match = re.findall(non_flag_pattern, query)
            plain_query = " ".join(plain_match)
        
        results: dict[tuple[str, str], Any] = {}

        if flags.get("type") is None or flags.get("type") == "tile":
            if plain_query.strip() or any(x in flags for x in ("sprite", "text", "source", "modded", "color", "tiling", "tag")):
                color = flags.get("color")
                f_color_x = f_color_y = None
                if color is not None:
                    match = re.match(r"(\d)/(\d)", color)
                    if match is None:
                        z = constants.COLOR_NAMES.get("color")
                        if z is not None:
                            f_color_x, f_color_y = z
                    else:
                        f_color_x = int(match.group(1))
                        f_color_y = int(match.group(2))
                rows = await self.bot.db.conn.fetchall(
                    f'''
                    SELECT * FROM tiles 
                    WHERE name LIKE "%" || :name || "%" AND (
                        CASE :f_text
                            WHEN NULL THEN 1
                            WHEN "false" THEN (name NOT LIKE "text_%")
                            WHEN "true" THEN (name LIKE "text_%")
                            ELSE 1
                        END
                    ) AND (
                        :f_source IS NULL OR source == :f_source
                    ) AND (
                        CASE :f_modded 
                            WHEN NULL THEN 1
                            WHEN "false" THEN (source == {repr(constants.BABA_WORLD)})
                            WHEN "true" THEN (source != {repr(constants.BABA_WORLD)})
                            ELSE 1
                        END
                    ) AND (
                        :f_color_x IS NULL AND :f_color_y IS NULL OR (
                            (
                                inactive_color_x == :f_color_x AND
                                inactive_color_y == :f_color_y
                            ) OR (
                                active_color_x == :f_color_x AND
                                active_color_y == :f_color_y
                            )
                        )
                    ) AND (
                        :f_tiling IS NULL OR CAST(tiling AS TEXT) == :f_tiling
                    ) AND (
                        :f_tag IS NULL OR INSTR(tags, :f_tag)
                    )
                    ORDER BY name, version ASC;
                    ''',
                    dict(
                        name=plain_query,
                        f_text=flags.get("text"),
                        f_source=flags.get("source"),
                        f_modded=flags.get("modded"),
                        f_color_x=f_color_x,
                        f_color_y=f_color_y,
                        f_tiling=flags.get("tiling"),
                        f_tag=flags.get("tag")
                    )
                )
                for row in rows:
                    results["tile", row["name"]] = TileData.from_row(row)
                    results["blank_space", row["name"]] = None

        if flags.get("type") is None or flags.get("type") == "level":
            if flags.get("custom") is None or flags.get("custom") == "true":
                f_author=flags.get("author")
                async with self.bot.db.conn.cursor() as cur:
                    if plain_query.strip():
                        await cur.execute(
                            '''
                            SELECT * FROM custom_levels 
                            WHERE code == :code AND (
                                :f_author IS NULL OR author == :f_author 
                            );
                            ''',
                            dict(code=plain_query, f_author=f_author)
                        )
                        row = await cur.fetchone()
                        if row is not None:
                            custom_data = CustomLevelData.from_row(row)
                            results["level", custom_data.code] = custom_data
                        await cur.execute(
                            '''
                            SELECT * FROM custom_levels
                            WHERE INSTR(LOWER(name), :name) AND (
                                :f_author IS NULL OR author == :f_author
                            )
                            ''',
                            dict(name=plain_query, f_author=f_author)
                        )
                        for row in await cur.fetchall():
                            custom_data = CustomLevelData.from_row(row)
                            results["level", custom_data.code] = custom_data
                    if any(x in flags for x in ("author", "custom")):
                        await cur.execute(
                            '''
                            SELECT * FROM custom_levels
                            WHERE (
                                :f_author IS NULL OR author == :f_author
                            )
                            ''',
                            dict(name=plain_query, f_author=f_author)
                        )
                        for row in await cur.fetchall():
                            custom_data = CustomLevelData.from_row(row)
                            results["level", custom_data.code] = custom_data
                    
            if flags.get("custom") is None or not flags.get("custom") == "false":
                if plain_query.strip() or any(x in flags for x in ("map", "world")):
                    levels = await self.bot.get_cog("Baba Is You").search_levels(plain_query, **flags)
                    for (world, id), data in levels.items():
                        results["level", f"{world}/{id}"] = data
        
        if flags.get("type") is None and plain_query or flags.get("type") == "palette":
            q = f"*{plain_query}*.png" if plain_query else "*.png"
            out = []
            for path in Path("data/palettes").glob(q):
                out.append((("palette", path.parts[-1][:-4]), path.parts[-1][:-4]))
            out.sort()
            for a, b in out:
                results[a] = b
        
        if flags.get("type") is None and plain_query or flags.get("type") == "mod":
            q = f"*{plain_query}*.json" if plain_query else "*.json"
            out = []
            for path in Path("data/custom").glob(q):
                out.append((("mod", path.parts[-1][:-5]), path.parts[-1][:-5]))
            out.sort()
            for a, b in out:
                results[a] = b

        if flags.get("type") is None and plain_query or flags.get("type") == "variant":
            for variant in self.bot.handlers.all_variants():
                if plain_query.lower() in variant.lower():
                    results["variant", variant] = variant

        await menus.MenuPages(
            source=SearchPageSource(
                list(results.items()),
                plain_query
            ),
            clear_reactions_after=True
        ).start(ctx)
        
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="palette")
    async def show_palette(self, ctx: Context, palette: str):
        '''Displays palette image.

        This is useful for picking colors from the palette.'''
        try:
            img = Image.open(f"data/palettes/{palette}.png")
        except FileNotFoundError:
            return await ctx.error(f"The palette `{palette}` could not be found.")
        scaled = img.resize(
            (img.width * constants.PALETTE_PIXEL_SIZE, img.height * constants.PALETTE_PIXEL_SIZE), 
            resample=Image.NEAREST
        )
        buf = BytesIO()
        scaled.save(buf, format="PNG")
        buf.seek(0)
        file = discord.File(buf, filename=f"palette_{palette}.png")
        await ctx.reply(f"Palette `{palette}`:", file=file)

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="hint", aliases=["hints"])
    async def show_hint(self, ctx: Context, *, level_query: str):
        '''Shows hints for a level.'''
        levels = await self.bot.get_cog("Baba Is You").search_levels(level_query)
        if len(levels) == 0:
            return await ctx.error(f"No levels found with the query `{level_query}`.")
        _, choice = levels.popitem(last=False)
        choice: LevelData
        
        hints = self.bot.db.hints(choice.id)
        if hints is None:
            if len(levels) > 0:
                return await ctx.error(
                    f"No hints found for `{choice.display()}`. "
                    "Please narrow your search if you meant a different level."
                )
            return await ctx.error(f"No hints found for `{choice.display()}`.")
        
        await menus.MenuPages(
            source=HintPageSource(
                list(hints.items()),
                choice,
                len(levels)
            ),
            clear_reactions_after=True
        ).start(ctx)

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="variants")
    async def list_variants(self, ctx: Context, tile: str):
        '''List valid sprite variants for a given tile.'''
        # Clean the input
        clean_tile = tile.strip().lower()

        
        output = discord.Embed(
            title=f"Valid sprite variants for `{clean_tile}`",
            color=self.bot.embed_color
        )

        tile_data_cache: dict[str, TileData] = {}
        data = await self.bot.db.tile(clean_tile)
        if data is not None:
            tile_data_cache[clean_tile] = data
        else:
            output.set_footer(text="Note: This tile doesn't exist in the database, so it's not necessarily valid.")
        
        raw_tile = RawTile(clean_tile, [])
        variant_groups = self.bot.handlers.valid_variants(raw_tile, tile_data_cache)
        for group, variants in variant_groups.items():
            output.add_field(
                name=group,
                value="\n".join(f"- {string}" for string in variants),
                inline=True
            )
        
        await ctx.reply(embed=output)

def setup(bot: Bot):
    bot.add_cog(UtilityCommandsCog(bot))
