from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import re
from os import listdir
import os.path
from typing import Any, Sequence, Optional

from src.db import CustomLevelData, LevelData, TileData
import zipfile

import glob
import discord
from discord.ext import commands, menus
from discord.ext.menus.views import ViewMenuPages
from PIL import Image, ImageFont, ImageDraw

import src.types

from . import flags
from .. import constants
from ..tile import Tile
from ..types import Bot, Context, Variant, Color
from ..utils import ButtonPages


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
        out.set_footer(text="Note: To search for things other than tiles, use command flags. See =help search.")
        lines = ["```"]
        for (type, short), long in entries:
            if isinstance(long, TileData):
                lines.append(
                    f"({type}) {short}\n    sprite: {long.sprite} source: {long.source}\n")
                lines.append(
                    f"    color: {long.inactive_color} active color: {long.active_color}\n    tiling: {src.types.TilingMode(long.tiling)}")
                if len(long.tags) > 0:
                    lines.append(f"\n    tags: {', '.join(long.tags)}")
                if len(long.extra_frames) > 0:
                    lines.append(f"\n    extra_frames: {', '.join(str(n) for n in long.extra_frames)}")
            elif isinstance(long, LevelData):
                lines.append(f"({type}) {short} {long.display()}")
            elif isinstance(long, CustomLevelData):
                lines.append(
                    f"({type}) {short} {long.name} (by {long.author})")
            elif long is None:
                continue
            else:
                lines.append(f"({type}) {short}")
            lines.append("\n")

        if len(lines) > 1:
            lines[-1] = "```"
            out.description = "".join(lines)
        else:
            out.title = f"No results found{target}"
        return out


class FlagPageSource(menus.ListPageSource):
    def __init__(
            self, data: Sequence[flags.Flag]):
        super().__init__(data, per_page=7)

    async def format_page(self, menu: menus.Menu, entries: Sequence[flags.Flag]) -> discord.Embed:
        embed = discord.Embed(
            color=menu.bot.embed_color,
            title=None,
        )
        embed.description = '\n'.join([str(entry) for entry in entries])
        embed.set_footer(text="Page " + str(menu.current_page +
                                            1) + "/" + str(self.get_max_pages()))
        return embed


type_format = {
    "sprite": "sprite augmentation",
    "tile": "tile creation",
    "post": "compositing",
    "sign": "sign text parsing"
}

class VariantSource(menus.ListPageSource):
    def __init__(
            self, data: list[Variant]):
        super().__init__(data, per_page=3)

    async def format_page(self, menu: menus.Menu, entries: list[Variant]) -> discord.Embed:
        embed = discord.Embed(
            title=f"{menu.current_page+1}/{self.get_max_pages()}",
            color=menu.bot.embed_color
        )
        embed.description = "```ansi"
        for entry in entries:
            embed.description += f"""
\u001b[1;4;37m{entry.__name__.removesuffix("Variant")}\u001b[0;30m - \u001b[0;37m{entry.__doc__}\u001b[0m
\u001b[0;30m- \u001b[0;34mSyntax: {entry.syntax}
\u001b[0;30m- \u001b[0;34mApplied during \u001b[1;36m{type_format[entry.type]}\u001b[0m
"""
        embed.description += "```"
        return embed


class UtilityCommandsCog(commands.Cog, name="Utility Commands"):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="undo")
    async def undo(self, ctx: Context):
        """Deletes the last message sent from the bot."""
        await ctx.typing()
        h = ctx.channel.history(limit=20)
        async for m in h:
            if m.author.id == self.bot.user.id and m.attachments:
                try:
                    reply = await ctx.channel.fetch_message(m.reference.message_id)
                    if reply.author == ctx.message.author:
                        await m.delete()
                        await ctx.send('Removed message.', delete_after=3.0)
                        return
                except BaseException:
                    pass
        await ctx.error('None of your commands were found in the last `20` messages.')

    @commands.command()
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def flags(self, ctx: Context):
        """Shows a list of render flags."""
        flags = self.bot.flags.list
        await ButtonPages(
            source=FlagPageSource(
                flags
            ),
        ).start(ctx)

    @commands.command()
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def search(self, ctx: Context, *, query: str = ""):
        """Searches through bot data based on a query.

        This can return tiles, levels, palettes, variants, and sprite mods.

        **Tiles** can be filtered with the flags, formatted like `--<name>=<value>`:
        * `sprite`: Will return only tiles that use that sprite.
        * `text`: Whether to only return text tiles (either `true` or `false`).
        * `source`: The source of the sprite. This should be a sprite mod.
        * `modded`: Whether to only return modded tiles (either `true` or `false`).
        * `color`: The color of the sprite. This can be a color name (`red`) or a palette (`0/3`).
        * `tiling`: The tiling type of the object.
        * `tag`: A tile tag, e.g. `animal` or `common`.

        **Levels** can be filtered with the flags:
        * `custom`: Whether to only return custom levels (either `true` or `false`).
        * `map`: Which map screen the level is from.
        * `world`: Which levelpack / world the level is from.
        * `author`: For custom levels, filters by the author.

        You can also filter by the result type:
        * `type`: What results to return. This can be `tile`, `level`, `palette`, `variant`, `world`, or `mod`.
        """
        # Pattern to match flags in the format (flag)=(value)
        flag_pattern = r"--([\d\w_/]+)=([\d\w\-_/]+)"
        match = re.search(flag_pattern, query)
        plain_query = query.lower()

        # Whether to use simple string matching
        has_flags = bool(match)

        # Determine which flags to filter with
        flags = {}
        if has_flags:
            if match:
                # Returns "flag"="value" pairs
                flags = dict(re.findall(flag_pattern, query))
            # Nasty regex to match words that are not flags
            plain_query = re.sub(flag_pattern, "", plain_query).strip()

        if "type" not in flags:
            flags["type"] = "tile"

        results: dict[tuple[str, str], Any] = {}

        if flags.get("type") == "tile":
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
            tiling = flags.get("tiling")
            if tiling is not None:
                tiling = +src.types.TilingMode.parse(tiling)
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
                        WHEN "false" THEN (source == 'vanilla' OR source == 'baba' OR source == 'new_adv' OR source == 'museum')
                        WHEN "true" THEN (source != 'vanilla' AND source != 'baba' AND source != 'new_adv' AND source != 'museum')
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
                    f_tiling=tiling,
                    f_tag=flags.get("tag")
                )
            )
            for row in rows:
                results["tile", row["name"]] = TileData.from_row(row)
                results["blank_space", row["name"]] = None

        if flags.get("type") == "level":
            if flags.get("custom") is None or flags.get("custom") == "true":
                f_author = flags.get("author")
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

            if flags.get("custom") is None or not flags.get(
                    "custom") == "false":
                levels = await self.bot.get_cog("Baba Is You").search_levels(plain_query, **flags)
                for (world, id), data in levels.items():
                    results["level", f"{world}/{id}"] = data

        if flags.get("type") == "palette":
            q = f"*{plain_query}*.png" if plain_query else "*.png"
            out = []
            for path in Path("data/palettes").glob(q):
                out.append(
                    (("palette", path.parts[-1][:-4]), path.parts[-1][:-4]))
            out.sort()
            for a, b in out:
                results[a] = b

        if flags.get("type") == "mod":
            q = f"*{plain_query}*.toml" if plain_query else "*.toml"
            out = []
            for path in Path("data/custom").glob(q):
                out.append((("mod", path.parts[-1][:-5]), path.parts[-1][:-5]))
            out.sort()
            for a, b in out:
                results[a] = b

        if flags.get("type") == "world":
            out = []
            for path in Path("data/levels").glob(plain_query if plain_query else "*"):
                out.append((("world", path.stem), path.stem))
            out.sort()
            for a, b in out:
                results[a] = b

        await ButtonPages(
            source=SearchPageSource(
                list(results.items()),
                plain_query
            ),
        ).start(ctx)

    @commands.command(name="variants", aliases=["var", "vars", "variant"])
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def variants(self, ctx: Context, query: Optional[str] = None):
        """Shows all the bot's variants."""
        def sort(variant):
            return variant.__name__
        variants = ctx.bot.variants._values
        if query is not None:
            variants = [var for var in variants if (query in var.__name__.lower() or query in var.__doc__) and not var.hidden]
        assert len(variants) > 0, f"No variants were found with the query `{query}`!"
        await ButtonPages(
            source=VariantSource(
                sorted(variants, key=sort)  # Sort alphabetically
            ),
        ).start(ctx)


    @commands.command()
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def grab(self, ctx: Context, name: str):
        """Gets the files for a specific tile from the bot."""
        async with self.bot.db.conn.cursor() as cur:
            await ctx.typing()
            result = await cur.execute(
                'SELECT DISTINCT sprite, source FROM tiles WHERE name = (?)',
                name
            )
            try:
                sprite_name, source = tuple(await result.fetchone())
            except BaseException:
                return await ctx.error(f'Tile `{name.replace("`", "")[:16]}` not found!')
            files = glob.glob(f'data/sprites/{source}/{sprite_name}_*.png')
            zipped_files = BytesIO()
            with zipfile.ZipFile(zipped_files, "x") as zip_file:
                for data_filename in files:
                    with open(data_filename, 'rb') as data_file:
                        zip_file.writestr(
                            os.path.basename(
                                os.path.normpath(data_filename)),
                            data_file.read())
            zipped_files.seek(0)
            return await ctx.send(f'Files for sprite `{name}` from `{source}`:',
                                  files=[discord.File(zipped_files, filename=f'{source}-{name}.zip')])

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="overlays")
    async def overlays(self, ctx: Context):
        """Lists all available overlays."""
        await ctx.send(embed=discord.Embed(
            title="Available overlays",
            colour=self.bot.embed_color,
            description="\n".join(f"{overlay[:-4]}" for overlay in listdir('data/overlays/'))))

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command()
    async def tiles(self, ctx: Context, source: str = None):
        """Sends a text file containing all tiles in a given source, or all tiles if no source is given."""
        result = None
        async with self.bot.db.conn.cursor() as cur:
            if source is None:
                result = await cur.execute("select distinct name, sprite, tiling, active_color_x, active_color_y from tiles")
            else:
                result = await cur.execute("select distinct name, sprite, tiling, active_color_x, active_color_y from tiles where source = ?", source)
            data_rows = await result.fetchall()
        buf = StringIO()
        seen_names = set()
        for (name, sprite, tiling, col_x, col_y) in data_rows:
            if name in seen_names: continue
            seen_names.add(name)
            if not re.fullmatch(r"^[A-Za-z0-9_]+$", name):
                escaped_name = name.replace("\"", "\\\"")
                name = f'"{escaped_name}"'
            buf.write(f"[{name}]\nsprite = \"{sprite}\"\ntiling = {tiling}\ncolor = [{col_x}, {col_y}]\n\n")
        buf.seek(0)
        await ctx.send(file=discord.File(buf, filename="tiles.toml"))

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="palette", aliases=['pal'])
    async def show_palette(self, ctx: Context, palette: str = 'default', color: str = None):
        """Displays palette image, or details about a palette index.

        This is useful for picking colors from the palette.
        """
        p_cache = self.bot.renderer.palette_cache
        img = p_cache.get(palette, None)
        if img is None:
            if "/" not in palette:
                return await ctx.error(f'The palette `{palette.replace("`", "")[:16]}` could not be found.')
            palette, color = "default", palette
        if color is not None:
            r, g, b, _ = Color.parse(Tile(name="<palette command>", palette=palette), p_cache, color)
            d = discord.Embed(
                color=discord.Color.from_rgb(r, g, b),
                title=f"Color: #{hex((r << 16) | (g << 8) | b)[2:].zfill(6)}"
            )
            return await ctx.reply(embed=d)
        else:
            txtwid, txthgt = img.size
            pal_img = img.resize(
                (img.width * constants.PALETTE_PIXEL_SIZE,
                 img.height * constants.PALETTE_PIXEL_SIZE),
                resample=Image.NEAREST
            ).convert("RGBA")
            font = ImageFont.truetype("data/fonts/04b03.ttf", 16)
            draw = ImageDraw.Draw(pal_img)
            for y in range(txthgt):
                for x in range(txtwid):
                    n = pal_img.getpixel(
                        (x * constants.PALETTE_PIXEL_SIZE,
                         (y * constants.PALETTE_PIXEL_SIZE)))
                    if (n[0] + n[1] + n[2]) / 3 > 128:
                        draw.text(
                            (x * constants.PALETTE_PIXEL_SIZE,
                             (y * constants.PALETTE_PIXEL_SIZE) - 2),
                            f"{x},{y}",
                            (1, 1, 1, 255),
                            font,
                            layout_engine=ImageFont.LAYOUT_BASIC)
                    else:
                        draw.text(
                            (x * constants.PALETTE_PIXEL_SIZE,
                             (y * constants.PALETTE_PIXEL_SIZE) - 2),
                            f"{x},{y}",
                            (255, 255, 255, 255),
                            font,
                            layout_engine=ImageFont.LAYOUT_BASIC)
            buf = BytesIO()
            pal_img.save(buf, format="PNG")
            buf.seek(0)
            file = discord.File(buf, filename=f"{palette[:16]}.png")
            await ctx.reply(f"Palette `{palette[:16]}`:", file=file)

    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    @commands.command(name="hint", aliases=["hints"])
    async def show_hint(self, ctx: Context):
        """Shows hints for a level."""
        return await ctx.send("""The =hint command has been deprecated. Look at [Baba Is Hint](https://www.keyofw.com/baba-is-hint/) for an updated record of hints.""")


async def setup(bot: Bot):
    await bot.add_cog(UtilityCommandsCog(bot))
