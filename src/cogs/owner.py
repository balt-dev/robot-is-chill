from __future__ import annotations

import shutil
from glob import glob
from io import BytesIO
import time
import typing
import zipfile
import pathlib
import re
import json
import tomlkit
import urllib
from pathlib import Path

import requests
import itertools
import collections

import tomlkit.exceptions
from src import constants
from src.types import TilingMode
from typing import Any, Optional
import os
import numpy as np
import subprocess
import asyncio

import discord
from discord.ext import commands
from PIL import Image, ImageChops, ImageDraw

from ..db import TileData
from ..types import Bot, Context


class OwnerCog(commands.Cog, name="Admin", command_attrs=dict(hidden=True)):
    def __init__(self, bot: Bot):
        self.bot = bot
        self.identifies = []
        self.resumes = []
        # Are assets loading?
        self.bot.loading = False

    @commands.command()
    @commands.is_owner()
    async def danger(self, ctx: Context):
        """Toggles danger mode."""
        self.bot.config['danger_mode'] = not self.bot.config['danger_mode']
        await ctx.send(f'Toggled danger mode o{"n" if self.bot.config["danger_mode"] else "ff"}.')

    @commands.command()
    @commands.is_owner()
    async def lockdown(self, ctx: Context, *, reason: str = ''):
        """Toggles owner-only mode."""
        assert self.bot.config['owner_only_mode'][0] or len(
            reason), 'Specify a reason.'
        self.bot.config['owner_only_mode'] = [
            not self.bot.config['owner_only_mode'][0], reason]
        await ctx.send(f'Toggled lockdown mode o{"n" if self.bot.config["owner_only_mode"][0] else "ff"}.')

    @commands.command(aliases=["rekiad", "relaod", "reloa", "re;pad", "relad", "reolad", "rr"])
    @commands.is_owner()
    async def reload(self, ctx: Context, cog: str = ""):
        """Reloads extensions within the bot while the bot is running."""
        if not cog:
            extensions = [a for a in self.bot.extensions.keys()]
            await asyncio.gather(*((self.bot.reload_extension(extension)) for extension in extensions))
            await ctx.send("Reloaded all extensions.")
        elif "src.cogs." + cog in self.bot.extensions.keys():
            await self.bot.reload_extension("src.cogs." + cog)
            await ctx.send(f"Reloaded extension `{cog}` from `src/cogs/{cog}.py`.")
        else:
            await ctx.send("Unknown extension provided.")
            return None

    @commands.command()
    @commands.is_owner()
    async def viewzip(self, ctx: Context):
        m = sorted(zipfile.ZipFile(BytesIO(await ctx.message.attachments[0].read())).namelist())
        n = '\n'.join(m)
        await ctx.send(f"```\n{n}```")

    @commands.group()
    @commands.is_owner()
    async def sprite(self, ctx: Context):
        pass

    @sprite.command()
    async def update(self, ctx: Context, sprite_name: str):
        async with self.bot.db.conn.cursor() as cur:
            pack_name = \
                (await (await cur.execute('SELECT source FROM tiles WHERE name = (?)', sprite_name)).fetchone())[0]
            if pack_name is None:
                return await ctx.error('The specified sprite doesn\'t exist.')
        try:
            zip = zipfile.ZipFile(BytesIO(await ctx.message.attachments[0].read()))
        except IndexError:
            return await ctx.error('You forgot to attach a zip.')
        file_name = None
        check = [
            os.path.basename(name)[
            :5] == 'text_' for name in zip.namelist()]
        for name in zip.namelist():
            name = os.path.basename(name)
            if name[:5] != 'text_' or all(check):
                file_name = re.match(r"(.+?)_\d+?_\d\.png", name)
                if file_name is not None:
                    file_name = file_name.groups()[0]
                    break
        if file_name is None:
            raise AssertionError('Couldn\'t find any valid sprites!')
        with open(f"data/custom/{pack_name}.toml", "r") as f:
            sprite_data = tomlkit.load(f)
        assert sprite_name in sprite_data, f"Sprite `{sprite_name}` not found!"
        data = sprite_data[sprite_name]
        old_sprite_name = data['sprite']
        data['sprite'] = file_name
        with open(f"data/custom/{pack_name}.toml", "w") as f:
            tomlkit.dump(data, f)
        for file in glob(f'data/sprites/{pack_name}/{old_sprite_name}*.png'):
            os.remove(file)
        for name in zip.namelist():
            sprite = zip.read(name)
            path = name.split("/")[-1]
            if len(path):
                with open(f"data/sprites/{pack_name}/{path}", "wb") as f:
                    f.write(sprite)
        await self.load_custom_tiles(pack_name)
        return await ctx.reply(f'Done. Updated the sprites of `{sprite_name}`.')

    @sprite.command()
    async def edit(self, ctx: Context, sprite_name: str, attribute: str, *value: str):
        async with self.bot.db.conn.cursor() as cur:
            pack_name = \
                (await (await cur.execute('SELECT source FROM tiles WHERE name = (?)', sprite_name)).fetchone())[0]
            if pack_name is None:
                return await ctx.error('The specified sprite doesn\'t exist.')
        if attribute not in ['sprite', 'tiling', 'color', 'name', 'tags']:
            return await ctx.error('You specified an invalid attribute.')
        if attribute == 'color':
            value = [int(val) for val in value]
        else:
            value = value[0]
        if attribute == 'tiling':
            value = TilingMode.parse(value)
        with open(f"data/custom/{pack_name}.toml", "r") as f:
            sprite_data = tomlkit.load(f)
        assert sprite_name in sprite_data, f"Sprite `{sprite_name}` not found!"
        if attribute == "name":
            data = sprite_data[sprite_name]
            sprite_data.remove(sprite_name)
            sprite_data[value] = data
        else:
            sprite_data[sprite_name][attribute] = data

        with open(f"data/custom/{pack_name}.toml", "w") as f:
            tomlkit.dump(sprite_data, f)
        await self.load_custom_tiles(pack_name)
        return await ctx.reply(f'Done. Replaced the attribute `{attribute}` in sprite `{sprite_name}` with `{value}`.')

    @sprite.command()
    async def add(
        self, ctx: Context, 
        pack_name: str, sprite_name: str, color_x: int, color_y: int, 
        tiling: typing.Literal["custom", "none", "directional", "tiling", "character", "animated_directional", "animated", "static_character", "diagonal_tiling"],
        *tags: str
    ):
        """Adds a sprite to the specified source."""
        tiling = TilingMode.parse(tiling)

        async with self.bot.db.conn.cursor() as cur:
            result = await cur.execute('SELECT DISTINCT name FROM tiles WHERE name = (?)', sprite_name)
            assert (await result.fetchone()) is None, "A sprite by that name already exists."
        try:
            zip = zipfile.ZipFile(BytesIO(await ctx.message.attachments[0].read()))
        except IndexError:
            return await ctx.error('You forgot to attach a zip.')
        file_name = None
        for name in zip.namelist():
            name = os.path.basename(name)
            file_name = re.match(r"(.+?)_\d+?_\d\.png", name)
            if file_name is not None:
                file_name = file_name.groups()[0]
                break
        if file_name is None:
            raise AssertionError('Couldn\'t find any valid sprites!')
        if sprite_name == '.':
            sprite_name = file_name
        for name in zip.namelist():
            sprite = zip.read(name)
            path = name.split("/")[-1]
            if len(path):
                try:
                    with open(f"data/sprites/{pack_name}/{path}", "wb") as f:
                        f.write(sprite)
                except FileNotFoundError:
                    return await ctx.error('That isn\'t a valid sprite directory.')
        with open(f"data/custom/{pack_name}.toml", "r") as f:
            sprite_data = tomlkit.load(f)
        data = {
            "sprite": file_name,
            "color": [ color_x, color_y ],
            "tiling": str(tiling)
        }
        if len(tags):
            data['tags'] = tags
        
        table = tomlkit.inline_table()
        table.update(data)
        sprite_data.add(tomlkit.nl())
        sprite_data.add(sprite_name, table)
        with open(f"data/custom/{pack_name}.toml", "w") as f:
            tomlkit.dump(sprite_data, f)
        await self.load_custom_tiles(pack_name)
        await ctx.send(f"Done. Added {sprite_name}.")

    @sprite.command(aliases=['del', 'rm', 'remove'])
    async def delete(self, ctx: Context, sprite_name: str):
        """Deletes a specified sprite."""
        async with self.bot.db.conn.cursor() as cur:
            source, sprite = await (
                await cur.execute('SELECT source, sprite FROM tiles WHERE name = (?)', sprite_name)).fetchone()
            if source is None or sprite is None:
                return await ctx.error(f'The sprite `{sprite_name}` doesn\'t exist.')
            await cur.execute('DELETE FROM tiles WHERE name = (?)', sprite_name)
        for file in glob(f'data/sprites/{source}/*.png'):
            # Extra check
            path = pathlib.Path(file)
            if re.match(rf"^{re.escape(sprite)}_\d+_\d+$", path.stem) is not None:
                os.remove(file)
        with open(f"data/custom/{source}.toml", "r") as f:
            sprite_data = tomlkit.load(f)
        if sprite_name in sprite_data:
            sprite_data.remove(sprite_name)
            with open(f"data/custom/{source}.toml", "w") as f:
                tomlkit.dump(sprite_data, f)
        await self.load_custom_tiles(source)
        await ctx.send(f"Done. Deleted `{sprite_name}` (`{sprite}` from `{source}`).")

    @commands.command(name="blacklist")
    @commands.is_owner()
    async def blacklist(self, ctx: Context, mode: str, user_id: int):
        """Set up a blacklist of users."""
        try:
            user = await self.bot.fetch_user(user_id)
        except discord.NotFound:
            return await ctx.error(f'User of id {user_id} was not found.')
        assert mode in [
            'add', 'remove'], 'Mode invalid! Has to be `add` or `remove`.'
        async with self.bot.db.conn.cursor() as cur:
            if mode == 'add':
                await cur.execute(f'''INSERT INTO blacklistedusers
                                VALUES ({user_id})''')
                return await ctx.reply(f'Added user `{user.name}#{user.discriminator}` to the blacklist.')
            else:
                await cur.execute(f'''DELETE FROM blacklistedusers
                                WHERE id={user_id}''')
                return await ctx.reply(f'Removed user `{user.name}#{user.discriminator}` from the blacklist.')
            

    @commands.command()
    @commands.is_owner()
    async def loadbab(self, ctx: Context):
        """Import all sprites from BAB BE U."""
        msg = await ctx.reply("Importing...")

        bab_palette = np.array(Image.open("data/misc/bab_palette.png").convert("RGBA"))
        color_table = {
            (3, 3): (3, 1),
            (3, 2): (3, 0),
            (5, 4): (2, 3),
            (3, 0): (4, 0),
            (3, 4): (3, 3),
            (3, 5): (3, 3)
        }

        data = {}

        for directory, filenames in (  # i wish this could be a dict
                ('objects', ("characters", "devs",
                                "special", "thingify", "ui", "unsorted")),
                ('text', ("conditions", "letters",
                            "properties", "tutorial", "unsorted", "verbs"))
        ):
            for filename in filenames:
                for babdata in requests.get(
                        f"https://raw.githubusercontent.com/lilybeevee/bab-be-u/master/assets/tiles/{directory}/{filename}.json").json():
                    data[babdata.pop("name")] = babdata
                    
        tiles = tomlkit.document()
        tiles.add(tomlkit.comment("Automatically generated by loadbab. Probably don't edit this."))
        tiles.add(tomlkit.nl())
        tiles.add(tomlkit.nl())
        tiles.add(tomlkit.nl())
        
        last_update = time.perf_counter()

        tilename_overrides = {
            "txt_:)": "txt_yay",
            "txt_:o": "txt_woah",
            "txt_:(": "txt_aw"
        }

        count = 0
        for name, tile in data.items():
            name = tilename_overrides.get(name, name)
            if name.startswith("txt_"):
                tilename = "text_bab_" + name[4:]
            else:
                tilename = "bab_" + name
            
            tilename = (
                tilename
                    .replace(">", "gt")
                    .replace(":", "colon")
                    .replace("&", "amp")
            )
            multicolor = False
            if len(tile['sprite']) > 1:
                color_x, color_y = 0, 3
                multicolor = True
            elif len(tile['color'][0]) == 2:
                color_x, color_y = tile['color'][0]
                if (color_x, color_y) in color_table:
                    color_x, color_y = color_table[(color_x, color_y)]
            else:
                with Image.open('data/palettes/default.png') as l:
                    default_palette = np.array(
                        l.convert('RGB'), dtype=np.uint8)
                closest_color = np.argmin(np.sum(abs(
                    default_palette - np.full(default_palette.shape, tile['color'][0])), axis=2))
                color_x, color_y = (closest_color %
                                    default_palette.shape[1], closest_color //
                                    default_palette.shape[1])
                color_x, color_y = int(color_x), int(color_y)
            
            sprite_name = ""
            for char in name:
                if char.isalnum() or char == '_':
                    sprite_name += char
                    continue
                sprite_name += hex(ord(char))[2:]

            sprites: list[Image.Image] = []
            broken = False
            for sprite in tile['sprite']:
                sprite = requests.get(
                    f"https://raw.githubusercontent.com/lilybeevee/bab-be-u/master/assets/sprites/{urllib.parse.quote(sprite)}.png"
                ).content
                try:
                    sprite = Image.open(BytesIO(sprite)).convert("RGBA")
                except:
                    broken = True
                    break
                sprites.append(sprite)
            if broken:
                continue
            if not multicolor:
                sprite = sprites[0]
            else:
                width = max(im.width for im in sprites)
                height = max(im.height for im in sprites)
                image = Image.new("RGBA", (width, height))
                for col, sprite in zip(tile['color'], sprites):
                    color = bab_palette[*col[::-1]] if len(col) < 3 else np.array([*col, 255]) # fuck it
                    sprite = np.array(sprite)
                    sprite = np.multiply(sprite, color.astype(float) / 255, casting="unsafe").astype(np.uint8)
                    sprite = Image.fromarray(sprite)
                    image.paste(sprite, mask=sprite)
                sprite = image

            for i in range(3):
                sprite.save(f"data/sprites/bab/{sprite_name}_0_{i + 1}.png")
            
            table = tomlkit.inline_table()
            table.update({"sprite": sprite_name, "color": [color_x, color_y], "tiling": str(TilingMode.NONE)})
            tiles.add(tomlkit.nl())
            tiles.add(tilename, table)
            
            count += 1
            if time.perf_counter() - last_update > 3:
                await msg.edit(content=f"Imported {count}/{len(data.keys())} tiles...")
                last_update = time.perf_counter()
        
        with open(f"data/custom/bab.toml", "w+") as f:
            tomlkit.dump(tiles, f)

        await self.load_custom_tiles("bab")
        await msg.edit(content="Done. Imported all tiles from bab.")


    @commands.command(aliases=["reboot", "rs"])
    @commands.is_owner()
    async def restart(self, ctx: Context):
        """Restarts the bot process."""
        await ctx.send("Restarting bot process...")
        await self.bot.change_presence(status=discord.Status.idle, activity=discord.Game(name="Rebooting..."))
        self.bot.exit_code = 1
        await self.bot.close()

    @commands.command(aliases=["kill", "yeet",
                               "defeat", "empty", "not", "kil", "k"])
    @commands.is_owner()
    async def logout(self, ctx: Context, endsentence: str = ""):
        """Kills the bot process."""
        if endsentence != "":  # Normally, logout doesn't trigger with arguments.
            if ctx.invoked_with == "not":
                if endsentence == "robot":  # Check if the argument is *actually* robot, making robot is not robot
                    await ctx.send("Poofing bot process...")
                    await self.bot.close()  # Trigger close before returning
            print("Almost killed")
            return  # Doesn't close the bot if any of these logic statements is false
        elif ctx.invoked_with == "not":
            return  # Catch "robot is not"
        elif ctx.invoked_with == "yeet":
            await ctx.send("Yeeting bot process...")
        elif ctx.invoked_with == "defeat":
            await ctx.send("Processing robot is defeat...")
        elif ctx.invoked_with == "empty":
            await ctx.send("Voiding bot process...")
        elif ctx.invoked_with == "kil":
            await ctx.send("<:wah:950360195199041556>")
        else:
            await ctx.send("Killing bot process...")
        await self.bot.close()

    @commands.command()
    @commands.is_owner()
    async def leave(self, ctx: Context, guild: Optional[discord.Guild] = None):
        if guild is None:
            if ctx.guild is not None:
                await ctx.send("Bye!")
                await ctx.guild.leave()
            else:
                await ctx.send("Not possible in DMs.")
        else:
            await guild.leave()
            await ctx.send(f"Left {guild}.")

    @commands.command()
    @commands.is_owner()
    async def loadsource(self, ctx: Context, source: str):
        """Reloads only a single source."""
        assert pathlib.Path(
            f'data/sprites/{source}').exists(), f'The source {source} doesn\'t exist.'
        self.bot.loading = True
        await self.load_custom_tiles(source)
        self.bot.loading = False
        return await ctx.send(f"Done. Loaded tile data from {source}.")

    @commands.command()
    @commands.is_owner()
    async def loadinitial(self, ctx: Context):
        self.bot.loading = True
        await self.load_initial_tiles()
        self.bot.loading = False
        return await ctx.send("Done. Loaded initial tile data.")

    @commands.command()
    @commands.is_owner()
    async def loadeditor(self, ctx: Context):
        self.bot.loading = True
        await self.load_editor_tiles()
        self.bot.loading = False
        return await ctx.send("Done. Loaded editor tile data.")

    @commands.command()
    @commands.is_owner()
    async def refreshslash(self, ctx: Context):
        async with ctx.typing():
            await self.bot.tree.sync()
            await ctx.send("Done.")

    @commands.command()
    @commands.is_owner()
    async def loadcustom(self, ctx: Context):
        self.bot.loading = True
        await self.load_custom_tiles()
        self.bot.loading = False
        return await ctx.send("Done. Loaded custom tile data.")

    @commands.command()
    @commands.is_owner()
    async def scrapevanilla(self, ctx: Context):
        """Scrapes tile data from the vanilla worlds' .ld files."""
        cached_exists = set()

        async with self.bot.db.conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM tiles WHERE source == 'baba' OR source == 'new_adv' OR source == 'museum'"
            )

        async def exists(tile: str):
            nonlocal ctx, cached_exists
            exists = tile in cached_exists or (await self.bot.db.tile(tile)) is not None
            cached_exists.add(tile)
            return exists

        await ctx.reply("Scraping tile metadata...")
        vanilla_worlds = ["baba", "new_adv", "museum"]
        async with ctx.typing():
            for world in vanilla_worlds:
                print(f"Scraping from {world}...")
                ld_files = (Path("data/levels") / world).glob("*.ld")
                sprites = {}
                for file in ld_files:
                    with open(file, "r") as f:
                        data = f.read().replace("\r\n", "\n")
                    if (start := data.find("[tiles]")) == -1:
                        continue
                    data = data[start + len("[tiles]"):]
                    end = data.find("\n[")
                    if end != -1:
                        data = data[:end]
                    data = data.strip()
                    # We now have a name=value file entry, essentially an ini
                    entries = {}
                    for line in data.splitlines():
                        eq = line.find("=")
                        if eq == -1: continue
                        name, value = line[:eq], line[eq+1:]
                        entries[name] = value
                    if not len(entries):
                        continue
                    changed = [id for id in entries["changed"].split(",") if len(id)]
                    tiles = {}
                    for object_id in changed:
                        try:
                            sprite_name = entries[f"{object_id}_image"]
                            if not await exists(sprite_name):
                                data = {
                                    name[len(object_id) + 1:]: value for name, value in entries.items()
                                    if name.startswith(object_id) 
                                        and name != sprite_name
                                }
                                data["object"] = object_id
                                if len(data):
                                    tiles[sprite_name] = data
                        except KeyError:
                            continue
                    for name, data in tiles.items():
                        if name not in sprites:
                            sprites[name] = {}
                        for key, value in data.items():
                            if key not in sprites[name]:
                                sprites[name][key] = []
                            sprites[name][key].append(value)
                # Consolidate the entries
                world_data = {}
                world_sprites = set()
                for name, data in sprites.items():
                    data = {key: collections.Counter(value).most_common(1)[0][0] for key, value in data.items()}
                    if (a := data.get("activecolour")) is not None:
                        data["colour"] = a
                    if (c := data.get("colour")) is not None:
                        data["colour"] = [int(c) for c in c.split(',')]
                    async with self.bot.db.conn.cursor() as cur:
                        res = await cur.execute(
                            "SELECT active_color_x, active_color_y, tiling FROM tiles WHERE object_id == ?",
                            data["object"]
                        )
                        row = await res.fetchone()
                        orig_color_x, orig_color_y, orig_tiling = row
                        data["colour"] = data.get("colour", [orig_color_x, orig_color_y])
                        data["tiling"] = int(data.get("tiling", orig_tiling))
                    data["tiling"] = str(TilingMode(data["tiling"]))
                    final_data = {
                        "color": data["colour"],
                        "tiling": data["tiling"],
                        "sprite": name
                    }
                    if "source" in data:
                        final_data["source"] = data["source"]
                    world_data[name] = final_data
                    world_sprites.add(name)
                
                # Now that we have the metadata, we need to add the sprites
                sprite_path = Path("data/sprites") / world
                world_sprite_path = Path("data/levels") / world / "Sprites"
                data_path = (Path("data/custom") / world).with_suffix(".toml")

                if sprite_path.exists():
                    shutil.rmtree(sprite_path)
                
                sprite_path.mkdir()
                # Copy the sprites
                for sprite in world_sprites:
                    for file in world_sprite_path.glob(f"{sprite}_*.png"):
                        shutil.copyfile(file, sprite_path / file.name)
                
                # Create the .toml
                doc = tomlkit.document()
                doc.add(tomlkit.comment("Generated by =scrapevanilla. Do not edit."))
                doc.add(tomlkit.nl())
                doc.add(tomlkit.nl())
                for name, data in world_data.items():
                    print(name, data)
                    table = tomlkit.inline_table()
                    table.update(data)
                    doc.add(tomlkit.nl())
                    doc.add(name, table)
                with open(data_path, "w+") as f:
                    tomlkit.dump(doc, f)
        
        await ctx.reply("Scraped vanilla worlds. Run =loadcustom.")
        self.bot.loading = False

    @commands.command()
    @commands.is_owner()
    async def loaddata(self, ctx: Context, flag: bool = False):
        """Reloads tile data from the world map, editor, and custom files.

        The boolean flag deletes all tiles from the database before
        updating with new tiles. [slow, do not do unless necessary]
        """
        self.bot.loading = True
        if flag:
            # Flush the tile database since it all gets reconstructed anyway
            await self.bot.db.conn.execute('DELETE FROM tiles')
        del self.bot.db.filter_cache  # Just to make absolutely sure that it gets flushed
        self.bot.db.filter_cache = {}
        await self.load_initial_tiles()
        await self.load_editor_tiles()
        await self.load_custom_tiles()
        self.bot.loading = False
        return await ctx.send("Done. Loaded all tile data.")

    async def load_initial_tiles(self):
        """Loads tile data from `data/values.lua` files."""
        # values.lua contains the data about which color (on the palette) is
        # associated with each tile.
        with open("data/values.lua", errors='ignore') as fp:
            data = fp.read()

        start = data.find("tileslist =\n")
        end = data.find("\n}", start)

        assert start > 0 and end > 0, "Failed to load values.lua!"
        spanned = data[start:end]

        def prepare(d: dict[str, Any]) -> dict[str, Any]:
            """From game format into DB format."""
            if d.get("type") is not None:
                d["text_type"] = d.pop("type")
            if d.get("image") is not None:
                d["sprite"] = d.pop("image")
            if d.get("colour") is not None:
                inactive = d.pop("colour").split(",")
                d["inactive_color_x"] = int(inactive[0])
                d["inactive_color_y"] = int(inactive[1])
            if d.get("activecolour") is not None:
                active = d.pop("activecolour").split(",")
                d["active_color_x"] = int(active[0])
                d["active_color_y"] = int(active[1])
            return d

        object_pattern = re.compile(
            r"(object\d+) =\n\t\{"
            r"\n\s*name = \"([^\"]*)\","
            r"\n\s*sprite = \"([^\"]*)\","
            r"\n.*\n.*\n\s*tiling = (-1|\d),"
            r"\n\s*type = (\d),"
            r"\n\s*(?:argextra = .*,\n\s*)?(?:argtype = .*,\n\s*)?"
            r"colour = \{(\d), (\d)},"
            r"\n\s*(?:active = \{(\d), (\d)},\n\s*)?"
            r".*\n.*\n.*\n\s*}",
        )
        initial_objects: dict[str, dict[str, Any]] = {}
        for match in re.finditer(object_pattern, spanned):
            obj, name, sprite, tiling, type, c_x, c_y, a_x, a_y = match.groups()
            print(f"Loading {name}")
            if a_x is None or a_y is None:
                inactive_x = active_x = int(c_x)
                inactive_y = active_y = int(c_y)
            else:
                inactive_x = int(c_x)
                inactive_y = int(c_y)
                active_x = int(a_x)
                active_y = int(a_y)
            tiling = int(tiling)
            if tiling == TilingMode.TILING:
                # Check for diagonal tiling
                if pathlib.Path(f"data/sprites/vanilla/{sprite}_16_1.png").exists():
                    tiling = +TilingMode.DIAGONAL_TILING
            type = int(type)
            initial_objects[obj] = dict(
                name=name,
                sprite=sprite,
                tiling=tiling,
                text_type=type,
                inactive_color_x=inactive_x,
                inactive_color_y=inactive_y,
                active_color_x=active_x,
                active_color_y=active_y,
                object_id=obj
            )

        await self.bot.db.conn.executemany(
            f'''
            INSERT INTO tiles(
                name,
                sprite,
                source,
                version,
                inactive_color_x,
                inactive_color_y,
                active_color_x,
                active_color_y,
                tiling,
                text_type,
                object_id
            )
            VALUES (
                :name,
                :sprite,
                'vanilla',
                0,
                :inactive_color_x,
                :inactive_color_y,
                :active_color_x,
                :active_color_y,
                :tiling,
                :text_type,
                :object_id
            )
            ON CONFLICT(name, version)
            DO UPDATE SET
                sprite=excluded.sprite,
                source='vanilla',
                inactive_color_x=excluded.inactive_color_x,
                inactive_color_y=excluded.inactive_color_y,
                active_color_x=excluded.active_color_x,
                active_color_y=excluded.active_color_y,
                tiling=excluded.tiling,
                object_id=excluded.object_id,
                text_type=excluded.text_type;
            ''',
            initial_objects.values()
        )

    async def load_editor_tiles(self):
        """Loads tile data from `data/editor_objectlist.lua`."""

        with open("data/editor_objectlist.lua", errors="replace") as fp:
            data = fp.read()

        start = data.find("editor_objlist = {")
        end = data.find("\n}", start)
        assert start > 0 and end > 0
        spanned = data[start:end]

        object_pattern = re.compile(
            r"\[\d+] = \{"
            r"\n\s*name = \"([^\"]*)\","
            r"(?:\n\s*sprite = \"([^\"]*)\",)?"
            r"\n.*"
            r"\n\s*tags = \{((?:\"[^\"]*?\"(?:,\"[^\"]*?\")*)?)},"
            r"\n\s*tiling = (-1|\d),"
            r"\n\s*type = (\d),"
            r"\n.*"
            r"\n\s*colour = \{(\d), (\d)},"
            r"(?:\n\s*colour_active = \{(\d), (\d)})?"
        )
        tag_pattern = re.compile(r"\"([^\"]*?)\"")
        objects = []
        for match in re.finditer(object_pattern, spanned):
            name, sprite, raw_tags, tiling, text_type, c_x, c_y, a_x, a_y = match.groups()
            print(f"Loading {name}")
            sprite = name if sprite is None else sprite
            a_x = c_x if a_x is None else a_x
            a_y = c_y if a_y is None else a_y
            active_x = int(a_x)
            active_y = int(a_y)
            inactive_x = int(c_x)
            inactive_y = int(c_y)
            tiling = int(tiling)
            if tiling == TilingMode.TILING:
                # Check for diagonal tiling
                if pathlib.Path(f"data/sprites/vanilla/{sprite}_16_1.png").exists():
                    tiling = +TilingMode.DIAGONAL_TILING
            text_type = int(text_type)
            tag_list = []
            for tag in re.finditer(tag_pattern, raw_tags):
                # hack but i am Not touching that regex
                tag_list.append(tag.group(0).replace('"', ''))
            tags = "\t".join(tag_list)

            objects.append(dict(
                name=name,
                sprite=sprite,
                tiling=tiling,
                text_type=text_type,
                inactive_color_x=inactive_x,
                inactive_color_y=inactive_y,
                active_color_x=active_x,
                active_color_y=active_y,
                tags=tags
            ))

        await self.bot.db.conn.executemany(
            f'''
            INSERT INTO tiles
            VALUES (
                :name,
                :sprite,
                'vanilla',
                1,
                :inactive_color_x,
                :inactive_color_y,
                :active_color_x,
                :active_color_y,
                :tiling,
                :text_type,
                NULL,
                :tags,
                '',
                NULL
            )
            ON CONFLICT(name, version)
            DO UPDATE SET
                sprite=excluded.sprite,
                source='vanilla',
                inactive_color_x=excluded.inactive_color_x,
                inactive_color_y=excluded.inactive_color_y,
                active_color_x=excluded.active_color_x,
                active_color_y=excluded.active_color_y,
                tiling=excluded.tiling,
                text_type=excluded.text_type,
                tags=:tags;
            ''',
            objects
        )

    async def load_custom_tiles(self, file='*'):
        """Loads custom tile data from `data/custom/*.toml`"""

        def prepare(source: str, name: str, d: dict[str, Any]) -> dict[str, Any]:
            """From config format to db format."""
            print(f"Loading {name}")
            db_dict = {key: value for key, value in d.items()}
            db_dict["name"] = name
            inactive = d.pop("color")
            if d.get("active") is not None:
                db_dict["inactive_color_x"] = inactive[0]
                db_dict["inactive_color_y"] = inactive[1]
                db_dict["active_color_x"] = d["active"][0]
                db_dict["active_color_y"] = d["active"][1]
            else:
                db_dict["inactive_color_x"] = db_dict["active_color_x"] = inactive[0]
                db_dict["inactive_color_y"] = db_dict["active_color_y"] = inactive[1]
            db_dict["source"] = d.get("source", source)
            db_dict["tiling"] = +TilingMode.parse(d.get("tiling", "none"))
            db_dict["text_type"] = d.get("text_type", 0)
            db_dict["text_direction"] = d.get("text_direction")
            db_dict["tags"] = "\t".join(d.get("tags", []))
            db_dict["extra_frames"] = "\t".join(str(value) for value in d.get("extra_frames", []))
            db_dict["object_id"] = d.get("object_id")
            return db_dict

        async with self.bot.db.conn.cursor() as cur:
            for path in pathlib.Path("data/custom").glob(f"{file}.toml"):
                source = path.parts[-1].split(".")[0]
                with open(path) as fp:
                    try:
                        objects = [prepare(source, name, obj) for name, obj in tomlkit.load(fp).items()]
                    except Exception as err:
                        raise AssertionError(f"Failed to load `{path}`!\n```\n{err}\n```")
                await cur.executemany(
                    '''
                    INSERT INTO tiles
                    VALUES (
                        :name,
                        :sprite,
                        :source,
                        2,
                        :inactive_color_x,
                        :inactive_color_y,
                        :active_color_x,
                        :active_color_y,
                        :tiling,
                        :text_type,
                        :text_direction,
                        :tags,
                        :extra_frames,
                        :object_id
                    )
                    ON CONFLICT(name, version)
                    DO UPDATE SET
                        sprite=excluded.sprite,
                        source=excluded.source,
                        inactive_color_x=excluded.inactive_color_x,
                        inactive_color_y=excluded.inactive_color_y,
                        active_color_x=excluded.active_color_x,
                        active_color_y=excluded.active_color_y,
                        tiling=excluded.tiling,
                        text_type=excluded.text_type,
                        text_direction=excluded.text_direction,
                        tags=excluded.tags,
                        extra_frames=excluded.extra_frames,
                        object_id=excluded.object_id;
                    ''',
                    objects
                )
                # this is a mega HACK, but I'm keeping it because the
                # alternative is a headache
                hacks = [x for x in objects if "baba_special" in x["tags"].split("\t")]
                await cur.executemany(
                    '''
                    INSERT INTO tiles
                    VALUES (
                        :name,
                        :sprite,
                        :source,
                        0,
                        :inactive_color_x,
                        :inactive_color_y,
                        :active_color_x,
                        :active_color_y,
                        :tiling,
                        :text_type,
                        :text_direction,
                        :tags,
                        :extra_frames,
                        :object_id
                    )
                    ON CONFLICT(name, version)
                    DO UPDATE SET
                        sprite=excluded.sprite,
                        source=excluded.source,
                        inactive_color_x=excluded.inactive_color_x,
                        inactive_color_y=excluded.inactive_color_y,
                        active_color_x=excluded.active_color_x,
                        active_color_y=excluded.active_color_y,
                        tiling=excluded.tiling,
                        text_type=excluded.text_type,
                        text_direction=excluded.text_direction,
                        tags=excluded.tags,
                        extra_frames=excluded.extra_frames,
                        object_id=excluded.object_id;
                    ''',
                    hacks
                )

    @commands.command()
    @commands.is_owner()
    async def hidden(self, ctx: Context):
        """Lists all hidden commands."""
        cmds = "\n".join([cmd.name for cmd in self.bot.commands if cmd.hidden])
        await ctx.author.send(f"All hidden commands:\n{cmds}")

    @commands.command(aliases=['clear', 'cls'])
    @commands.is_owner()
    async def clearconsole(self, ctx: Context):
        os.system('cls||clear')
        await ctx.send('Console cleared.')

    @commands.command(aliases=['execute', 'exec'], rest_is_raw=True)
    @commands.is_owner()
    async def run(self, ctx: Context, *, command: str):
        """Run a command from the command prompt."""
        result = subprocess.getoutput(command)
        if len(result) + 15 > 2000:
            result = result[:1982] + '...'
        await ctx.send(f'Output:\n```\n{result}```')

    @commands.command()
    @commands.is_owner()
    async def sql(self, ctx: Context, *, query: str):
        """Run some sql."""
        filemode = False
        if query[:3] == '-f ':
            query = query[3:]  # hardcode but whatever
            filemode = True
        async with self.bot.db.conn.cursor() as cur:
            result = await cur.execute(query)
            try:
                data_rows = await result.fetchall()
                data_column_headers = np.array(
                    [column[0] for column in result.get_cursor().description])
                data_columns = np.array(data_rows, dtype=object)
                formattable_columns = np.vstack(
                    [data_column_headers, data_columns]).T
                header = '+'
                for i, column in enumerate(formattable_columns):
                    max_length = 0
                    for cell in column:
                        if len(str(cell)) > max_length:
                            max_length = len(str(cell))
                    for j, cell in enumerate(column):
                        column[j] = f'{str(cell):{max_length}}'.replace(
                            '\t', ' ')
                    formattable_columns[i] = column
                    header = header + '-' * max_length + '+'
                formattable_rows = formattable_columns.T
                formatted = '|' + \
                            '|'.join(formattable_rows[0]) + f'|\n{header}'
                for row in formattable_rows[1:]:
                    formatted_row = '|' + '|'.join(row) + '|'
                    if not filemode and len(row) + len(formatted) > 1800:
                        formatted = formatted + '\n...Reached character limit!'
                        break
                    formatted = formatted + '\n' + formatted_row
            except TypeError:
                return await ctx.send(f"No output.")
        if filemode:
            await ctx.send('Writing file...', delete_after=5)
            out = BytesIO()
            out.write(bytes(formatted, 'utf-8'))
            out.seek(0)
            return await ctx.send('Output:', file=discord.File(out, filename='sql-output.txt'))
        return await ctx.send(f"Output:\n```\n{formatted}\n```")

    @commands.command()
    @commands.is_owner()
    async def loadletters(self, ctx: Context):
        """Scrapes individual letters from vanilla sprites."""
        await self.bot.db.conn.execute("DELETE FROM letters")
        ignored = constants.LETTER_IGNORE
        fetch = await self.bot.db.conn.fetchall(
            f'''
            SELECT * FROM tiles
            WHERE sprite LIKE "text\\___%" ESCAPE "\\"
                AND (source == 'vanilla' OR source == 'baba' OR source == 'new_adv' OR source == 'museum')
                AND text_direction IS NULL;
            '''
        )
        for i, row in enumerate(fetch):
            data = TileData.from_row(row)
            if data.sprite not in ignored:
                try:
                    await self.load_letter(
                        data.sprite,
                        data.text_type,  # type: ignore
                        data.source
                    )
                except FileNotFoundError:
                    pass

        await self.load_ready_letters()

        await ctx.send("Letters loaded.")

    @commands.command()
    @commands.is_owner()
    async def loadreadyletters(self, ctx: Context):
        await self.load_ready_letters()
        await ctx.send("Ready letters loaded.")

    @commands.command(aliases=['mkdir'])
    @commands.is_owner()
    async def makedir(self, ctx: Context, name: str):
        """Makes a directory for sprites to go in."""
        os.mkdir(f'data/sprites/{name}')
        with open(f'data/custom/{name}.toml', mode='x') as f:
            f.write("# Please read CONTRIBUTING.md for guidance on how to properly edit this file.\n\n\n")
        await ctx.send(f"Made directory `{name}`.")

    async def load_letter(self, word: str, tile_type: int, source: str):
        """Scrapes letters from a sprite."""
        chars = word[5:]  # Strip "text_" prefix

        # Get the number of rows
        two_rows = len(chars) >= 4

        # Background plates for type-2 text,
        # in 1 bit per pixel depth
        plates = [self.bot.db.plate(None, i)[0].getchannel(
            "A").convert("1") for i in range(3)]

        # Maps each character to three bounding boxes + images
        # (One box + image for each frame of animation)
        # char_pos : [((x1, y1, x2, y2), Image), ...]
        char_sizes: dict[tuple[int, str], Any] = {}

        # Scrape the sprites for the sprite characters in each of the three
        # frames
        for i, plate in enumerate(plates):
            # Get the alpha channel in 1-bit depth
            alpha = Image.open(f"data/sprites/{source}/{word}_0_{i + 1}.png") \
                .convert("RGBA") \
                .getchannel("A") \
                .convert("1")

            # Type-2 text has inverted text on a background plate
            if tile_type == 2:
                alpha = ImageChops.invert(alpha)
                alpha = ImageChops.logical_and(alpha, plate)

            # Get the point from which characters are seeked for
            x = 0
            y = 6 if two_rows else 12

            # Flags
            skip = False

            # More than 1 bit per pixel is required for the flood fill
            alpha = alpha.convert("L")
            for i, char in enumerate(chars):
                if skip:
                    skip = False
                    continue

                while alpha.getpixel((x, y)) == 0:
                    if x == alpha.width - 1:
                        if two_rows and y == 6:
                            x = 0
                            y = 18
                        else:
                            break
                    else:
                        x += 1
                # There's a letter at this position
                else:
                    clone = alpha.copy()
                    ImageDraw.floodfill(clone, (x, y), 128)  # 1 placeholder
                    clone = Image.eval(clone, lambda x: 255 if x == 128 else 0)
                    clone = clone.convert("1")

                    # Get bounds of character blob
                    x1, y1, x2, y2 = clone.getbbox()  # type: ignore
                    # Run some checks
                    # # Too wide => Skip 2 characters (probably merged two chars)
                    # if x2 - x1 > (1.5 * alpha.width * (1 + two_rows) / len(chars)):
                    #     skip = True
                    #     alpha = ImageChops.difference(alpha, clone)
                    #     continue

                    # Too tall? Scrap the rest of the characters
                    if y2 - y1 > 1.5 * alpha.height / (1 + two_rows):
                        break

                    # too thin! bad letter.
                    if x2 - x1 <= 2:
                        alpha = ImageChops.difference(alpha, clone)
                        continue

                    # Remove character from sprite, push to char_sizes
                    alpha = ImageChops.difference(alpha, clone)
                    clone = clone.crop((x1, y1, x2, y2))
                    entry = ((x1, y1, x2, y2), clone)
                    char_sizes.setdefault((i, char), []).append(entry)
                    continue
                return

        results = []
        for (_, char), entries in char_sizes.items():
            # All three frames clearly found the character in the sprite
            if len(entries) == 3:
                x1_min = min(entries, key=lambda x: x[0][0])[0][0]
                y1_min = min(entries, key=lambda x: x[0][1])[0][1]
                x2_max = max(entries, key=lambda x: x[0][2])[0][2]
                y2_max = max(entries, key=lambda x: x[0][3])[0][3]

                blobs = []
                mode = "small" if two_rows else "big"
                width = 0
                height = 0
                for i, ((x1, y1, _, _), img) in enumerate(entries):
                    frame = Image.new("1", (x2_max - x1_min, y2_max - y1_min))
                    frame.paste(img, (x1 - x1_min, y1 - y1_min))
                    width, height = frame.size
                    buf = BytesIO()
                    frame.convert("L").save(buf, format="PNG")
                    blobs.append(buf.getvalue())
                results.append((mode, char, width, *blobs))

        await self.bot.db.conn.executemany(
            '''
            INSERT INTO letters
            VALUES (?, ?, ?, ?, ?, ?);
            ''',
            results
        )

    async def load_ready_letters(self):
        def channel_shenanigans(im: Image.Image) -> Image.Image:
            if im.mode == "L":
                return im
            elif im.mode in ("RGB", "1"):
                return im.convert("L")
            return im.convert("RGBA").getchannel("A")

        data = []
        for path in pathlib.Path("data/letters").glob("*/*/*/*_0.png"):
            _, _, mode, char, w, name = path.parts
            replacelist = [('asterisk', '*'),
                           ('questionmark', '?'),
                           ('period', '.')]
            for original, substitute in replacelist:
                char = char.replace(original, substitute)
            width = int(w)
            prefix = name[:-6]
            # mo ch w h
            buf_0 = BytesIO()
            channel_shenanigans(Image.open(path)).save(buf_0, format="PNG")
            blob_0 = buf_0.getvalue()
            buf_1 = BytesIO()
            channel_shenanigans(Image.open(
                path.parent / f"{prefix}_1.png")).save(buf_1, format="PNG")
            blob_1 = buf_1.getvalue()
            buf_2 = BytesIO()
            channel_shenanigans(Image.open(
                path.parent / f"{prefix}_2.png")).save(buf_2, format="PNG")
            blob_2 = buf_2.getvalue()
            data.append((mode, char, width, blob_0, blob_1, blob_2))

        await self.bot.db.conn.executemany(
            '''
            INSERT INTO letters
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            data
        )

    @commands.command()
    @commands.is_owner()
    async def loadbaba(self, ctx: Context, *, path: str):
        """Adds all missing files from the base game. Should only have to be run when the game updates."""
        path = pathlib.Path(path) / "Data"
        assert path.exists, "Invalid directory!"
        bot_path = pathlib.Path(os.getcwd()) / "data"

        def replace(src, dst, *args):
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, *args)

        message = await ctx.reply(f"Adding sprites...")
        replace(path / "Worlds" / "baba" / "Sprites", bot_path / "sprites" / "vanilla")
        shutil.copytree(path / "Sprites", bot_path / "sprites" / "vanilla", dirs_exist_ok=True)
        shutil.copytree(path / "Palettes", bot_path / "palettes", dirs_exist_ok=True)
        shutil.copy2(path / "merged.ttf", bot_path / "fonts" / "default.ttf")
        shutil.copy2(path / "Editor" / "editor_objectlist.lua", bot_path / "editor_objectlist.lua")
        shutil.copy2(path / "values.lua", bot_path / "values.lua")
        for world in glob(str(path / "Worlds" / "*")):
            world = pathlib.Path(world)
            world_name = world.stem
            await message.edit(content=f"Adding world `{world_name}`...")
            if (bot_world_path := pathlib.Path(bot_path) / "levels" / world_name).exists():
                shutil.rmtree(bot_world_path)
            shutil.copytree(world, bot_world_path, dirs_exist_ok=True)
            replace(world / "Images", pathlib.Path(bot_path) / "images" / world_name)
        await message.edit(content="Done.")


    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        webhook = await self.bot.fetch_webhook(self.bot.webhook_id)
        embed = discord.Embed(
            color=self.bot.embed_color,
            title="Joined Guild",
            description=f"Joined {guild.name}."
        )
        embed.add_field(name="ID", value=str(guild.id))
        embed.add_field(name="Member Count", value=str(guild.member_count))
        await webhook.send(embed=embed)


async def setup(bot: Bot):
    await bot.add_cog(OwnerCog(bot))
