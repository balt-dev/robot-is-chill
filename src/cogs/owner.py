from __future__ import annotations

import shutil
from glob import glob
from io import BytesIO
import json
import zipfile
import pathlib
import re
from json import JSONDecodeError

import requests
import itertools
import collections
from src import constants
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
        with open(f"data/custom/{pack_name}.json", "r") as f:
            sprite_data = json.load(f)
        found = False
        # make sure this doesn't delete anything on accident
        old_sprite_name = "ALSDFHASDFHSADLFBCASDPFHINsaDLKJFFBSADLFBLSADKASDFNLSADKF"
        for i in range(len(sprite_data)):
            if sprite_data[i]['name'] == sprite_name:  # this is dumb
                old_sprite_name = sprite_data[i]['sprite']
                sprite_data[i]['sprite'] = file_name
                found = True
                break
        assert found, f"Sprite `{sprite_name}` not found!"
        with open(f"data/custom/{pack_name}.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
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
        value = ' '.join(value)
        async with self.bot.db.conn.cursor() as cur:
            pack_name = \
                (await (await cur.execute('SELECT source FROM tiles WHERE name = (?)', sprite_name)).fetchone())[0]
            if pack_name is None:
                return await ctx.error('The specified sprite doesn\'t exist.')
        if attribute not in ['sprite', 'tiling', 'color', 'name', 'tags']:
            return await ctx.error('You specified an invalid attribute.')
        if attribute == 'color':
            value = value.split(' ')
            assert len(value) == 2, 'Invalid color format!'
        elif attribute == 'tags':
            value = value.replace(' ', '\t')
        with open(f"data/custom/{pack_name}.json", "r") as f:
            sprite_data = json.load(f)
        found = False
        for i in range(len(sprite_data)):
            if sprite_data[i]['name'] == sprite_name:  # this is dumb
                sprite_data[i][attribute] = value
                found = True
                break
        assert found, f"Sprite `{sprite_name}` not found!"
        with open(f"data/custom/{pack_name}.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
        await self.load_custom_tiles(pack_name)
        return await ctx.reply(f'Done. Replaced the attribute `{attribute}` in sprite `{sprite_name}` with `{value}`.')

    @sprite.command()
    async def add(self, ctx: Context, pack_name: str, sprite_name: str, color_x: int, color_y: int, tiling: str,
                  *tags: str):  # int | str didn't wanna work for me
        """Adds sprites to a specified sprite pack."""
        try:
            tiling = int(tiling)
        except ValueError:
            async with self.bot.db.conn.cursor() as cur:
                result = await cur.execute('SELECT DISTINCT name FROM tiles WHERE name = (?)', sprite_name)
                assert (await result.fetchone()) is None, "A sprite by that name already exists."
                result = await cur.execute('SELECT DISTINCT tiling FROM tiles WHERE name = (?)', tiling)
                try:
                    tiling = (await result.fetchone())[0]
                except BaseException:
                    return await ctx.error(f'The specified tile doesn\'t exist.')
        try:
            zip = zipfile.ZipFile(BytesIO(await ctx.message.attachments[0].read()))
        except IndexError:
            return await ctx.error('You forgot to attach a zip.')
        check = [
            os.path.basename(name)[
            :5] == 'text_' for name in zip.namelist()]
        withtext = any(check) and not all(check)
        file_name = None
        for name in zip.namelist():
            name = os.path.basename(name)
            if name[:5] != 'text_' or all(check):
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
        with open(f"data/custom/{pack_name}.json", "r") as f:
            sprite_data = json.load(f)
        data = {
            "name": sprite_name,
            "sprite": file_name,
            "color": [
                str(color_x),
                str(color_y)
            ],
            "tiling": str(tiling)
        }
        if len(tags):
            data['tags'] = tags
        sprite_data.append(data)
        if withtext:
            text_data = {
                "name": f'text_{sprite_name}',
                "sprite": f'text_{file_name}',
                "color": [
                    str(color_x),
                    str(color_y)
                ],
                "tiling": "-1"
            }
            if len(tags):
                text_data['tags'] = tags.join('\t')
            sprite_data.append(text_data)
        with open(f"data/custom/{pack_name}.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
        await self.load_custom_tiles(pack_name)
        await ctx.send(f"Done. Added {sprite_name}.")

    @sprite.command(aliases=['del', 'rm', 'remove'])
    async def delete(self, ctx: Context, sprite_name: str):
        """Deletes a specified sprite."""
        async with self.bot.db.conn.cursor() as cur:
            source, sprite = await (
                await cur.execute('SELECT source, sprite FROM tiles WHERE name = (?)', sprite_name)).fetchone()
            if source is None or sprite is None:
                return await ctx.error(f'The sprite {sprite_name} doesn\'t exist.')
            result = await cur.execute('DELETE FROM tiles WHERE name = (?)', sprite_name)
        for file in glob(f'data/sprites/{source}/{sprite}*.png'):
            os.remove(file)
        with open(f"data/custom/{source}.json", "r") as f:
            sprite_data = json.load(f)
        for i in range(len(sprite_data)):
            if sprite_data[i]['name'] == sprite_name:  # this is dumb
                del sprite_data[i]
                break
        with open(f"data/custom/{source}.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
        await self.load_custom_tiles(source)
        await ctx.send(f"Done. Deleted `{sprite_name}` (`{sprite}` from `{source}`).")

    @sprite.command()
    async def move(self, ctx: Context, sprite_name: str, new_source: str, reload: bool = True):
        """Moves a specified sprite to a different source."""
        data = None
        assert pathlib.Path(
            f'data/sprites/{new_source}').exists(), f'The source {new_source} doesn\'t exist.'
        async with self.bot.db.conn.cursor() as cur:
            source, sprite = await (
                await cur.execute('SELECT source, sprite FROM tiles WHERE name = (?)', sprite_name)).fetchone()
            if source is None or sprite is None:
                return await ctx.error(f'The sprite {sprite_name} doesn\'t exist.')
            await cur.execute('UPDATE tiles SET source = (?) WHERE name = (?)', (new_source, sprite_name))
        for file in glob(f'data/sprites/{source}/{sprite}*.png'):
            os.rename(
                file,
                f'data{os.sep}sprites{os.sep}{new_source}{os.sep}{pathlib.Path(file).stem}.png')
        with open(f"data/custom/{source}.json", "r") as f:
            sprite_data = json.load(f)
        for i in range(len(sprite_data)):
            if sprite_data[i]['name'] == sprite_name:  # this is dumb
                data = sprite_data.pop(i)
                break
        with open(f"data/custom/{source}.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
        with open(f"data/custom/{new_source}.json", "r") as f:
            new_data = json.load(f)
        new_data.append(data)
        with open(f"data/custom/{new_source}.json", "w") as f:
            json.dump(new_data, f, indent=4)
        if reload:
            await self.load_custom_tiles(source)
            await self.load_custom_tiles(new_source)
        await ctx.send(
            f"Done. Moved {sprite_name} to {new_source}.{' (Remember to reload custom tiles!)' if not reload else ''}")

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
    async def importbab(self, ctx: Context, name: str, color_x: int = None, color_y: int = None,
                        transform_txt_text: bool = True):
        """Auto-import a bab sprite."""
        await ctx.send(f"Hold on, robobot be scan bab...")
        assert name.find(':') == -1, 'Name has colon in it, so it won\'t work.'
        color_table = {
            (3, 3): (3, 1),
            (3, 2): (3, 0),
            (5, 4): (2, 3),
            (3, 0): (4, 0),
            (3, 4): (3, 3),
            (3, 5): (3, 3)
        }

        def scanforname(name):
            for directory, filenames in (  # i wish this could be a dict
                    ('objects', ("characters", "devs",
                                 "special", "thingify", "ui", "unsorted")),
                    ('text', ("conditions", "letters",
                              "properties", "tutorial", "unsorted", "verbs"))
            ):
                for filename in filenames:
                    for babdata in requests.get(
                            f"https://raw.githubusercontent.com/lilybeevee/bab-be-u/master/assets/tiles/{directory}/{filename}.json").json():
                        if babdata["name"] == name:
                            return babdata

        babdata = scanforname(name)
        try:
            sprite = requests.get(
                f"https://raw.githubusercontent.com/lilybeevee/bab-be-u/master/assets/sprites/{babdata['sprite'][0]}.png").content
        except TypeError:
            raise AssertionError(f'Bab til `{name}` not found!')
        if isinstance(color_x, type(None)) or isinstance(color_y, type(None)):
            if len(babdata['color'][0]) == 2:
                color_x, color_y = babdata['color'][0]
                if (color_x, color_y) in color_table:
                    color_x, color_y = color_table[(color_x, color_y)]
            else:
                with Image.open('data/palettes/default.png') as l:
                    default_palette = np.array(
                        l.convert('RGB'), dtype=np.uint8)
                closest_color = np.argmin(np.sum(abs(
                    default_palette - np.full(default_palette.shape, babdata['color'][0])), axis=2))
                color_x, color_y = (closest_color %
                                    default_palette.shape[1], closest_color //
                                    default_palette.shape[1])
        # if not os.path.isdir(f"data/sprites/{pack_name}") or not os.path.isfile(f"data/custom/{pack_name}.json"):
        # return await ctx.error(f"Pack {pack_name} doesn't exist.") #fuck off,
        # the bab pack exists.
        pilsprite = Image.open(BytesIO(sprite))
        pilsprite = pilsprite.resize(
            ((pilsprite.width * 3) // 4,
             (pilsprite.height * 3) // 4),
            Image.NEAREST)
        if transform_txt_text:
            if name.startswith("txt_"):
                name = "text_" + name[4:]
        for i in range(3):
            # with open(f"data/sprites/{pack_name}/{name}_0_{i+1}.png", "wb") as f:
            #     f.write(sprite)
            pilsprite.save(f"data/sprites/bab/{name}_0_{i + 1}.png")
        with open(f"data/custom/bab.json", "r") as f:
            sprite_data = json.load(f)
        sprite_data.append({
            "name": name,
            "sprite": name,
            "color": [
                str(color_x),
                str(color_y)
            ],
            "tiling": "-1"
        })
        with open(f"data/custom/bab.json", "w") as f:
            json.dump(sprite_data, f, indent=4)
        await self.load_custom_tiles()
        await ctx.send(f"Added {name} from bab.")

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
        """Loads tile data from `data/values.lua` and `.ld` files."""
        # values.lua contains the data about which color (on the palette) is
        # associated with each tile.
        with open("data/values.lua", errors='ignore') as fp:
            data = fp.read()

        start = data.find("tileslist =\n")
        end = data.find("\n}\n", start)

        assert start > 0 and end > 0
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
            if a_x is None or a_y is None:
                inactive_x = active_x = int(c_x)
                inactive_y = active_y = int(c_y)
            else:
                inactive_x = int(c_x)
                inactive_y = int(c_y)
                active_x = int(a_x)
                active_y = int(a_y)
            tiling = int(tiling)
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
            )

        changed_objects: list[dict[str, Any]] = []

        by_name = itertools.groupby(
            sorted(changed_objects, key=lambda x: x["name"]),
            key=lambda x: x["name"]
        )
        ready: list[dict[str, Any]] = []
        for name, duplicates in by_name:
            def freeze_dict(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
                """Hashable (frozen) dict."""
                return tuple(d.items())

            counts = collections.Counter(map(freeze_dict, duplicates))
            most_common, _ = counts.most_common(1)[0]
            ready.append(dict(most_common))

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
                text_type
            )
            VALUES (
                :name,
                :sprite,
                {repr(constants.BABA_WORLD)},
                0,
                :inactive_color_x,
                :inactive_color_y,
                :active_color_x,
                :active_color_y,
                :tiling,
                :text_type
            )
            ON CONFLICT(name, version)
            DO UPDATE SET
                sprite=excluded.sprite,
                source={repr(constants.BABA_WORLD)},
                inactive_color_x=excluded.inactive_color_x,
                inactive_color_y=excluded.inactive_color_y,
                active_color_x=excluded.active_color_x,
                active_color_y=excluded.active_color_y,
                tiling=excluded.tiling,
                text_type=excluded.text_type;
            ''',
            initial_objects.values()
        )

        await self.bot.db.conn.executemany(
            f'''
            INSERT INTO tiles
            VALUES (
                :name,
                :sprite,
                {repr(constants.BABA_WORLD)},
                0,
                :inactive_color_x,
                :inactive_color_y,
                :active_color_x,
                :active_color_y,
                :tiling,
                :text_type,
                NULL,
                ""
            )
            ON CONFLICT(name, version) DO NOTHING;
            ''',
            ready
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
            sprite = name if sprite is None else sprite
            a_x = c_x if a_x is None else a_x
            a_y = c_y if a_y is None else a_y
            active_x = int(a_x)
            active_y = int(a_y)
            inactive_x = int(c_x)
            inactive_y = int(c_y)
            tiling = int(tiling)
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
                {repr(constants.BABA_WORLD)},
                1,
                :inactive_color_x,
                :inactive_color_y,
                :active_color_x,
                :active_color_y,
                :tiling,
                :text_type,
                NULL,
                :tags
            )
            ON CONFLICT(name, version)
            DO UPDATE SET
                sprite=excluded.sprite,
                source={repr(constants.BABA_WORLD)},
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
        """Loads custom tile data from `data/custom/*.json`"""

        def prepare(source: str, d: dict[str, Any]) -> dict[str, Any]:
            """From config format to db format."""
            inactive = d.pop("color")
            if d.get("active") is not None:
                d["inactive_color_x"] = inactive[0]
                d["inactive_color_y"] = inactive[1]
                d["active_color_x"] = d["active"][0]
                d["active_color_y"] = d["active"][1]
            else:
                d["inactive_color_x"] = d["active_color_x"] = inactive[0]
                d["inactive_color_y"] = d["active_color_y"] = inactive[1]
            d["source"] = d.get("source", source)
            d["tiling"] = d.get("tiling", -1)
            d["text_type"] = d.get("text_type", 0)
            d["text_direction"] = d.get("text_direction")
            d["tags"] = d.get("tags", "")
            return d

        async with self.bot.db.conn.cursor() as cur:
            try:
                for path in pathlib.Path("data/custom").glob(f"{file}.json"):
                    source = path.parts[-1].split(".")[0]
                    with open(path) as fp:
                        objects = [prepare(source, obj) for obj in json.load(fp)]
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
                            :tags
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
                            tags=excluded.tags;
                        ''',
                        objects
                    )
                    # this is a mega HACK, but I'm keeping it because the
                    # alternative is a headache
                    hacks = [
                        x for x in objects if "baba_special" in x["tags"].split("\t")]
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
                            :tags
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
                            tags=excluded.tags;
                        ''',
                        hacks
                    )
            except JSONDecodeError as e:
                raise AssertionError(f"Failed to parse `{path}`!\n{e.msg}")

    @commands.command()
    @commands.is_owner()
    async def hidden(self, ctx: Context):
        """Lists all hidden commands."""
        cmds = "\n".join([cmd.name for cmd in self.bot.commands if cmd.hidden])
        await ctx.send(f"All hidden commands:\n{cmds}")

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
        ignored = json.load(open("config/letterignore.json"))
        fetch = await self.bot.db.conn.fetchall(
            f'''
            SELECT * FROM tiles
            WHERE sprite LIKE "text\\___%" ESCAPE "\\"
                AND source == {repr(constants.BABA_WORLD)}
                AND text_direction IS NULL;
            '''
        )
        for i, row in enumerate(fetch):
            data = TileData.from_row(row)
            if data.sprite not in ignored:
                await self.load_letter(
                    data.sprite,
                    data.text_type  # type: ignore
                )

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
        with open(f'data/custom/{name}.json', mode='x') as f:
            f.write('[]')
        await ctx.send(f"Made directory `{name}`.")

    async def load_letter(self, word: str, tile_type: int):
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
            alpha = Image.open(f"data/sprites/{constants.BABA_WORLD}/{word}_0_{i + 1}.png") \
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
        replace(path / "Worlds" / "baba" / "Sprites", bot_path / "sprites" / "baba")
        shutil.copytree(path / "Sprites", bot_path / "sprites" / "baba", dirs_exist_ok=True)
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
