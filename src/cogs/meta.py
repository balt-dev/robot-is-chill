from __future__ import annotations

import asyncio
import colorsys
import os
from functools import reduce
import itertools
from datetime import datetime
from subprocess import PIPE, STDOUT, TimeoutExpired, run
from time import time
from typing import Optional, Sequence

import re

import discord
from discord.ext import commands, menus
from discord.ext.commands import Command

from ..types import Bot, Context
from ..utils import ButtonPages


class CommandPageSource(menus.ListPageSource):
    def __init__(self, data: Sequence[tuple[Command]]):
        super().__init__(data, per_page=1)

    async def format_page(self, menu: menus.Menu, entry: Command) -> discord.Embed:
        embed = discord.Embed(
            color=menu.bot.embed_color,
            title=entry.name,
            description=entry.help
        )
        return embed

class MetaCog(commands.Cog, name="Other Commands"):
    def __init__(self, bot: Bot):
        self.bot = bot

    # Check if the bot is loading
    async def cog_check(self, ctx: Context):
        return not self.bot.loading


    @commands.command(aliases=["pong"])
    @commands.cooldown(5, 8, commands.BucketType.channel)
    async def ping(self, ctx: Context):
        """Returns bot latency."""
        def clamp(val, mn, mx): return max(min(val, mx), mn)
        pingns = int(self.bot.latency * 1000)
        color = reduce(
            lambda a, b: (a << 8) + b,
            [int(255 * n) for n in colorsys.hsv_to_rgb((0.33333333 -
                                                        ((pingns / 250) * 0.33333333)) % 1, 0.4, 1)]
        )
        await ctx.send(embed=discord.Embed(
            title="Latency",
            color=discord.Color(color),
            description=f"{pingns} ms"))

    @commands.command(aliases=["commands"])
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def cmds(self, ctx: Context, query: Optional[str] = None):
        """Lists the bot's commands. You are here!
    Commands can be specified through a regular expression."""
        if query is None:
            query = ""
        elif query == "list":
            names = sorted((cmd.name for cmd in self.bot.commands if not cmd.hidden))
            nl = "\n"
            return await ctx.send(f"""```
{nl.join(names)}```""")
        await ButtonPages(
            source=CommandPageSource(
                sorted((cmd for cmd in self.bot.commands if re.match(query, cmd.name) and not cmd.hidden), key=lambda cmd: cmd.name)
            ),
        ).start(ctx)

    @commands.command(aliases=["about", "info"])
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def help(self, ctx: Context, query: Optional[str] = None):
        """Directs new users on what they can do and where they can go."""
        if query is not None:
            return self.cmds(ctx, query)
        embed = discord.Embed(
            title="ROBOT IS CHILL",
            color=self.bot.embed_color
        )
        embed.add_field(
            name="",
            value="""Welcome to the bot! This help page is still under construction.
- If you need a list of tiles you can use, look through `search`.
- If you need a list of commands, look at `commands`.
- If you need to make a render, look at `commands tile`.
- If you need help on a level, look at `hints <level name>`.
- If you need to look at a level, look at `level`.""",
            inline=False
        )
        ut = datetime.utcnow() - self.bot.started
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute("SELECT COUNT(DISTINCT name) FROM tiles")
            tile_amount = (await cur.fetchone())[0]
        embed.add_field(
            name="Statistics",
            value=f"""Guilds: {len(self.bot.guilds)}
Channels: {sum(len(g.channels) for g in self.bot.guilds)}
Uptime: {str(ut).split('.', 1)[0]}
Tiles: {tile_amount}""",
            inline=True
        )
        embed.add_field(
            name="Developers",
            value="""_balt#6423_ - Current lead
_CenTdemeern1#3610_ - Co-lead
_RocketRace#0798_ - Original lead
""",
            inline=True
        )
        embed.add_field(
            name="Links",
            value="""[Invitation](https://balt.sno.mba/chill/invite)
[Support Guild](https://balt.sno.mba/chill/server)
[GitHub](https://balt.sno.mba/chill/github)""",
            inline=True
        )
        embed.add_field(
            name="Credits",
            value="""[Baba Is Bookmark](https://baba-is-bookmark.herokuapp.com/) - SpiccyMayonnaise
        [Baba Is Hint](https://www.keyofw.com/baba-is-hint/) - keyofw""",
            inline=True
        )
        await ctx.send(embed=embed)

    @commands.command(aliases=["interpret"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def babalang(self, ctx: Context, program: str, *program_input: str):
        """Interpret a [Babalang v1.1.1](https://esolangs.org/wiki/Babalang)
        program.

        The first argument must be the source code for the program, escaped in quotes:

        * e.g. `"baba is group and word and text"`

        The second argument is the optional input, also escaped in quotes:

        * e.g. `"foo bar"`

        Both arguments can be multi-line. The input argument will be automatically padded
        with trailing newlines as necessary.
        """
        prog_input = program_input
        if len(prog_input) > 1:
            program = " ".join([program] + list(prog_input))
            prog_input = ""
        elif len(prog_input) == 1 and prog_input[0] and prog_input[0][-1] != "\n":
            prog_input = prog_input[0] + "\n"
        else:
            prog_input = ""

        def interpret_babalang():
            try:
                if os.name == "nt":
                    babalang_executable_path = "./src/babalang.exe"
                else:
                    babalang_executable_path = "./src/babalang"
                process = run(
                    [babalang_executable_path, "-c", f"'{program}'"],
                    stdout=PIPE,
                    stderr=STDOUT,
                    timeout=1.0,
                    input=prog_input.encode("utf-8", "ignore"),
                )
                if process.stdout is not None:
                    return (process.returncode,
                            process.stdout[:1000].decode("utf-8", "replace"))
                else:
                    return (process.returncode, "")
            except TimeoutExpired as timeout:
                if timeout.output is not None:
                    if isinstance(timeout.output, bytes):
                        return (None, timeout.output[:1000].decode(
                            "utf-8", "replace"))
                    else:
                        return (None, timeout.output)
                else:
                    return (None, None)
        return_code, output = await self.bot.loop.run_in_executor(None, interpret_babalang)

        too_long = False
        if output:
            lines = output.splitlines()
            if len(lines) > 50:
                output = "\n".join(lines[:50])
                too_long = True
            if len(output) > 500:
                output = output[:500]
                too_long = True

        message = []
        if return_code is None:
            message.append("The program took too long to execute:\n")
        else:
            message.append(
                f"The program terminated with return code `{return_code}`:\n")

        if not output:
            message.append("```\n[No output]\n```")
        elif too_long:
            message.append(
                f"```\n{output} [...]\n[Output too long, truncated]\n```")
        else:
            message.append(f"```\n{output}\n```")

        await ctx.send("".join(message))

    @commands.Cog.listener()
    async def on_disconnect(self):
        start = time()
        try:
            await self.bot.wait_for("ready", timeout=5.0)
        except asyncio.TimeoutError:
            try:
                await self.bot.wait_for("ready", timeout=55.0)
            except asyncio.TimeoutError:
                err = description = f"{self.bot.user.mention} has disconnected.",
            else:
                err = f"{self.bot.user.mention} has reconnected. Downtime: {str(round(time() - start, 2))} seconds."
        else:
            err = f"{self.bot.user.mention} has reconnected. Downtime: {str(round(time() - start, 2))} seconds."
        logger = await self.bot.fetch_webhook(594692503014473729)
        await logger.send(text=err)


async def setup(bot: Bot):
    await bot.add_cog(MetaCog(bot))
