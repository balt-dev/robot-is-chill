from __future__ import annotations

import asyncio
import itertools
from datetime import datetime
from subprocess import PIPE, STDOUT, TimeoutExpired, run
from time import time

import discord
from discord.ext import commands

from ..types import Bot, Context


# Custom help command implementation
class PrettyHelpCommand(commands.DefaultHelpCommand):

    def __init__(self, embed_color: int, **kwargs):
        self.embed_color = embed_color
        super().__init__(**kwargs)

    async def send_pages(self, note: str = "", inline: bool = False):
        # Overwrite the send method to send each page in an embed instead
        destination = self.get_destination()

        for page in self.paginator.pages:
            formatted = discord.Embed(color=self.embed_color)
            
            split = page.split("**")
            if len(split) == 1:
                formatted.description = page
            else:
                split = iter(split)
                header = next(split)
                formatted.description = header

                for segment in split:
                    if segment.strip() == "":
                        continue
                    
                    title = segment
                    content = next(split)

                    formatted.add_field(name=title, value=content, inline=inline)

            formatted.set_footer(text=note)
            
            await destination.send(embed=formatted)

    def add_indented_commands(self, commands: list[commands.Command], *, heading: str, max_size: int | None = None):
        if not commands:
            return

        self.paginator.add_line()    
        self.paginator.add_line(heading)
        max_size = max_size or self.get_max_size(commands)

        for command in commands:
            name = command.name
            self.paginator.add_line(self.shorten_text("\u200b  `" + name + "`"))
            self.paginator.add_line(self.shorten_text(command.short_doc))

    async def send_bot_help(self, mapping):
        ctx = self.context
        bot = ctx.bot

        if bot.description:
            # <description> portion
            self.paginator.add_line(bot.description, empty=True)

        def get_category(command, *, no_category: str = f'\u200b**{self.no_category}**') -> str:
            cog = command.cog
            return "**" + cog.qualified_name + '**' if cog is not None else no_category

        filtered = await self.filter_commands(bot.commands, sort=True, key=get_category)
        max_size = self.get_max_size(filtered)
        to_iterate = itertools.groupby(filtered, key=get_category)

        # Now we can add the commands to the page.
        for category, commands in to_iterate:
            commands = sorted(commands, key=lambda c: c.name) if self.sort_commands else list(commands)
            self.add_indented_commands(commands, heading=category, max_size=max_size)

        note = self.get_ending_note()

        await self.send_pages(note=note, inline=True)

    def get_ending_note(self) -> str:
        """Returns help command's ending note. This is mainly useful to override for i18n purposes."""
        command_name = self.invoked_with
        return "Type {0}{1} command for more info on a command.".format(self.clean_prefix, command_name)

    def get_command_signature(self, command: commands.Command) -> str:
        parent = command.full_parent_name
        if len(command.aliases) > 0:
            aliases = '|'.join(command.aliases)
            fmt = '[%s|%s]' % (command.name, aliases)
            if parent:
                fmt = parent + ' ' + fmt
            alias = fmt
        else:
            alias = command.name if not parent else parent + ' ' + command.name

        return '`%s%s %s`' % (self.clean_prefix, alias, command.signature)

class MetaCog(commands.Cog, name="Other Commands"):
    def __init__(self, bot: Bot):
        self.bot = bot
        self._original_help_command = bot.help_command
        # Sets up the help command
        bot.help_command = PrettyHelpCommand(bot.embed_color, **dict(paginator=commands.Paginator(prefix="", suffix="")))
        bot.help_command.cog = self

    # Check if the bot is loading
    async def cog_check(self, ctx: Context):
        return not self.bot.loading

    @commands.command(aliases=["info"])
    @commands.cooldown(5, 8, commands.BucketType.channel)
    async def about(self, ctx: Context):
        """
        Displays bot information.
        """
        about_embed = discord.Embed(
            title="About This Bot", 
            type="rich", 
            colour=self.bot.embed_color, 
            description="\n".join([
                f"{ctx.me.name} - Bot for Discord based on the indie game Baba Is You. "
                "Written by RocketRace#0798 (and modded by balt#6423) using the [discord.py](https://github.com/Rapptz/discord.py) library."
            ])
        )
        about_embed.add_field(name="Links", value="[GitHub repository](https://github.com/balt-is-you-and-shift/robot-is-chill)\n"
        )
        ut = datetime.utcnow() - self.bot.started
        stats = "".join([
            f"\nGuilds: {len(self.bot.guilds)}",
            f"\nChannels: {sum(len(g.channels) for g in self.bot.guilds)}",
            f"\nUptime: {ut.days}d {ut.seconds // 3600}h {ut.seconds % 3600 // 60}m {ut.seconds % 60}s"
        ])
        about_embed.add_field(name="Statistics", value=stats)
        about_embed.add_field(name="Valid Prefixes", value="\n".join([
            "`" + p + "`" for p in self.bot.prefixes
        ]))
        about_embed.add_field(name="Credits", value="\n".join([
            "[Baba Is Bookmark](https://baba-is-bookmark.herokuapp.com/) (custom level database) by SpiccyMayonnaise",
            "[Baba Is Hint](https://www.keyofw.com/baba-is-hint/) (hints for levels) by keyofw",
        ]))
        await ctx.send(embed=about_embed)
    
    @commands.command(aliases=["pong"])
    @commands.cooldown(5, 8, commands.BucketType.channel)
    async def ping(self, ctx: Context):
        '''Returns bot latency.'''
        await ctx.send(f"Latency: {round(self.bot.latency, 3)} seconds")

    @commands.command()
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def invite(self, ctx: Context):
        '''Links for the bot support server'''
        msg = discord.Embed(colour=self.bot.embed_color, title="Invite Link", description="[Click to invite](https://discord.com/api/oauth2/authorize?client_id=753421978324566046&permissions=8&scope=bot)"
        )
        await ctx.send(embed=msg)
    
    @commands.command(aliases=["interpret"])
    @commands.cooldown(5, 8, type=commands.BucketType.channel)
    async def babalang(self, ctx: Context, program: str, *program_input: str):
        '''Interpret a [Babalang v1.1.1](https://esolangs.org/wiki/Babalang) program.
        
        The first argument must be the source code for the program, escaped in quotes:
        
        * e.g. `"baba is group and word and text"`

        The second argument is the optional input, also escaped in quotes:

        * e.g. `"foo bar"`

        Both arguments can be multi-line. The input argument will be automatically padded 
        with trailing newlines as necessary.
        '''
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
                process = run(
                    ["./src/babalang",  "-c", f"'{program}'"], 
                    stdout=PIPE,
                    stderr=STDOUT,
                    timeout=1.0,
                    input=prog_input.encode("utf-8", "ignore"),
                )
                if process.stdout is not None:
                    return (process.returncode, process.stdout[:1000].decode("utf-8", "replace"))
                else:
                    return (process.returncode, "")
            except TimeoutExpired as timeout:
                if timeout.output is not None:
                    if isinstance(timeout.output, bytes):
                        return (None, timeout.output[:1000].decode("utf-8", "replace"))
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
            message.append(f"The program terminated with return code `{return_code}`:\n")

        if not output:
            message.append("```\n[No output]\n```")
        elif too_long:
            message.append(f"```\n{output} [...]\n[Output too long, truncated]\n```")
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
                err = discord.Embed(
                    title="Disconnect", 
                    type="rich", 
                    description=f"{self.bot.user.mention} has disconnected.", 
                    color=0xff8800)
            else:
                err = discord.Embed(
                    title="Reconnected",
                    type="rich",
                    description=f"{self.bot.user.mention} has reconnected. Downtime: {str(round(time() - start, 2))} seconds.",
                    color=0xffaa00
                )
        else: 
            err = discord.Embed(
                title="Resumed", 
                type="rich", 
                description=f"{self.bot.user.mention} has reconnected. Downtime: {str(round(time() - start, 2))} seconds.", 
                color=0xffff00
            )
        logger = await self.bot.fetch_webhook(594692503014473729)
        await logger.send(embed=err)
    
    def cog_unload(self):
        self.bot.help_command = self._original_help_command

def setup(bot: Bot):
    bot.add_cog(MetaCog(bot))
