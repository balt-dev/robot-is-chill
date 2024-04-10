import io
import signal
from datetime import datetime
from typing import Literal

import discord
from discord import Member, User
from discord.ext import commands, menus

from .. import constants
from ..types import Bot, Context, Macro, BuiltinMacro
from ..utils import ButtonPages

import re


async def coro_part(func, *args, **kwargs):
    async def wrapper():
        result = func(*args, **kwargs)
        return await result

    return wrapper


async def start_timeout(fn, *args, **kwargs):
    def handler(_signum, _frame):
        raise AssertionError("The command took too long and was timed out.")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(constants.TIMEOUT_DURATION))
    return fn(*args, **kwargs)


class MacroQuerySource(menus.ListPageSource):
    def __init__(
            self, data: list[str]):
        self.count = len(data)
        super().__init__(data, per_page=15)

    async def format_page(self, menu: menus.Menu, entries: list[str]) -> discord.Embed:
        embed = discord.Embed(
            title="Search results",
            # color=menu.bot.embed_color  I think the theme color suits it better.
        ).set_footer(
            text=f"Page {menu.current_page + 1} of {self.get_max_pages()}   ({self.count} entries)",
        )
        while len(entries) > 0:
            field = ""
            for entry in entries[:5]:
                field += f"{entry[:50]}\n"
            embed.add_field(
                name="",
                value=field,
            )
            del entries[:5]
        return embed


class BuiltinMacroQuerySource(menus.ListPageSource):
    def __init__(
            self, data: list[tuple[str, BuiltinMacro]]):
        self.count = len(data)
        super().__init__(data, per_page=5)

    async def format_page(self, menu: menus.Menu, entries: list[tuple[str, BuiltinMacro]]) -> discord.Embed:
        embed = discord.Embed(
            title="Built-in Macros",
        ).set_footer(
            text=f"Page {menu.current_page + 1} of {self.get_max_pages()}   ({self.count} entries)",
        )
        desc = []
        for (name, macro) in entries:
            desc.append(f"**{name}**\n> {macro.description}")
        embed.description = "\n".join(desc)
        return embed


class MacroCommandCog(commands.Cog, name='Macros'):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.group(aliases=["m", "macros"], pass_context=True, invoke_without_command=True)
    async def macro(self, ctx: Context):
        """Front-end for letting users (that means you!) create, edit, and remove variant macros.
    Macros are simply a way of aliasing one or more variants to one name.
    For example, if a macro called `face` with the value `csel-1` exists,
    rendering `baba:m!face` would actually render `baba:csel-1`.
    Arguments can be specified in macros with $<number>. As an example,
    `transpose` aliased to `rot$1:scale$2` would mean that
    rendering `baba:m!transpose/45/2` would give you `baba:rot45:scale2`.
    Important to note, double negatives are valid inputs to variants, so
    something like `baba:scale--2` would give the same as `baba:scale2`.
    $# will be replaced with the amount of arguments given to the macro."""
        await ctx.invoke(ctx.bot.get_command("cmds"), "macro")

    @macro.command(aliases=["r"])
    @commands.is_owner()
    async def refresh(self, ctx: Context):
        """Refreshes the macro database."""
        self.bot.macros = {}
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute("SELECT * from macros")
            for (name, value, description, author) in await cur.fetchall():
                self.bot.macros[name] = Macro(value, description, author)
        return await ctx.reply("Refreshed database.")

    @macro.command()
    @commands.is_owner()
    async def chown(self, ctx: Context, source: Member | User, dest: Member | User):
        """Changes the owner of all macros by a user to another user. Use sparingly."""
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute("""
                UPDATE macros
                SET creator = ?
                WHERE creator == ?
            """, dest.id, source.id)
        return await ctx.reply(f"Done. Moved all macros from account {source} to account {dest}.")

    @macro.command(aliases=["mk", "make"])
    async def create(self, ctx: Context, name: str, value: str, *, description: str = None):
        """Adds a macro to the database."""
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute("SELECT value FROM macros WHERE name == ?", name)
            dname = await cursor.fetchone()
            if dname is not None:
                return await ctx.error(
                    f"Macro of name `{name}` already exists in the database!")
            assert re.search(r"(?<!(?<!\\)\\)/", name) is None, \
                "A macro's name can't have an unescaped slash in it, as it'd clash with parsing arguments."
            # Call the user out on not adding a description, hopefully making them want to add a good one
            assert description is not None, \
                "A description is _required_. Please describe what your macro does understandably!"
            command = "INSERT INTO macros VALUES (?, ?, ?, ?);"
            args = (name, value, description, ctx.author.id)
            self.bot.macros[name] = Macro(value, description, ctx.author.id)
            await cursor.execute(command, args)
            return await ctx.reply(f"Successfully added `{name}` to the database, aliased to `{value}`!")

    @macro.command(aliases=["e"])
    async def edit(self, ctx: Context, name: str, attribute: Literal["value", "description", "name"], *, new: str):
        """Edits a macro. You must own said macro to edit it."""
        assert name in self.bot.macros, f"Macro `{name}` isn't in the database!"
        if attribute == "name":
            assert new in self.bot.macros, f"Macro `{new}` is already in the database!"
        async with self.bot.db.conn.cursor() as cursor:
            if not await ctx.bot.is_owner(ctx.author):
                await cursor.execute("SELECT name FROM macros WHERE name == ? AND creator == ?", name, ctx.author.id)
                check = await cursor.fetchone()
                assert check is not None, "You can't edit a macro you don't own, silly."
            # NOTE: I know I shouldn't use fstrings with execute, but it won't allow me to specify a row name with ?.
            await cursor.execute(f"UPDATE macros SET {attribute} = ? WHERE name == ?", new, name)
        if attribute == "name":
            mac = self.bot.macros[name]
            del self.bot.macros[name]
            self.bot.macros[new] = mac
        else:
            setattr(self.bot.macros[name], attribute, new)
        return await ctx.reply(f"Edited `{name}`'s {attribute} to be `{new}`.")

    @macro.command(aliases=["rm", "remove", "del"])
    async def delete(self, ctx: Context, name: str):
        """Deletes a macro. You must own said macro to delete it."""
        assert name in self.bot.macros, f"Macro `{name}` already isn't in the database!"
        async with self.bot.db.conn.cursor() as cursor:
            if not await ctx.bot.is_owner(ctx.author):
                await cursor.execute("SELECT name FROM macros WHERE name == ? AND creator == ?", name, ctx.author.id)
                check = await cursor.fetchone()
                assert check is not None, "You can't delete a macro you don't own, silly."
            await cursor.execute(f"DELETE FROM macros WHERE name == ?", name)
        del self.bot.macros[name]
        return await ctx.reply(f"Deleted `{name}`.")

    @macro.command(aliases=["?", "list", "query"])
    async def macro_search(self, ctx: Context, *, pattern: str = '.*'):
        """Searches the database for macros by name."""
        author = None
        if match := re.search(r"--?a(?:uthor)?=(\S+)", pattern):
            author = match.group(1)
            try:
                author = await commands.UserConverter().convert(ctx, author)
            except commands.errors.UserNotFound:
                author = await commands.MemberConverter().convert(ctx, author)
            pattern = pattern[:match.start()] + pattern[match.end():]

        pattern = pattern.strip()
        print(pattern)
        async with self.bot.db.conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT name FROM macros 
                WHERE name REGEXP :name
                AND (
                    :author IS NULL OR creator == :author
                )
                """,
                {
                    "name": pattern,
                    "author": None if author is None else author.id
                }
            )
            names = [name for (name,) in await cursor.fetchall()]
        return await ButtonPages(MacroQuerySource(sorted(names))).start(ctx)

    @macro.command(aliases=["x", "run"])
    async def execute(self, ctx: Context, *, macro: str):
        """Executes some given macroscript and outputs its return value."""
        try:
            macros = ctx.bot.macros | {}
            debug_info = False
            if match := re.match(r"^\s*--?d(?:ebug|bg)?", macro):
                debug_info = True
                macro = macro[match.end():]
            while match := re.match(r"^\s*--?mc=((?:(?!(?<!\\)\|).)*)\|((?:(?!(?<!\\)\s).)*)", macro):
                macros[match.group(1)] = Macro(value=match.group(2), description="<internal>", author=-1)
                macro = macro[match.end():]

            def parse():
                nonlocal debug_info
                return ctx.bot.macro_handler.parse_macros(macro.strip(), debug_info)

            macro, debug = await start_timeout(parse)

            message, files = "", []

            if macro is not None:
                if len(macro) > 1900:
                    out = io.BytesIO()
                    out.write(bytes(macro, 'utf-8'))
                    out.seek(0)
                    files.append(discord.File(out, filename=f'output-{datetime.now().isoformat()}.txt'))
                    message = 'Output:'
                else:
                    message = f'Output: ```\n{macro.replace("```", "``ˋ")}\n```'
            else:
                message = "Error occurred while parsing macro. See debug info for details."
            if debug is not None:
                debug_file = "\n".join(debug)
                out = io.BytesIO()
                out.write(bytes(debug_file, 'utf-8'))
                out.seek(0)
                files.append(discord.File(out, filename=f'debug-{datetime.now().isoformat()}.txt'))
            return await ctx.reply(message, files=files)
        finally:
            signal.alarm(0)

    @macro.command(aliases=["i", "get"])
    async def info(self, ctx: Context, name: str):
        """Gets info about a specific macro."""
        assert name in self.bot.macros, f"Macro `{name}` isn't in the database!"
        macro = self.bot.macros[name]
        emb = discord.Embed(
            title=name
        )
        emb.add_field(
            name="",
            value=macro.description
        )
        emb.add_field(
            name="Value",
            value=f"```\n{macro.value.replace('`', 'ˋ')}```",
            inline=False
        )
        user = await ctx.bot.fetch_user(macro.author)
        emb.set_footer(text=f"{user.name}#{user.discriminator}",
                       icon_url=user.avatar.url if user.avatar is not None else
                       f"https://cdn.discordapp.com/embed/avatars/{int(user.discriminator) % 5}.png")
        try:
            await ctx.reply(embed=emb)
        except discord.errors.HTTPException:
            emb.set_field_at(
                1,
                name="Value",
                value=f"_Value too long to embed. It has been attached as a text file._",
                inline=False
            )
            buf = io.BytesIO()
            buf.write(macro.value.encode("utf-8", "ignore"))
            buf.seek(0)
            await ctx.reply(embed=emb, file=discord.File(buf, filename=f"{name}-value.txt"))

    @macro.command(aliases=["b"])
    async def builtins(self, ctx: Context):
        """Lists off the builtin macros of the bot."""
        return await ButtonPages(BuiltinMacroQuerySource(
            list(ctx.bot.macro_handler.builtins.items())
        )).start(ctx)


async def setup(bot: Bot):
    await bot.add_cog(MacroCommandCog(bot))
