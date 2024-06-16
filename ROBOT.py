from __future__ import annotations

import asyncio
import glob

import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Coroutine

import discord
from PIL import Image
from discord import app_commands
from discord.ext import commands

import auth
import config
import webhooks
from src.types import Macro
from src.db import Database

from numpy import set_printoptions as numpy_set_printoptions


class Context(commands.Context):
    silent: bool = False
    ephemeral: bool = False

    async def error(self, msg: str, embed: discord.Embed | None = None) -> Coroutine[discord.Message]:
        try:
            await self.message.add_reaction("\u26a0\ufe0f")
        except discord.errors.NotFound:
            pass
        if embed is not None:
            return await self.reply(msg, embed=embed)
        else:
            return await self.reply(msg)

    async def send(self, content: str = "", embed: discord.Embed | None = None, **kwargs):
        content = str(content)
        kwargs['ephemeral'] = self.ephemeral
        kwargs['silent'] = self.silent
        if len(content) > 2000:
            msg = " [...] \n\n (Character limit reached!)"
            content = content[:2000 - len(msg)] + msg
        if embed is not None:
            if content:
                return await super().send(content, embed=embed, **kwargs)
            return await super().send(embed=embed, **kwargs)
        elif content:
            return await super().send(content, embed=embed, **kwargs)
        return await super().send(**kwargs)

    async def reply(self, *args, mention_author: bool = False, **kwargs):
        kwargs['mention_author'] = mention_author
        kwargs['reference'] = self.message
        kwargs['ephemeral'] = self.ephemeral
        return await self.send(*args, **kwargs)


class Bot(commands.Bot):
    """Custom bot class :)"""
    db: Database

    def __init__(
            self,
            *args,
            cogs: list[str],
            embed_color: discord.Color,
            webhook_id: int,
            prefixes: list[str],
            db_path: str,
            **kwargs
    ):
        self.started = datetime.utcnow()
        self.loading = True
        self.exit_code = 0
        self.embed_color = embed_color
        self.webhook_id = webhook_id
        self.prefixes = prefixes
        self.db = Database(self)
        self.db_path = db_path
        self.config = config.__dict__
        self.renderer = None
        self.flags = None
        self.variants = None
        self.palette_cache = {}
        self.macros = {}
        self.baba_loaded = True
        for path in glob.glob("data/palettes/*.png"):
            with Image.open(path) as im:
                self.palette_cache[Path(path).stem] = im.convert("RGBA").copy()
        numpy_set_printoptions(
            threshold=sys.maxsize,
            linewidth=sys.maxsize
        )
        super().__init__(*args, **kwargs)
        self.remove_command('help')
        # has to be after __init__
        async def gather_cogs():
            await asyncio.gather(*(self.load_extension(cog, package='ROBOT') for cog in cogs))

        asyncio.run(gather_cogs())

    async def get_context(self, message: discord.Message, **kwargs) -> Context:
        return await super().get_context(message, cls=Context)

    async def close(self) -> None:
        await self.db.close()
        await super().close()

    async def on_ready(self) -> None:
        await self.db.connect(self.db_path)
        print("Loading macros...")
        async with self.db.conn.cursor() as cur:
            await cur.execute("SELECT * from macros")
            for (name, value, description, author) in await cur.fetchall():
                self.macros[name] = Macro(value, description, author)
        print(f"Logged in as {self.user}!")
        if bot.baba_loaded:
            await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="commands..."))
        else:
            await bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.playing,
                    name="Still being set up..."
                ),
                status=discord.Status.do_not_disturb
            )

    async def is_owner(self, user: discord.User):
        if user.id == 280756504674566144:  # Implement your own conditions here
            return True

        # Else fall back to the original
        return await super().is_owner(user)


discord.utils.setup_logging()

# Establishes the bot
bot = Bot(
    # Prefixes
    commands.when_mentioned_or(*config.prefixes) if config.trigger_on_mention else config.prefixes,
    # Other behavior parameters
    case_insensitive=True,
    activity=discord.Game(name=config.activity),
    description=config.description,
    # Never mention roles, @everyone or @here
    allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False),
    # Only receive message and reaction events
    intents=discord.Intents(messages=True, reactions=True, guilds=True, message_content=True),
    # Disable the member cache
    member_cache_flags=discord.MemberCacheFlags.none(),
    # Disable the message cache
    max_messages=None,
    # Don't chunk guilds
    chunk_guilds_at_startup=False,
    # custom fields
    cogs=config.cogs,
    embed_color=config.embed_color,
    webhook_id=webhooks.logging_id,
    prefixes=config.prefixes,
    db_path=config.db_path
)


@bot.event
async def on_command(ctx):
    try:
        webhook = await bot.fetch_webhook(webhooks.logging_id)
        ctx: Context
        embed = discord.Embed(
            description=ctx.message.content,
            color=config.logging_color)
        embed.set_author(name=f'{ctx.author.name}'[:32],
                         icon_url=ctx.author.avatar.url if ctx.author.avatar else None)
        if not isinstance(ctx.channel, discord.DMChannel):
            embed.set_footer(text=f"{ctx.message.guild.name} ({ctx.message.guild.id})", icon_url=ctx.message.guild.icon.url)
        await webhook.send(embed=embed)
    except Exception as e:
        warnings.warn("\n".join(traceback.format_exception(e)))


bot.run(auth.token, log_handler=None)
sys.exit(bot.exit_code)
