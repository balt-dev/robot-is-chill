from __future__ import annotations
import asyncio
from io import BytesIO

import json
import logging
import os
import pathlib
import random
import sys
from datetime import datetime
from typing import Iterable

import discord
from discord.ext import commands
from PIL import Image

import auth
import config
import webhooks
from src.db import Database

class Context(commands.Context):
	async def error(self, msg: str, embed: discord.Embed | None = None) -> discord.Message:
		await self.message.add_reaction("\u26a0\ufe0f")
		if embed is not None:
			return await self.reply(msg,embed=embed)
		else:
			return await self.reply(msg)

	async def send(self, content: str = "", embed: discord.Embed | None = None, **kwargs):
		content = str(content)
		if len(content) > 2000:
			msg = " [...] \n\n (Character limit reached!)"
			content = content[:2000-len(msg)] + msg
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
		return await self.send(*args, **kwargs)

class Bot(commands.Bot):
	'''Custom bot class :)'''
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
		self.db = Database()
		self.db_path = db_path
		self.config = config.__dict__
		sys.stdout = BytesIO()
		super().__init__(*args, **kwargs)
		sys.stdout = sys.__stdout__
		# has to be after __init__
		async def gather_cogs():
			asyncio.gather(*(self.load_extension(cog, package='ROBOT') for cog in cogs))
		asyncio.run(gather_cogs())

	async def get_context(self, message: discord.Message) -> Context:
		return await super().get_context(message, cls=Context)

	async def close(self) -> None:
		await self.db.close()
		await super().close()

	async def on_ready(self) -> None:
		await self.db.connect(self.db_path)
		print(f"Logged in as {self.user}!")
		await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="commands..."))

logging.basicConfig(filename=config.log_file, level=logging.INFO)
	
# Establishes the bot
bot = Bot(
	# Prefixes
	commands.when_mentioned_or(*config.prefixes) if config.trigger_on_mention else config.prefixes,
	# Other behavior parameters
	case_insensitive=True, 
	activity=discord.Game(name=config.activity), 
	description=config.description,
	# Never mention roles, @everyone or @here
	allowed_mentions=discord.AllowedMentions(everyone=False, roles=False),
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
	webhook_id=webhooks.webhook_id,
	prefixes=config.prefixes,
	db_path=config.db_path
)

@bot.event
async def on_command(ctx):
	webhook = await bot.fetch_webhook(webhooks.logging_id)
	embed = discord.Embed(
		title = None,
		description = (ctx.message.content),
		color=config.logging_color
	)
	embed.set_author(name=f'{ctx.author.name}#{ctx.author.discriminator}'[:32], url=None, icon_url=ctx.author.avatar.url if ctx.author.avatar else None)
	embed.set_footer(text=str(ctx.author.id))
	await webhook.send(embed=embed)

bot.run(auth.token, log_handler=None)
sys.exit(bot.exit_code)