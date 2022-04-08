from __future__ import annotations
from io import BytesIO
from pathlib import Path

import re
from os import listdir
import os.path
from typing import Any, Sequence, Literal
import numpy as np
import cv2

from src.tile import RawTile
from src.db import CustomLevelData, LevelData, TileData
import zipfile 

import glob
import time
import discord
from discord.ext import commands, menus
from PIL import Image, ImageFont, ImageDraw

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

	@commands.cooldown(5, 8, type=commands.BucketType.channel)
	@commands.command(name="undo")
	async def undo(self, ctx: Context):
		'''Deletes the last message sent from the bot.'''
		await ctx.trigger_typing()
		await ctx.message.delete()
		h = ctx.channel.history(limit=20)
		async for m in h:
			if m.author.id == self.bot.user.id and m.attachments:
				try:
					reply = await ctx.channel.fetch_message(m.reference.message_id)
					if reply.author == ctx.message.author:
						await reply.delete()
						await m.delete()
						await ctx.send('Removed message.', delete_after=3.0)
						return
				except:
					pass
		await ctx.error('None of your commands were found in the last `20` messages.')

	@commands.command()
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def flags(self, ctx: Context, *, query: str = False):
		'''Gets help on a render flag.'''
		#i wish i could update this manually but alas i can't without some EXTREME jank.
		flaglist = (("-b[=x/y]","Sets the background to the palette index specified, or 0/0 if not.")
					,("-p=string","Sets the palette to the specified palette. (Try `=search type:palette`?)")
					,("-r[=name]","Outputs the render as a raw file, ready to put in Baba Is You. If you specify a name, the tile will be called said name.")
					,("--letter","Makes text smaller than 3 long render as big letters by default.")
					,("--comment=anything","Just a comment. Does nothing.")
					,("-f=(comma separated values in [1,2,3])","Chooses which wobble frames to display.")
					,("-c","Prepends the render in the message you replied to in the render, or the last message you sent in the channel if there is none.")
					,("-speed=float","Multiplies the frame delta of the image by the reciprocal of the specified number.")
					,("-g=string","Applies the specified string as a variant to every tile in the image")
					,("-co","Disables randomization of tile wobble.")
					,("-gr=int/int","Adds a grid overlay, dimensions being the specified numbers.")
					,("-s=float","Upscales the image pre-processing.")
					,("-m=float","Upscales the image post-processing.")
					,("-sw","Rotates the stack axis along the x axis. May be useful for debugging large stacks.")
					,("-l=int[/int]","Shows only the stack layer or range of stack layers specified. First is inclusive, second is exclusive.")
					,("-tb","Makes tiles connect with the edge of the image.")
					,("-crop=int/int/int/int","Crops the render to the specified pixels.")
					,("-pad=int/int/int/int","Pads the render with the specified pixels.")
					,("-v","Gives more information about the render.")
					,("-f=<gif|png>","Sets the format of the render output.")
					#,("","")
		)
				   
		emb = discord.Embed(
			title=f"Flag list",
			colour=self.bot.embed_color,
			description="\n".join([f'`{a}`: {b}' for a,b in flaglist]))
		return await ctx.reply(embed=emb, mention_author=False)
			
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
	@commands.command()
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def variants(self, ctx: Context):
		'''Alias for =search type:variant'''
		await self.search(ctx,query='type:variant')

	@commands.command()
	@commands.cooldown(4, 8, type=commands.BucketType.channel)
	async def grabtile(self, ctx: Context, sprite_name: str):
		'''Gets the files for a specific tile from the bot'''
		#
		async with self.bot.db.conn.cursor() as cur:
			await ctx.trigger_typing()
			result = await cur.execute('SELECT DISTINCT source FROM tiles WHERE name = (?)',sprite_name)
			try:
				source = (await result.fetchone())[0]
			except:
				return await ctx.error(f'Tile {sprite_name} not found!')
			files = glob.glob(f'data/sprites/{source}/{sprite_name}_*.png')
			zipped_files = BytesIO()
			with zipfile.ZipFile(zipped_files, "x") as f_a:
				for f_b in files:
					with open(f_b,'rb') as f_c:
						f_a.writestr(os.path.basename(os.path.normpath(f_b)), f_c.read())
				with open(f_b,'rb') as f_c:
						f_a.writestr('attributes.txt', '''f_c.read()''')
			zipped_files.seek(0)
			return await ctx.send(f'Files for sprite `{sprite_name}` from `{source}`:',files=[discord.File(zipped_files,filename=f'{source}-{sprite_name}.zip')])
		
	@commands.cooldown(5, 8, type=commands.BucketType.channel)
	@commands.command(name="overlays")
	async def overlays(self, ctx: Context):
		"""Lists all available overlays."""
		await ctx.send(embed=discord.Embed(
			title="Available overlays",
			colour=self.bot.embed_color,
			description="\n".join(f"{overlay[:-4]}" for overlay in listdir('data/overlays/'))))

	@commands.command(name="seedcracker", aliases=['cracker'])
	async def send_seedcracker(self, ctx: Context):
		'''Sends the seedcracker program as a file.'''
		with open('src/seedcracker.py',mode='rb') as f:
			await ctx.reply(file=discord.File(f,filename='seedcracker.py'))

	@commands.cooldown(5, 8, type=commands.BucketType.channel)
	@commands.command(name="palette", aliases=['pal'])
	async def show_palette(self, ctx: Context, palette: str = 'default', raw: str = None):
		'''Displays palette image.

		This is useful for picking colors from the palette.'''
		if palette == '-r' or palette == '--raw':
			palette = 'default'
			raw = '--raw'
		assert palette.find('/') == -1 and palette.find('\\') == -1, 'No looking at the host\'s hard drive, thank you very much.'
		try:
			img = Image.open(f"data/palettes/{palette}.png")
		except FileNotFoundError:
			return await ctx.error(f"The palette `{palette}` could not be found.")
		if raw and (raw == '-r' or raw == '--raw'):  
			zipbuf = BytesIO()
			file = zipfile.PyZipFile(zipbuf, "x")
			palbuf = BytesIO()
			img.save(palbuf, "PNG")
			file.writestr(f"{palette}.png", palbuf.getvalue())
			file.close()
			zipbuf.seek(0)
			palbuf.seek(0)
			zipf = discord.File(zipbuf, filename=f"{palette}.zip")
			palf = discord.File(palbuf, filename=f"{palette}.png")
			await ctx.reply(f"Palette `{palette}`:", files=[palf,zipf])
		else:
			txtwid, txthgt = img.size
			img = img.resize(
				(img.width * constants.PALETTE_PIXEL_SIZE, img.height * constants.PALETTE_PIXEL_SIZE), 
				resample=Image.NEAREST
			)
			font = ImageFont.truetype("data/04b03.ttf", 16)
			draw = ImageDraw.Draw(img)
			for y in range(txthgt):
				for x in range(txtwid):
					try:
						n = img.getpixel((x * constants.PALETTE_PIXEL_SIZE, (y * constants.PALETTE_PIXEL_SIZE)))
						if (n[0]+n[1]+n[2])/3 > 128:
							draw.text((x * constants.PALETTE_PIXEL_SIZE, (y * constants.PALETTE_PIXEL_SIZE)-2), str(x)+","+str(y), (1,1,1,255), font, layout_engine=ImageFont.LAYOUT_BASIC)
						else:
							draw.text((x * constants.PALETTE_PIXEL_SIZE, (y * constants.PALETTE_PIXEL_SIZE)-2), str(x)+","+str(y), (255,255,255,255), font, layout_engine=ImageFont.LAYOUT_BASIC)
					except:
						pass
			buf = BytesIO()
			img.save(buf, format="PNG")
			buf.seek(0)
			file = discord.File(buf, filename=f"palette_{palette}.png")
			await ctx.reply(f"Palette `{palette}`:", file=file)

	@commands.cooldown(5, 8, type=commands.BucketType.channel)
	@commands.command(name="blacklist")
	@commands.has_permissions(administrator=True)
	async def blacklist(self, ctx: Context, *, args: str):
		'''Set up a blacklist of channels or users. Channels are server moderator only, users are bot owner only.'''
		try:
			sub_command, mode, id = args.split(' ')
		except ValueError:
			assert False, 'Error! Not enough arguments'
		channels = []
		for channel in ctx.channel.guild.channels:
			if str(channel.type) == 'text':
				channels.append(channel.id)
		assert sub_command in ['channel','user'], 'Error! `sub-command` needs to be one of `channel,user`'
		assert mode in ['add','remove'], 'Error! `move` needs to be one of `add,remove`'
		try:
			id = int(id)
		except:
			assert False, 'Error! `id` needs to be of type `int`'
		assert (
			sub_command == 'user' and 
			commands.is_owner()
		) or (
			sub_command == 'channel' and 
			id in channels
		), 'You\'re not authorized to do that.'
		async with self.bot.db.conn.cursor() as cur:
			if mode == 'add':
				await cur.execute(f'''INSERT INTO blacklisted{sub_command}s
								VALUES ({id})''')
				await ctx.reply(f'Added `{sub_command}` of id `{id}` to the blacklist.')
			elif mode == 'remove':
				await cur.execute(f'''DELETE FROM blacklisted{sub_command}s
								WHERE id={id}''')
				await ctx.reply(f'Removed `{sub_command}` of id `{id}` from the blacklist.')
		
				
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


def setup(bot: Bot):
	bot.add_cog(UtilityCommandsCog(bot))
