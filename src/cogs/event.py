import discord
from discord.ext import commands

import time

from ..types import Bot, Context

class EventCog(commands.Cog, name='Events'):
	def __init__(self, bot: Bot):
		self.bot = bot

	@commands.Cog.listener()
	async def on_guild_join(self,guild):
		async with self.bot.db.conn.cursor() as cur:
			await cur.execute(f'''INSERT INTO ServerActivity(id, timestamp) VALUES(?,0);''',(guild.id))
		return #implement leaving when above 70 servers

	@commands.Cog.listener()
	async def on_guild_remove(self,guild):
		async with self.bot.db.conn.cursor() as cur:
			await cur.execute(f'''DELETE FROM ServerActivity WHERE id LIKE ?;''',(guild.id))

	async def bot_check(self,ctx):
		for sub, id in [('channel',ctx.channel.id),('user',ctx.author.id)]:
			async with self.bot.db.conn.cursor() as cur:
				await cur.execute(f'''SELECT DISTINCT * FROM BLACKLISTED{sub}S WHERE id = {id}''') #this is risky but both values are guaranteed ints so i think it's fine?
				if len(await cur.fetchall()):
					return False
		try:
			async with self.bot.db.conn.cursor() as cur:
				await cur.execute(f'''REPLACE INTO ServerActivity(id, timestamp) VALUES(?,?);''',(ctx.guild.id,time.time()))
		except AttributeError:
			pass
		return True
	
def setup(bot: Bot):
	bot.add_cog(EventCog(bot))