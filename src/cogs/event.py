from discord.ext import commands

import time

from ..types import Bot

import config


class EventCog(commands.Cog, name='Events'):
    def __init__(self, bot: Bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute(f'''INSERT INTO ServerActivity(id, timestamp) VALUES(?,0);''', (guild.id))
        return  # implement leaving when above 70 servers

    @commands.Cog.listener()
    async def on_guild_remove(self, guild):
        async with self.bot.db.conn.cursor() as cur:
            await cur.execute(f'''DELETE FROM ServerActivity WHERE id LIKE ?;''', (guild.id))

    async def bot_check(self, ctx):
        async with self.bot.db.conn.cursor() as cur:
            # this is risky but guaranteed int so i think it's fine?
            await cur.execute(f'SELECT DISTINCT * FROM BLACKLISTEDUSERS WHERE id = ?;', ctx.author.id)
            if len(await cur.fetchall()):
                dm_channel = await self.bot.create_dm(ctx.author)
                await dm_channel.send('''You can\'t use this bot, as you have been blacklisted. This may be for a few reasons:
> You did something bad (like pedophilia) and the bot owner heard about it
> You spammed the bot
> Other reasons
If you feel this was unjustified, please DM the bot owner.''')
                return False
        try:
            async with self.bot.db.conn.cursor() as cur:
                await cur.execute(f'''REPLACE INTO ServerActivity(id, timestamp) VALUES(?,?);''', (ctx.guild.id, time.time()))
        except AttributeError:
            pass
        if self.bot.config['owner_only_mode'][0] and ctx.author.id != self.bot.owner_id:
            await ctx.error(f'The bot is currently in owner only mode. The owner specified this reason:\n`{config.owner_only_mode[1]}`')
            return False
        return True


async def setup(bot: Bot):
    await bot.add_cog(EventCog(bot))
