import discord

activity = "Booting..."
description = "*An entertainment bot for rendering levels and custom scenes based on the indie game Baba Is You.*"
prefixes = ["=", "robot is ", "ROBOT IS "]
trigger_on_mention = True
embed_color = discord.Color(12877055)
logging_color = 0xffffff
auth_file = "config/auth.json"
log_file = "log.txt"
db_path = "robot.db"
cogs = [
	"src.cogs.owner",
	"src.cogs.global",
	"src.cogs.meta",
	"src.cogs.errorhandler",
	"src.cogs.reader",
	"src.cogs.render",
	"src.cogs.variants",
	"src.cogs.utilities",
	"src.cogs.generator",
	"src.cogs.event",
	"src.cogs.flags",
	"src.cogs.macro_commands",
	"src.cogs.macros",
	"jishaku"
]
danger_mode = False
debug = False
owner_only_mode = [False,'']
