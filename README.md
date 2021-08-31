# About

A fun Discord bot based on the indie game [Baba Is You](https://store.steampowered.com/app/736260/Baba_Is_You/) (by Arvi Teikari). This bot was written with the [discord.py](https://discordpy.readthedocs.io/en/latest/) library for Python 3.x.

# Functionality

A list of all commands can be seen using the `help` command. (By default, the bot has a prefix of `+`.)

This bot primarily features rendering and animation of
tiles in *Baba Is You* (through the `tile` and `rule` commands), 
as well as *Baba Is You* levels (using the `level` command).

*Example command:*

![Example command](https://cdn.discordapp.com/attachments/420095557231443988/596606587800387594/unknown.png)

*Example output:*

![Example command output](https://cdn.discordapp.com/attachments/420095557231443988/596606636215500816/unknown.png)

**NOTE: Output is in the form of an animated gif. Try the commands yourself for the best experience.**

# Invite

[Invite the bot to your server!](https://discordapp.com/api/oauth2/authorize?client_id=480227663047294987&scope=bot&permissions=388160)

# Support server

Leave any suggestions, bug reports or questions in the official [support Discord server](https://discord.gg/rMX3YPK).

# To Host This Yourself

Please follow the terms of the license!

Install the requirements: `pip install -r requirements.txt`.

Run the bot using `python3 ROBOT.py`.

(The bot may not work properly on Windows, as it makes use of some unix-ish shell commands for the convenience of the programmer.)

## Required files

Bot configuration is in `config.py`. It contains the following values:

* `activity`: `str` - A "playing" message to set at login.
* `description`: `str` - A description to use in the help command.
* `prefixes`: `list[str]` - A list of strings that can be used to trigger commands.
* `trigger_on_mention`: `bool` - Whether or not bot @mentions will behave as a command prefix.
* `webhook_id`: `int` - The ID of a webhook to report command errors to. Requires the `manage webhooks` permission in the webhook's channel.
* `embed_color`: `discord.Color` - The color of embedded messages.
* `log_file`: `str` - The file to report logs to.
* `cogs`: `list[str]` - A list of strings -- cogs to load into the bot.

In addition, authentication information should be placed in `auth.py`:

* `token`: `str` - The bot token.

If the bot complains about missing files or directories in `cache/` or `target/`, create them.

## Setup commands (bot owner only)

`<>` denotes a required argument, and `[]` denotes an optional argument.

* `loaddata` Collects tile metadata from `values.lua`, `editor_objectlist.lua`, `data/worlds/baba/*.ld` files and `data/custom/*.json` files, and saves it to disk. The following commands are also available, but it is **strongly recommended** to use `loaddata`.
* `loadletters` Scrapes individual letter sprites from image sprites in `data/sprites/*`, as well as pre-made letters from `data/letters/**/*` and places the results in `target/letters/`.
* `loadmap <world_name> <level_id>` Reads and renders an animated GIF of the provided level. `world_name` should be `baba` in most cases. The level metadata is untouched. Useful for re-rendering levels changed in an update without re-doing everything.
* `loadmaps` Reads and renders every single level in `data/levels/baba/`. Also collects metadata.

* To load tile data, run the `loaddata` command. To load letter data (for custom text), run the `loadletters` command. To load and pre-render levels, run the `loadmaps` command.

## Adminstrative commands (bot owner only)

`<>` denotes a required argument, and `[]` denotes an optional argument.

* `load [cog]`(aliases: `reload`, `reloadcog`) Reloads a cog. Useful to hot-reload modules of the bot. If the argument is omitted, all cogs are reloaded.
* `restart` Exits the bot with a return code of 1. (I use this with a process manager that restarts failed tasks.)
* `logout` (aliases: `kill`, `yeet`) Exits the bot with a return code of 0. 
* `debug` Gives some debug information about the bot health, including the number of IDENTIFY and RESUME payloads in the past 24 hours.
* `ban <user_id>` Adds a user ID to the list of blacklisted users. (This might actually be broken, haven't tested properly)
* `leave <guild_id>` Leaves a guild.
* `hidden` Lists all hidden commands.
* `doc <command>` Displays the docstring for a command.

The bot additionally uses [Jishaku](https://github.com/Gorialis/jishaku/) to interface with `git`, run shell commands and evaluate python. Read more about the `jsk` command at [Jishaku's documentation](https://jishaku.readthedocs.io/en/latest/).

