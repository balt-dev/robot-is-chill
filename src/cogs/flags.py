from __future__ import annotations

import random
import re
from os import listdir
from typing import TYPE_CHECKING

import requests
from PIL import Image

from .. import constants
from ..errors import InvalidFlagError
from ..tile import Tile
from ..types import Color, Macro, RenderContext

if TYPE_CHECKING:
    from ...ROBOT import Bot


class Flag:
    def __init__(
            self,
            *,
            match: str,
            syntax: str,
            description: str,
            mutator=(lambda x, c: x)):
        self.pattern = re.compile(match)
        self.syntax = syntax
        self.description = description
        self.mutator = mutator

    async def match(self, potential_flag: str, ctx: RenderContext) -> bool:
        """Matches the potential flag with the flag.
        Returns the to_delete list passed in, plus the coordinates to delete if matched, and the kwargs."""
        if match := self.pattern.fullmatch(potential_flag):
            await self.mutator(match, ctx)
        return bool(match)

    def __str__(self):
        nl = "\n"
        return f"""
`{self.syntax}`
> {f"{nl}> ".join(self.description.splitlines())}"""

    def __repr__(self):
        return str(self)


class Flags:
    def __init__(self):
        self.list = []

    def register(self, match: str, syntax: str):
        def decorator(f):
            self.list.append(
                Flag(
                    match=match,
                    syntax=syntax,
                    description=f.__doc__,
                    mutator=f))

        return decorator


async def find_message(ctx):
    msg = None
    do_finally = True
    try:
        msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if not msg.attachments:
            do_finally = False
            await ctx.error('The replied message doesn\'t have an attachment. Did you reply to the bot?')
            return None
    except BaseException:
        async for m in ctx.channel.history(limit=constants.MESSAGE_LIMIT):
            if m.author.id == ctx.bot.user.id and m.attachments:
                try:
                    reply = await ctx.channel.fetch_message(m.reference.message_id)
                    if reply.author == ctx.message.author:
                        msg = m
                        break
                except BaseException:
                    pass
    finally:
        if msg is None:
            await ctx.error(f'None of your commands were found in the last `{constants.MESSAGE_LIMIT}` messages.')
            return None
        if do_finally:
            # try:
            assert int(
                requests.head(msg.attachments[0].url, stream=True)
                .headers.get('content-length', 0)) <= constants.COMBINE_MAX_FILESIZE, \
                f'Prepended image too large! Max filesize is `{constants.COMBINE_MAX_FILESIZE}` bytes.'
            with Image.open(requests.get(msg.attachments[0].url, stream=True).raw) as im:
                out = []
                for frame in range(im.n_frames):
                    im.seek(frame)
                    out.append(im.copy())
                return tuple(out)


async def setup(bot: Bot):
    flags = Flags()
    bot.flags = flags

    @flags.register(match=r"(?:--background|-b)(?:=("
                          rf"(?:#(?:[0-9A-Fa-f]{{2}}){{3,4}})|"
                          rf"(?:#(?:[0-9A-Fa-f]){{3,4}})|"
                          rf"(?:{'|'.join(constants.COLOR_NAMES.keys())})|"
                          rf"(?:-?\d+\/-?\d+)))?",
                    syntax="(-b | --background)=#<color: Color>")
    async def background(match, ctx):
        """Sets the background of a render to a color."""
        m = match.group(1)
        if m is None:
            m = 0, 4
        ctx.background = Color.parse(Tile(palette=ctx.palette), bot.renderer.palette_cache, m)

    @flags.register(match=r"(?:--palette|-p)=(\w+)",
                    syntax="(-p | --palette)=<palette: str>")
    async def palette(match, ctx):
        """Sets the palette to use for the render. For a list of palettes, try `search type:palette`."""
        palette = match.group(1)
        if palette == "random":
            palette = random.choice(listdir("data/palettes"))[:-4]
        elif palette + ".png" not in listdir("data/palettes"):
            raise InvalidFlagError(
                f"Could not find a palette with name \"{palette}\".")
        ctx.palette = palette

    @flags.register(match=r"--raw|-r(?:=(.+))?",
                    syntax="(-r | --raw)=<name: str>")
    async def raw(match, ctx):
        """Outputs a zip file with the render as PNG frames.
        Also makes the default color white for everything, and sets the render scale to 1."""
        ctx.raw_output = True
        ctx.upscale = 1
        ctx.extra_name = match.group(1) if match.group(1) else None

    @flags.register(match=r"--comment=(.*)",
                    syntax="--comment=<comment: str>")
    async def comment(match, ctx):
        """Just a comment, does nothing."""
        pass

    @flags.register(match=r"--letter",
                    syntax="--letter")
    async def letters(match, ctx):
        """Makes text default to letters."""
        ctx.letters = True

    @flags.register(
        match=r"(?:--frames|-frames|-f)=([123]+)",
        syntax="(--frames | -f)=<frame: 1, 2, or 3 (arbitrary # of times)>")
    async def frames(match, ctx):
        """Sets which wobble frames to use."""
        frames = []
        for frame in list(match.group(1)):
            frames.append(int(frame))
        ctx.frames = frames

    @flags.register(match=r"-c|--combine",
                    syntax="-c | --combine",
                    )
    async def combine(match, ctx):
        """Sets an image to combine this render with."""
        ctx.before_images = await find_message(ctx.ctx)

    @flags.register(match=r"-bg|--combine-background",
                    syntax="-bg | --combine-background",
                    )
    async def combine_background(match, ctx):
        """Sets an image to set as the background for this render."""
        ctx.background_images = await find_message(ctx.ctx)

    @flags.register(match=r"(?:--speed|-speed)=(\d+)(%)?",
                    syntax="(--speed | -speed)=<speed: int>[%]",
                    )
    async def speed(match, ctx):
        """Sets how fast the render is.
Use % to set a percentage of the default render speed."""
        speed = int(match.group(1))
        if match.group(2) is not None:
            speed = int(200 // (speed / 100))
        if speed < 20:
            raise InvalidFlagError(
                f'Frame delta of {speed} milliseconds is too small for the specified file format to handle.')
        ctx.speed = speed

    @flags.register(match=r"(?:--global|-g)=(.+)",
                    syntax="(--global | -g)=<variant: str>",
                    )
    async def global_variant(match, ctx):
        """Applies a variant to every tile."""
        ctx.global_variant = ':' + match.group(1)

    @flags.register(match=r"--consistent|-co|--synchronize|-sync",
                    syntax="--consistent|-co|--synchronize|-sync",
                    )
    async def sync(match, ctx):
        """Removes the random animation offset."""
        ctx.random_animation = False

    @flags.register(match=r"(?:--crop)=(\d+)/(\d+)/(\d+)/(\d+)",
                    syntax="--crop=<left: int>/<top: int>/<right: int>/<bottom: int>",
                    )
    async def crop(match, ctx):
        """Crops the render on each side."""
        ctx.crop = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))

    @flags.register(match=r"--pad=(\d+)/(\d+)/(\d+)/(\d+)",
                    syntax="--pad=<left: int>/<top: int>/<right: int>/<bottom: int>",
                    )
    async def pad(match, ctx):
        """Pads the render on each side."""
        ctx.pad = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))

    @flags.register(match=r"(?:--scale|-s)=(\d+)",
                    syntax="--scale|-s=<scale: int>",
                    )
    async def gscale(match, ctx):
        """Sets the scale of all sprites. Applied before variants."""
        ctx.gscale = int(match.group(1))

    @flags.register(match=r"(?:--multiplier|-m)=(\d+(?:\.\d+)?)",
                    syntax="--multiplier|-m=<scale: float>",
                    )
    async def upscale(match, ctx):
        """Sets the scale of all sprites. Applied after variants."""
        ctx.upscale = float(match.group(1))

    @flags.register(match=r"--verbose|-v",
                    syntax="--verbose|-v",
                    )
    async def verbose(match, ctx):
        """Shows some extra stats about the render."""
        ctx.do_embed = True

    @flags.register(match=r"--noloop|-nl",
                    syntax="--noloop|-nl",
                    )
    async def noloop(match, ctx):
        """Makes the render not loop."""
        ctx.loop = False

    @flags.register(match=r"--expand|-ex",
                    syntax="--expand|-ex",
                    )
    async def expand(match, ctx):
        """Expands the render for tiles displaced with the `:displace` variant."""
        ctx.expand = True

    @flags.register(match=r"(?:--anim|-am)=(\d+)/(\d+)",
                    syntax="--anim|-am=<wobble: int>/<timestep: int>",
                    )
    async def anim(match, ctx):
        """Makes the wobble frames independent of the animation.
The first number is how many frames are in a wobble frame, and the second is how many frames are in a timestep."""
        ctx.animation = (int(match.group(1))), (int(match.group(2)))

    @flags.register(match=r'(?:--format|-f)=(gif|png)',
                    syntax="--format|-f=<format: gif | png>",
                    )
    async def format(match, ctx):
        """Set the format of the render.
Note that PNG formats won't animate inside of Discord, you'll have to open them in the browser."""
        ctx.image_format = match.group(1)

    @flags.register(match=r"(?:--spacing|-sp)=-?(\d+)",
                    syntax="(--spacing | -sp)=<spacing: int>",
                    )
    async def spacing(match, ctx):
        """Adds spacing to the render."""
        ctx.spacing = int(match.group(1))

    @flags.register(match=r"--tileborder|-tb",
                    syntax="--tileborder|-tb",
                    )
    async def tileborder(match, ctx):
        """Makes the render's border connect to tiles that tile."""
        ctx.tileborder = True

    @flags.register(match=r"--boomerang|-br",
                    syntax="--boomerang|-br",
                    )
    async def boomerang(match, ctx):
        """Make the render reverse at the end."""
        ctx.boomerang = True

    @flags.register(match=r"(?:--macro|-mc)=(.+?)\|(.+)",
                    syntax="--macro|-mc=<name: str>|<variants: Variant[]>",
                    )
    async def macro(match, ctx):
        """Define macros for variants."""
        assert ";" not in match.group(2), "Can't have persistent variants in macros!"
        ctx.macros[match.group(1)] = Macro(value=match.group(2), description="<internal>", author=-1)
