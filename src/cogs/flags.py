from __future__ import annotations

import asyncio
import random
import re
from os import listdir
from typing import TYPE_CHECKING

import requests
from PIL import Image

from .. import constants
from ..errors import InvalidFlagError
from ..types import Context

if TYPE_CHECKING:
    from ...ROBOT import Bot


class Flag:
    def __init__(
        self,
        *,
        match: str,
        syntax: str,
        description: str,
        kwargs: list[str],
        mutator=(
            lambda x,
            c: x)):
        self.pattern = re.compile(match)
        self.syntax = syntax
        self.description = description
        self.kwargs = kwargs
        self.mutator = mutator

    async def match(self, ctx: Context, potential_flag: str, x: int, y: int, kwargs: dict, to_delete: list) -> list:
        """Matches the potential flag with the flag.
        Returns the to_delete list passed in, plus the coordinates to delete if matched, and the kwargs."""
        if match := self.pattern.fullmatch(potential_flag):
            to_delete.append((x, y))
            values = await self.mutator(match, ctx)
            for kwarg, value in zip(self.kwargs, values):
                if type(kwargs.get(kwarg, None)) is dict:
                    kwargs[kwarg] |= value
                else:
                    kwargs[kwarg] = value
        return to_delete, kwargs

    def __str__(self):
        return f"""
> `{self.syntax}`
{self.description}"""


class Flags:
    def __init__(self):
        self.list = []

    def register(self, match: str, syntax: str, kwargs: list[str]):
        def decorator(f):
            self.list.append(
                Flag(
                    match=match,
                    syntax=syntax,
                    description=f.__doc__,
                    kwargs=kwargs,
                    mutator=f))

        return decorator


async def setup(bot: Bot):
    flags = Flags()
    bot.flags = flags

    @flags.register(match=r"(?:--background|-b)(?:=(\d)/(\d))?",
                    syntax="(-b | --background)[=<x: int>/<y: int>]",
                    kwargs=["background"])
    async def background(match, _):
        """Sets the background of a render to a palette color."""
        if match.group(1) is not None:
            tx, ty = int(match.group(1)), int(match.group(2))
            if not (0 <= tx < 7 and 0 <= ty < 5):
                raise InvalidFlagError(
                    "The provided background color is invalid.")
            return ((tx, ty),)
        else:
            return ((0, 4),)

    @flags.register(match=r"(?:--background|-b)=#([\da-fA-F]{6})",
                    syntax="(-b | --background)=#<color: hex>",
                    kwargs=["background"])
    async def background_hex(match, _):
        """Sets the background of a render to a hexadecimal color."""
        return [match.group(1)]

    @flags.register(match=r"(?:--palette|-p)=(\w+)",
                    syntax="(-p | --palette)=<palette: str>",
                    kwargs=["palette"])
    async def palette(match, _):
        """Sets the palette to use for the render."""
        palette = match.group(1)
        if palette == "random":
            palette = random.choice(listdir("data/palettes"))[:-4]
        elif palette + ".png" not in listdir("data/palettes"):
            raise InvalidFlagError(
                f"Could not find a palette with name \"{palette}\".")
        return [palette]

    @flags.register(match=r"--raw|-r(?:=(.+))?",
                    syntax="(-r | --raw)=<name: str>",
                    kwargs=["raw_output", "upscale", "extra_name"])
    async def raw(match, _):
        """Outputs a zip file with the render as PNG frames.
        Also makes the default color white for everything, and sets the render scale to 1."""
        return (True, 1, (match.group(1) if match.group(1) else None))

    @flags.register(match=r"--comment=(.*)",
                    syntax="--comment=<comment: str>",
                    kwargs=[])
    async def comment(_, __):
        """Just a comment, does nothing."""
        await asyncio.sleep(1)
        return []

    @flags.register(match=r"--letter",
                    syntax="--letter",
                    kwargs=["letters"])
    async def letters(_, __):
        """Makes text default to letters."""
        return [True]

    @flags.register(
        match=r"(?:--frames|-frames|-f)=([123]+)",
        syntax="(--frames | -f)=<frame: 1, 2, or 3 (arbitrary # of times)>",
        kwargs=["frames"])
    async def frames(match, _):
        """Sets which wobble frames to use."""
        frames = []
        for frame in list(match.group(1)):
            # you have no idea how much i wish i could use a yield here
            frames.append(int(frame))
        return [frames]

    @flags.register(match=r"-c|--combine",
                    syntax="-c | --combine",
                    kwargs=["before_image"])
    async def combine(_, ctx):
        """Sets an image to combine this render with."""  # god this one's so jank.
        msg = None
        do_finally = True
        try:
            msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            if not msg.attachments:
                do_finally = False
                return await ctx.error(
                    'The replied message doesn\'t have an attachment. Did you reply to the bot?')
        except BaseException:
            async for m in ctx.channel.history(limit=10):
                if m.author.id == bot.user.id and m.attachments:
                    try:
                        reply = await ctx.channel.fetch_message(m.reference.message_id)
                        if reply.author == ctx.message.author:
                            msg = m
                            break
                    except BaseException:
                        pass
            if msg is None:
                do_finally = False
                return await ctx.error('None of your commands were found in the last `10` messages.')
        finally:
            if do_finally:
                # try:
                assert int(
                    requests.head(
                        msg.attachments[0].url, stream=True).headers.get(
                        'content-length',
                        0)) <= constants.COMBINE_MAX_FILESIZE, f'Prepended image too large! Max filesize is `{constants.COMBINE_MAX_FILESIZE}` bytes.'
                return [Image.open(requests.get(
                    msg.attachments[0].url, stream=True).raw)]
                # except AttributeError:
                #    raise InvalidFlagError("not actually sure what causes this, if you get this please ping me")

    @flags.register(match=r"(?:--speed|-speed)=(\d+)(%)?",
                    syntax="(--speed | -speed)=<speed: int>[%]",
                    kwargs=["speed"])
    async def speed(match, _):
        """Sets how fast the render is.
Use % to set a percentage of the default render speed."""
        speed = int(match.group(1))
        if match.group(2) is not None:
            speed = int(200 // (speed / 100))
        if speed < 20:
            raise InvalidFlagError(
                f'Frame delta of {speed} milliseconds is too small for the specified file format to handle.')
        return speed,

    @flags.register(match=r"(?:--global|-g)=(.+)",
                    syntax="(--global | -g)=<variant: str>",
                    kwargs=["global_variant"])
    async def global_variant(match, _):
        """Applies a variant to every tile."""
        return ':' + match.group(1),

    @flags.register(match=r"--consistent|-co|--synchronize|-sync",
                    syntax="--consistent|-co|--synchronize|-sync",
                    kwargs=["random_animations"])
    async def sync(_, __):
        """Removes the random animation offset."""
        return False,

    @flags.register(match=r"(?:--grid|-gr)=(\d+)/(\d+)",
                    syntax="--grid|-gr=<width: int>/<height: int>",
                    kwargs=["gridol"])
    async def grid(match, _):
        """Adds a grid overlay."""
        return ((int(match.group(1))), (int(match.group(2)))),

    @flags.register(match=r"(?:--crop)=(\d+)/(\d+)/(\d+)/(\d+)",
                    syntax="--crop=<left: int>/<top: int>/<right: int>/<bottom: int>",
                    kwargs=["crop"])
    async def crop(match, _):
        """Crops the render to the bounding box."""
        return ((int(match.group(1))), (int(match.group(2))),
                (int(match.group(3))), (int(match.group(4)))),

    @flags.register(match=r"--pad=(\d+)/(\d+)/(\d+)/(\d+)",
                    syntax="--pad=<left: int>/<top: int>/<right: int>/<bottom: int>",
                    kwargs=["pad"])
    async def pad(match, _):
        """Pads the render on each side."""
        return ((int(match.group(1))), (int(match.group(2))),
                (int(match.group(3))), (int(match.group(4)))),

    @flags.register(match=r"(?:--scale|-s)=(\d+)",
                    syntax="--scale|-s=<scale: int>",
                    kwargs=["gscale"])
    async def gscale(match, _):
        """Sets the scale of all sprites. Applied before variants."""
        return (int(match.group(1))),

    @flags.register(match=r"(?:--multiplier|-m)=(\d+(?:\.\d+)?)",
                    syntax="--multiplier|-m=<scale: float>",
                    kwargs=["upscale"])
    async def upscale(match, _):
        """Sets the scale of all sprites. Applied after variants."""
        return (float(match.group(1))),

    @flags.register(match=r"--verbose|-v",
                    syntax="--verbose|-v",
                    kwargs=["do_embed"])
    async def verbose(_, __):
        """Shows some extra stats about the render."""
        return True,

    @flags.register(match=r"--noloop|-nl",
                    syntax="--noloop|-nl",
                    kwargs=["loop"])
    async def noloop(_, __):
        """Makes the render not loop."""
        return False,

    @flags.register(match=r"(?:--anim|-am)=(\d+)/(\d+)",
                    syntax="--anim|-am=<wobble: int>/<timestep: int>",
                    kwargs=["animation"])
    async def anim(match, _):
        """Makes the wobble frames independent from the animation.
The first number is how many frames are in a wobble frame, and the second is how many frames are in a timestep."""
        return ((int(match.group(1))), (int(match.group(2)))),

    @flags.register(match=r'(?:--format|-f)=(gif|png)',
                    syntax="--format|-f=<format: gif | png>",
                    kwargs=["image_format"])
    async def format(match, _):
        """Set the format of the render.
Note that PNG formats won't animate inside of Discord, you'll have to open them in the browser."""
        return match.group(1),

    @flags.register(match=r"(?:--spacing|-sp)=-?(\d+)",
                    syntax="(--spacing | -sp)=<spacing: int>",
                    kwargs=["spacing"])
    async def spacing(match, _):
        """Adds spacing to the render."""
        return (int(match.group(1))),

    @flags.register(match=r"--expand|-ex",
                    syntax="--expand|-ex",
                    kwargs=["expand"])
    async def expand(_, __):
        """Expands the render for tiles displaced with the `:displace` variant."""
        return True,

    @flags.register(match=r"--tileborder|-tb",
                    syntax="--tileborder|-tb",
                    kwargs=["tileborder"])
    async def tileborder(_, __):
        """Makes the render's border connect to tiles that tile."""
        return True,

    @flags.register(match=r"--boomerang|-br",
                    syntax="--boomerang|-br",
                    kwargs=["boomerang"])
    async def boomerang(_, __):
        """Make the render reverse at the end."""
        return True,

    @flags.register(match=r"(?:--macro|-mc)=(.+?)\|(.+)",
                    syntax="--macro|-mc=<name: str>|<variants: Variant[]>",
                    kwargs=["macro"])
    async def macro(match, __):
        """Define macros for variants."""
        assert ";" not in match.group(2), "Can't have persistent variants in macros!"
        return {match.group(1): match.group(2)},
