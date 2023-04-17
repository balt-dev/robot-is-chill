from __future__ import annotations

import sys
import typing
from typing import Any, Literal

import discord
import numpy as np
import json
import random
import time
import re
import zlib
import zipfile
import cv2

from io import BytesIO
from discord.ext import commands
from PIL import Image, ImageOps
from cpmpy import Model, intvar, boolvar
from cpmpy.expressions.core import Comparison
from enum import Enum

from ..types import Bot, Context
from .. import constants


class CharacterGenerator:
    def generate(
            self,
            seed: int | None = None,
            shape: Literal[*constants.CHARACTER_SHAPES] | None = None,
            eye_count: int | None = None,
            eye_shape: Literal["normal", "angry", "wide"] | None = None,
            variant: Literal[*constants.CHARACTER_VARIANTS] | None = None,
            legs: int | None = None,
            mouth: bool | None = None,
            ears: bool | None = None,
            color: Literal[*tuple(constants.COLOR_NAMES.keys())] | None = None
    ) -> tuple[np.array, tuple[int, str, int, str, str, int, bool, bool, str]]:
        # I'm not proud of this code.
        # This sucks, a lot, and shows me being overconfident.
        # I've removed a naive implementation for a better, less buggy one.
        if seed is None: seed = random.randint(-sys.maxsize - 1, sys.maxsize)
        r = random.Random()
        r.seed(seed)

        # There's a bit left over in the JSONs from my first implementation, but it's alright

        m = Model()

        # Enumeration of non-integer values
        shapes = constants.CHARACTER_SHAPES
        variants = constants.CHARACTER_VARIANTS

        shape_cs = intvar(0, len(shapes)-1, name="shape")
        variant_cs = intvar(0, len(variants)-1, name="variant")
        legs_cs = intvar(0, 4, name="legs")
        m += legs_cs != 1
        eyes_cs = intvar(0, 4, name="eyes")

        # Constraints

        # -- Variant constraints
        m += (shape_cs == shapes.index("curved")).implies(variant_cs != variants.index("skinny"))
        m += (shape_cs == shapes.index("round")).implies(variant_cs < variants.index("skinny"))
        m += (shape_cs == shapes.index("segmented")).implies(variant_cs < variants.index("fuzzy"))

        # -- Eye constraints
        m += ((shape_cs == shapes.index("tall")) | (shape_cs == shapes.index("curved"))).implies(eyes_cs < 4)
        m += (shape_cs == shapes.index("segmented")).implies(eyes_cs < 3)

        # -- Leg constraints
        m += ((shape_cs == shapes.index("curved")) | (shape_cs == shapes.index("long"))).implies(
            (legs_cs == 3) | (legs_cs == 4) | (legs_cs == 0))
        m += ((shape_cs == shapes.index("tall")) | (shape_cs == shapes.index("round"))).implies(
            (legs_cs == 2) | (legs_cs == 3) | (legs_cs == 0))
        m += (shape_cs == shapes.index("segmented")).implies((legs_cs == 2) | (legs_cs == 0))

        # User constraints

        if shape is not None: m += Comparison("==", shape_cs, shapes.index(shape))
        if variant is not None: m += Comparison("==", variant_cs, variants.index(variant))
        if legs is not None: m += Comparison("==", legs_cs, legs)
        if eye_count is not None: m += Comparison("==", eyes_cs, eye_count)

        solutions = []

        def collect():
            solutions.append([
                shape_cs.value(),
                variant_cs.value(),
                legs_cs.value(),
                eyes_cs.value()
            ])

        m.solveAll(display=collect)

        assert len(solutions), "There's no combination of attributes that's valid for the specified arguments."
        shape, variant, legs, eye_count = r.choice(solutions)
        print(shape, variant, legs, eye_count)
        shape = shapes[shape]
        variant = variants[variant]

        eye_shape = eye_shape or r.choice(["normal", "angry", "wide"])
        mouth = mouth if mouth is not None else r.random() > 0.5
        ears = ears if ears is not None else r.random() > 0.5

        with open('data/generator/eyes/eyes.json') as f:
            eyes_json = json.load(f)
        with open('data/generator/legs/legs.json') as f:
            legs_json = json.load(f)
        eye_displacement = [0, 0]
        if variant in eyes_json[shape]['special']:
            eye_displacement = eyes_json[shape
            ]['special'][variant]
        eye_locations = eyes_json[shape]['generic']
        for dir in eye_locations.keys():
            for eyes, locs in eye_locations[dir].items():
                for i, eye in enumerate(locs):
                    for j, wobble in enumerate(eye):
                        for k, loc in enumerate(wobble):
                            if dir == 'left':
                                displacement = np.array(
                                    eye_displacement[0]) * [1, -1]
                            elif dir == 'right':
                                displacement = np.array(eye_displacement[0])
                            else:
                                displacement = np.array(eye_displacement[1])
                            eye_locations[dir][eyes][i][j][k] = (
                                    np.array(loc) + displacement).tolist()
        with Image.open(f'data/generator/eyes/{eye_shape}-awake.png') as f:
            eye_awake = f.copy()
        with Image.open(f'data/generator/eyes/{eye_shape}-sleep.png') as f:
            eye_asleep = f.copy()
        final_arr = np.zeros((96, 24, 24, 4), dtype=np.uint8)
        for dir_name, dir_a in [('right', 0), ('up', 8),
                                ('left', 16), ('down', 24)]:
            dir = dir_a if dir_a != 16 else 0
            for walkcycle_frame in range(-1, 4):
                for wobble_frame in range(3):
                    with Image.open(
                            f'data/generator/bodies/{shape}-{variant}_{(dir + walkcycle_frame) % 32}_{wobble_frame + 1}.png') as base:
                        if ears:
                            try:
                                with Image.open(
                                        f'data/generator/ears/{shape}-{variant}_{(dir + walkcycle_frame) % 32}_{wobble_frame + 1}.png') as ears_sprite:
                                    ears_sprite = ears_sprite.convert('RGBA')
                                    base.paste(ears_sprite, mask=ears_sprite.getchannel('A'))
                            except FileNotFoundError:
                                pass
                        if legs:
                            with Image.open(
                                    f'data/generator/legs/{shape}-{legs}_{(dir + min(walkcycle_frame, 0)) % 32}_{wobble_frame + 1}.png') as leg_sprite:
                                leg_sprite = np.array(leg_sprite.convert('RGBA'))
                                if walkcycle_frame > 0 and walkcycle_frame % 2 == 1:
                                    leg_sprite = np.roll(leg_sprite, (0, -1), (1, 0))
                                offset = \
                                    legs_json[shape]["offset"].get(variant, legs_json[shape]["offset"]["generic"])[
                                        dir_name if dir_name != "left" else "right"]
                                leg_sprite = np.roll(leg_sprite, offset, (1, 0))
                                if variant == "belt-like" and dir != 8:
                                    leg_sprite[:-7, :, :] = 0
                                leg_sprite = Image.fromarray(leg_sprite)
                                leg_sprite.paste(base, mask=base.getchannel('A'))
                                base = leg_sprite
                        if dir != 8:
                            if eye_count != 0:
                                eye = eye_awake if walkcycle_frame != -1 else eye_asleep
                                if variant != 'belt-like':
                                    eye = np.array(eye, dtype=np.uint8)
                                    eye[:, :, :3] = 0
                                    eye = Image.fromarray(eye)
                                eye_offset = (
                                    1 if walkcycle_frame == -1 else -1 if walkcycle_frame %
                                                                          2 == 1 and dir != 24 else 0)
                                for right_eye in \
                                        eye_locations[dir_name if dir_name != "left" else "right"]['right_eyes'][
                                            wobble_frame][
                                            eye_count - 1]:
                                    base.paste(ImageOps.mirror(eye),
                                               (np.array(right_eye) -
                                                [3 - eye_offset, 0]).tolist()[::-1],
                                               ImageOps.mirror(eye).getchannel(3))
                                for left_eye in \
                                        eye_locations[dir_name if dir_name != "left" else "right"]['left_eyes'][
                                            wobble_frame][
                                            eye_count - 1]:
                                    base.paste(eye, (np.array(
                                        left_eye) - [3 - eye_offset, 2]).tolist()[::-1], eye.getchannel(3))
                            if mouth:
                                with Image.open(
                                        f'data/generator/mouths/{shape}/{shape}_{dir}_{wobble_frame + 1}.png') as mouth_sprite:
                                    mouth_sprite = np.array(mouth_sprite)
                                if variant == 'belt-like':
                                    if shape == 'tall' and dir == 24:
                                        mouth_sprite = np.roll(mouth_sprite, (0, 3), (1, 0))
                                    mouth_sprite[:, :, :3] = 255
                                if walkcycle_frame > 0 and walkcycle_frame % 2 == 1:
                                    mouth_sprite = np.roll(mouth_sprite, (0, -1), (1, 0))
                                mouth_sprite = Image.fromarray(mouth_sprite)
                                base.paste(mouth_sprite, mask=mouth_sprite)
                        final_image = np.array(base)
                        if dir_a == 16:
                            final_image = final_image[:, ::-1, :]
                        final_arr[(((dir_a + walkcycle_frame) % 32) *
                                   3) + wobble_frame] = final_image
        if color is None:
            color = r.choice(list(constants.COLOR_NAMES.keys()))
        args = (
            seed,
            shape,
            eye_count,
            eye_shape,
            variant,
            legs,
            mouth,
            ears,
            color
        )
        return final_arr, args


def recolor(sprite: Image.Image, color: str,
            palette: np.ndarray) -> Image.Image:
    """Apply rgb color."""
    r, g, b = palette[constants.COLOR_NAMES[color][::-1]]
    arr = np.asarray(sprite, dtype="float64")
    arr[..., 0] *= r / 256
    arr[..., 1] *= g / 256
    arr[..., 2] *= b / 256
    return Image.fromarray(arr.astype("uint8"))


class GeneratorCog(commands.Cog, name="Generation Commands"):
    def __init__(self, bot: Bot):
        self.bot = bot

    # Subclassing Random so I can get the seed
    # https://stackoverflow.com/a/34699351/13290530
    class Random(random.Random):
        _current_seed: int

        def seed(self, a=None, **kwargs):
            if a is None:
                # use fractional seconds
                a = int(time.time() * 256) % (2 ** 64)
            self._current_seed = a
            super().seed(a)

        def get_seed(self):
            return self._current_seed

    # New code for character generation
    @commands.command(aliases=["char"])
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def character(
            self, ctx: Context, *,
            kwargs: str = ""
    ):
        f"""
        Randomly generate a character using prefabs.
        Arguments can be specified with a `name:value` syntax.
        The possible arguments are:
        - `seed: any`
            RNG seed, can be anything
        - `shape: {'" | "'.join(constants.CHARACTER_SHAPES)}`
            Character shape
        - `variant: "{'" | "'.join(constants.CHARACTER_VARIANTS)}"`
            Character variant
        - `eye_count: int`
            Number of eyes
        - `eye_shape: "normal" | "angry" | "wide"
            Eye type
        - `legs: int`
            Leg amount
        - `mouth: bool`
            Has/doesn't have a mouth
        - `ears: bool`
            Has/doesn't have ears
        - `color: Color`
            Color of the character
        """
        await ctx.typing()
        possible_kwargs = {
            "seed": int,
            "shape": Literal[*constants.CHARACTER_SHAPES],
            "eye_count": int,
            "eye_shape": Literal["normal", "angry", "wide"],
            "variant": Literal[*constants.CHARACTER_VARIANTS],
            "legs": int,
            "mouth": bool,
            "ears": bool,
            "color": Literal[*constants.COLOR_NAMES]
        }
        flags = {}
        for match in re.finditer(r"(\w+?)=(\w+?)\b", kwargs):
            key, value = match.groups()
            assert key in possible_kwargs, f"Invalid attribute of name `{key}`!"
            val_type = possible_kwargs[key]
            if len(args := typing.get_args(val_type)):
                assert value in args, f"Invalid value `{value}` for attribute `{key}`! Allowed values are `{','.join(args)}."
            elif val_type == bool:
                assert value in (
                "1", "true", "0", "false"), f"Invalid value `{value}` for attribute `{key}` of type `bool`!"
                value = value in ("1", "true")
            else:
                try:
                    value = val_type(value)
                except (TypeError, ValueError):
                    raise AssertionError(f"Invalid value `{value}` for attribute `{key}` of type `{val_type}`!")
            flags[key] = value
        print(flags)
        final_arr, attributes = CharacterGenerator().generate(**flags)
        preview = []
        zip_buffer = BytesIO()
        with zipfile.PyZipFile(zip_buffer, "x") as final_zip:
            for dir_name, dir in [
                ('right', 0), ('up', 8), ('left', 16), ('down', 24)
            ]:
                for walkcycle_frame in range(-1, 4):
                    for wobble_frame in range(3):
                        buffer = BytesIO()  # hey this is better than duped code
                        Image.fromarray(
                            final_arr[(((dir + walkcycle_frame) % 32) * 3) + wobble_frame]).save(buffer, "PNG")
                        final_zip.writestr(
                            f"{attributes[0]}_{(dir + walkcycle_frame) % 32}_{wobble_frame + 1}.png",
                            buffer.getvalue())
            with Image.open('data/generator/preview_bg.png') as bg:
                palette = np.array(self.bot.renderer.palette_cache["default"])
                bg: Image.Image = bg.convert('RGBA')
                for image in final_zip.namelist():
                    with final_zip.open(image) as imfile:
                        with Image.open(imfile) as im:
                            color = palette[constants.COLOR_NAMES[attributes[8]][::-1]]
                            im = np.array(im.convert("RGBA"))
                            kern = np.array(((0, 1, 0), (1, -4, 1), (0, 1, 0)))
                            outline = cv2.filter2D(src=np.pad(im[..., 3], ((1, 1), (1, 1))), ddepth=0, kernel=kern)
                            outline = (outline > 0).astype(np.uint8) * 255
                            im = np.multiply(im, color / 255, casting="unsafe").astype(np.uint8)
                            im = outline[..., np.newaxis] + np.pad(im, ((1, 1), (1, 1), (0, 0)))
                            im = Image.fromarray(im).resize((192, 192), Image.Resampling.NEAREST)
                            out = bg.copy()
                            out.paste(
                                im,
                                (32, 32),
                                mask=im.getchannel("A")
                            )
                            preview.append(out)
        preview_file = BytesIO()
        preview[0].save(
            preview_file,
            format="GIF",
            interlace=True,
            save_all=True,
            append_images=preview[1:],
            loop=0,
            duration=200,
            optimize=False
        )
        preview_file.seek(0)
        zip_buffer.seek(0)
        embed = discord.Embed(
            color=self.bot.embed_color,
            description=None)
        embed.description = "```" + '\n'.join([f'{a}: {b}' for a, b in zip(possible_kwargs.keys(), attributes)]) + "```"
        file = discord.File(preview_file, filename="preview.gif")
        embed.set_image(url=f"attachment://preview.gif")
        return await ctx.reply(embed=embed, files=[file, discord.File(zip_buffer, filename='out.zip')])

    # Old code for character generation
    # This code sucks worse than the above
    # Only kept because people didn't want it removed

    def old_blacken(self, sprite: Image.Image, palette) -> np.ndarray:
        """Apply black (convenience)"""
        return recolor(np.array(sprite.convert("RGBA")), "black", palette)

    def old_paste(self, src: Image.Image, dst: Image.Image,
                  loc: tuple[int, int], snap: int = 0):
        src.paste(dst,
                  tuple([int(x - (s / 2)) for x,
                  s in zip(loc, dst.size)]) if snap == 0 else (
                      int(loc[0] - (dst.width / 2)),
                      min(loc[1], 24 - dst.height)
                  ) if snap == 1 else (
                      int(loc[0] - (dst.width / 2)),
                      loc[1] - dst.height
                  ),
                  dst.convert("RGBA"))
        return src

    def old_generate_image(self, ears, legs, eyes, mouth,
                           color, variant, type, rand):
        with Image.open(f"data/generator/legacy/sprites/{type}_{variant}.png") as im:
            with Image.open(f"data/palettes/default.png").convert("RGB") as p:
                palette = np.array(p)
                with open("data/generator/legacy/spritedata.json") as f:
                    spritedata = json.loads(f.read())

                if legs != 0:
                    positions = spritedata[type][variant][(
                        "1leg" if legs == 1 else f"{legs}legs")]
                    for leg in positions:
                        with Image.open(f"data/generator/legacy/sprites/parts/legs/{rand.randint(1, 5)}.png") as i:
                            im = self.old_paste(im, i, leg, 1)
                if ears != 0:
                    positions = spritedata[type][variant][(
                        "1ear" if ears == 1 else "2ears")]
                    for ear in positions:
                        with Image.open(f"data/generator/legacy/sprites/parts/ears/{rand.randint(1, 4)}.png") as i:
                            im = self.old_paste(im, i, ear, 2)
                if eyes != 0:
                    with Image.open(f"data/generator/legacy/sprites/parts/eyes/{eyes}.png") as i:
                        im = self.old_paste(
                            im, self.old_blacken(
                                i, palette), spritedata[type][variant]["eyes"][0])
                if mouth:
                    try:
                        with Image.open(f"data/generator/legacy/sprites/parts/mouth.png") as i:
                            im = self.old_paste(
                                im, self.old_blacken(
                                    i, palette), spritedata[type][variant]["mouth"][0])
                    except BaseException:
                        pass

                # Recolor after generation
                im = recolor(np.array(im), color, palette)

                # Send generated sprite
                btio = BytesIO()
                im.resize((192, 192), Image.NEAREST).save(btio, "png")
                btio.seek(0)
                return btio

    @commands.command(aliases=["oldchar"], hidden=True)
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def old_character(self, ctx: Context, *, seed: str = None):
        """Old code for =char, kept for legacy purposes."""
        rand = self.Random()
        try:
            seed = int(seed)
        except ValueError:
            seed = int.from_bytes(bytes(seed, "utf-8"), "big") % (2 ** 64)
        except TypeError:
            seed = None
        rand.seed(seed, )
        ears = rand.choice([0, 0, 0, 1, 2, 2, 2, 2])
        legs = rand.choice([0, 0, 1, 2, 2, 2, 3, 4, 4, 4])
        eyes = rand.choice([0, 0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6])
        mouth = rand.random() > 0.75
        color = rand.choice(["pink",
                             "red",
                             "maroon",
                             "yellow",
                             "orange",
                             "gold",
                             "brown",
                             "lime",
                             "green",
                             "cyan",
                             "blue",
                             "purple",
                             "white",
                             "silver",
                             "grey"])
        variant = rand.choice(
            ["smooth", "fuzzy", "fluffy", "polygonal", "skinny", "belt"])
        typ = rand.choice(["long", "tall", "curved", "round"])
        a = rand.choice(["b",
                         "c",
                         "d",
                         "f",
                         "g",
                         "h",
                         "j",
                         "k",
                         "l",
                         "m",
                         "p",
                         "q",
                         "r",
                         "s",
                         "t",
                         "v",
                         "w",
                         "x",
                         "y",
                         "z",
                         "sh",
                         "ch",
                         "th",
                         "ph",
                         "cr",
                         "gr",
                         "tr",
                         "br",
                         "dr",
                         "pr",
                         "bl",
                         "sl",
                         "pl",
                         "cl",
                         "gl",
                         "fl",
                         "sk",
                         "sp",
                         "st",
                         "sn",
                         "sm",
                         "sw"])
        b = rand.choice(["a", "e", "i", "o", "u", "ei",
                         "oi", "ea", "ou", "ai", "au", "bu"])
        c = rand.choice(["b",
                         "c",
                         "d",
                         "f",
                         "g",
                         "h",
                         "j",
                         "k",
                         "l",
                         "m",
                         "p",
                         "q",
                         "r",
                         "s",
                         "t",
                         "v",
                         "w",
                         "x",
                         "y",
                         "z",
                         "sh",
                         "ch",
                         "ck",
                         "th",
                         "ph",
                         "sk",
                         "sp",
                         "st"])
        name = rand.choice([a + b + a + b,
                            a + b,
                            a + b + c,
                            b + c,
                            a + c + b,
                            a + c + b + a + c + b,
                            b + c + b + c,
                            a + b + c + a + b + c,
                            b + a]).title()
        embed = discord.Embed(
            color=self.bot.embed_color,
            title=name,
            description=f"{name} is a __**{color}**__, __**{variant}**__, __**{typ}**__ creature with __**{eyes}**__ eye{'s' if eyes != 1 else ''}, __**{ears}**__ ear{'s' if ears != 1 else ''}{', __**a mouth**__' if mouth else ''}{f',and __**{legs}'}**__ leg{'s' if legs != 1 else ''}.")
        embed.set_footer(text=f"Seed: {rand.get_seed()}")
        file = discord.File(
            self.old_generate_image(
                ears,
                legs,
                eyes,
                mouth,
                color,
                variant,
                typ,
                rand),
            filename=f"{name}-{rand.get_seed()}.png")
        embed.set_image(url=f"attachment://{name}-{rand.get_seed()}.png")
        # note to self: it's literally this easy what are you doing
        await ctx.send(embed=embed, file=file)

    @commands.command(aliases=['oldcustomchar', 'oldcc'], hidden=True)
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def old_customcharacter(self, ctx: Context, ears: str, legs: str, eyes: str, mouth: str, color: str,
                                  variant: str, type: str, name: str):
        """Old code to generate a specified character."""
        try:
            assert ears == '*' or (int(ears) in range(3)
                                   ), 'Invalid face! Ears has to be between `0` and `2`.'
            assert legs == '*' or (int(legs) in range(5)
                                   ), 'Invalid face! Legs has to be between `0` and `4`.'
            assert eyes == '*' or (int(eyes) in range(7)
                                   ), 'Invalid face! Eyes has to be between `0` and `6`.'
            assert mouth == '*' or (int(mouth) in range(2)
                                    ), 'Invalid face! Mouth has to be `0` or `1`.'
            assert color == '*' or color in ['pink', 'red', 'maroon', 'yellow', 'orange', 'gold', 'brown', 'lime',
                                             'green', 'cyan', 'blue', 'purple', 'white', 'silver',
                                             'grey'], 'Invalid color!\nColor must be one of `pink, red, maroon, yellow, orange, gold, brown, lime, green, cyan, blue, purple, white, silver, grey`.'
            assert variant == '*' or variant in ['smooth', 'fuzzy', 'fluffy', 'polygonal', 'skinny',
                                                 'belt'], 'Invalid variant!\nVariant must be one of `smooth, fuzzy, fluffy, polygonal, skinny, belt`.'
            assert type == '*' or type in ['long', 'tall', 'curved',
                                           'round'], 'Invalid type!\nType must be one of `long, tall, curved, round`.'
            assert name == '*' or re.fullmatch(
                r'((?:[bcdfghj-mp-tv-z]|[cpt]h|[bcdgpt]r|[bcfgp]l|s[hk-nptw])(?:(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt])?|(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt])(?:[aeiou]|[aeo]i|ea|[abo]u))|(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt]))\1?|(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|[cpt]h|[bcdgpt]r|[bcfgp]l|s[hk-nptw])',
                name.lower()), 'Invalid name!\nThe naming scheme is pretty complex, just trial and error it, sorry ¯\\_(ツ)_/¯'
            # shoutouts to jony for doing the regex here
            # tysm <3
            ears = random.choice([0, 0, 0, 1, 2, 2, 2, 2]
                                 ) if ears == '*' else int(ears)
            legs = random.choice(
                [0, 0, 1, 2, 2, 2, 3, 4, 4, 4]) if legs == '*' else int(legs)
            eyes = random.choice(
                [0, 0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6]) if eyes == '*' else int(eyes)
            mouth = random.random() > 0.75 if mouth == '*' else int(mouth)
            color = random.choice(['pink',
                                   'red',
                                   'maroon',
                                   'yellow',
                                   'orange',
                                   'gold',
                                   'brown',
                                   'lime',
                                   'green',
                                   'cyan',
                                   'blue',
                                   'purple',
                                   'white',
                                   'silver',
                                   'grey']) if color == '*' else color
            variant = random.choice(['smooth',
                                     'fuzzy',
                                     'fluffy',
                                     'polygonal',
                                     'skinny',
                                     'belt']) if variant == '*' else variant
            type = random.choice(
                ['long', 'tall', 'curved', 'round']) if type == '*' else type
            if name == '*':
                a = random.choice(['b',
                                   'c',
                                   'd',
                                   'f',
                                   'g',
                                   'h',
                                   'j',
                                   'k',
                                   'l',
                                   'm',
                                   'p',
                                   'q',
                                   'r',
                                   's',
                                   't',
                                   'v',
                                   'w',
                                   'x',
                                   'y',
                                   'z',
                                   'sh',
                                   'ch',
                                   'th',
                                   'ph',
                                   'cr',
                                   'gr',
                                   'tr',
                                   'br',
                                   'dr',
                                   'pr',
                                   'bl',
                                   'sl',
                                   'pl',
                                   'cl',
                                   'gl',
                                   'fl',
                                   'sk',
                                   'sp',
                                   'st',
                                   'sn',
                                   'sm',
                                   'sw'])
                b = random.choice(
                    ['a', 'e', 'i', 'o', 'u', 'ei', 'oi', 'ea', 'ou', 'ai', 'au', 'bu'])
                c = random.choice(['b',
                                   'c',
                                   'd',
                                   'f',
                                   'g',
                                   'h',
                                   'j',
                                   'k',
                                   'l',
                                   'm',
                                   'p',
                                   'q',
                                   'r',
                                   's',
                                   't',
                                   'v',
                                   'w',
                                   'x',
                                   'y',
                                   'z',
                                   'sh',
                                   'ch',
                                   'ck',
                                   'th',
                                   'ph',
                                   'sk',
                                   'sp',
                                   'st'])
                name = random.choice([a + b + a + b,
                                      a + b,
                                      a + b + c,
                                      b + c,
                                      a + c + b,
                                      a + c + b + a + c + b,
                                      b + c + b + c,
                                      a + b + c + a + b + c,
                                      b + a])
        except AssertionError as e:
            return await ctx.error(e.args[0])
        name = name.title()
        embed = discord.Embed(
            color=self.bot.embed_color,
            title=name,
            description=f"{name} is a __**{color}**__, __**{variant}**__, __**{type}**__ creature with __**{eyes}**__ eye{'s' if eyes != 1 else ''}, __**{ears}**__ ear{'s' if ears != 1 else ''}{', __**a mouth**__' if mouth else ''}{f',and __**{legs}'}**__ leg{'s' if legs != 1 else ''}."
        )
        embed.set_footer(text=f'Custom-generated, no seed for you!')
        file = discord.File(
            self.old_generate_image(
                ears,
                legs,
                eyes,
                mouth,
                color,
                variant,
                type,
                self.Random()),
            filename=f'{name}-custom.png')
        embed.set_image(url=f'attachment://{name}-custom.png')
        # note to self: it's literally this easy what are you doing
        await ctx.send(embed=embed, file=file)

    # Level generation

    @commands.command()
    @commands.cooldown(4, 8, type=commands.BucketType.channel)
    async def genlevel(self, ctx: Context, width: int, height: int):
        """Generates a blank level given a width and height."""
        assert width >= 1 and height >= 1, "Too small!"
        t = time.time()
        width += 2
        height += 2
        if width * height > 4194304 and not await ctx.bot.is_owner(ctx.author):
            return await ctx.error(
                "Level size too big! Levels are capped at an area of 2048 by 2048, including borders.")
        b = bytearray(
            b"\x41\x43\x48\x54\x55\x4e\x47\x21\x05\x01\x4d\x41\x50\x20\x02\x00\x00\x00\x00\x00\x4c\x41\x59\x52\x1d\x01\x00\x00\x03\x00")
        blankrow_bordered = b"\x00\x00" + \
                            (b"\xFF\xFF" * (width - 2)) + b"\x00\x00"
        blankrow_borderless = b"\xFF\xFF" * (width)
        for n in range(3):
            b.extend(int.to_bytes(width, length=4, byteorder="little"))
            b.extend(int.to_bytes(height, length=4, byteorder="little"))
            b.extend(
                b"\x0c\x00\x0c\x00\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x80\x3f\x00\x00\x01\x00\x00\x80\x3f\xff\xff\xff\x02\x4d\x41\x49\x4e")
            l = bytearray()
            for m in range(height):
                if n == 0:
                    if m == 0 or m == height - 1:
                        l.extend(b"\x00\x00" * width)
                    else:
                        l.extend(blankrow_bordered)
                else:
                    l.extend(blankrow_borderless)
            cl = zlib.compress(l)
            b.extend(int.to_bytes(len(cl), length=4, byteorder="little"))
            b.extend(cl)
            b.extend(b"\x44\x41\x54\x41\x01\x00\x00\x00\x00")
            datbin = zlib.compress(bytearray([3] * (len(l) // 2)))
            b.extend(len(datbin).to_bytes(length=4, byteorder="little"))
            b.extend(datbin)
        zipbuf = BytesIO()
        with zipfile.PyZipFile(zipbuf, "x") as f:
            f.writestr(f"{width - 2}x{height - 2}.l", b)
            f.writestr(
                f"{width - 2}x{height - 2}.ld",
                f"""[general]
selectorY=-1
unlockcount=0
leveltype=0
specials=0
disableshake=0
levels=0
selectorX=-1
disableruleeffect=0
customruleword=
music=baba
rhythm=10
author=robot is chill
levelid=-1
subtitle=
currobjlist_total=0
paths=0
particles=
disableparticles=0
levelz=20
palette=default.png
localmusic=0
customparent=
paletteroot=1
name={width - 2}x{height - 2}""",
            )
            f.writestr(f"{width - 2}x{height - 2}.png",
                       b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\xd8\x00\x00\x00\x70\x08\x03\x00\x00\x00\x4b\xb1\x4e\x5c\x00\x00\x00\x01\x73\x52\x47\x42\x00\xae\xce\x1c\xe9\x00\x00\x00\x04\x67\x41\x4d\x41\x00\x00\xb1\x8f\x0b\xfc\x61\x05\x00\x00\x00\x0f\x50\x4c\x54\x45\x5b\x82\x38\xa4\xb0\x3e\x72\x72\x72\x10\x10\x10\x00\x00\x00\x0a\xbd\x8d\x19\x00\x00\x00\x05\x74\x52\x4e\x53\xff\xff\xff\xff\x00\xfb\xb6\x0e\x53\x00\x00\x00\x09\x70\x48\x59\x73\x00\x00\x0e\xc2\x00\x00\x0e\xc2\x01\x15\x28\x4a\x80\x00\x00\x02\x5c\x49\x44\x41\x54\x78\x5e\xed\xd4\xcb\x72\xa4\x30\x10\x04\x40\xef\xda\xff\xff\xcd\x7b\x50\xb6\x22\xc0\x30\x88\x97\xc7\xc1\x56\x5e\x26\x10\xa8\xab\xeb\x32\x1f\x5f\xbb\x7c\x0c\xf1\xf1\x5b\xa5\x58\x63\xf3\x0d\x3e\x7e\xab\x14\x6b\x6c\xfe\xf1\x67\x99\xb7\xd7\x10\x79\x50\x8a\x35\x32\x53\xec\x0c\x91\x07\x9d\x2b\xf6\x17\x8f\xd7\x10\xd1\x89\xde\x29\xc5\x1a\x59\x29\x76\x84\x88\x4e\xf4\x4e\xd7\x16\xf3\x78\x8c\x19\x9d\xa8\x62\x83\x51\x29\xd6\x08\x49\xb1\x3d\xcc\xe8\x44\x15\x1b\x8c\xba\xb5\xd8\xe7\x8c\xe3\x0d\x66\x75\x22\x6d\x30\x2a\xc5\x1a\x21\x29\x36\xe5\x78\x83\x59\x9d\x48\x1b\x8c\x3a\x57\x6c\xce\x6a\x45\x9f\xce\x71\x71\x5a\x9c\x76\x66\xa6\xd8\x54\x8a\x35\x42\x9e\x5b\x0c\xd1\xf3\x42\x83\x14\x2a\x4e\x3b\xb3\x53\x6c\x2a\xc5\x1a\x21\x45\xf4\x03\x8a\x15\x61\xa2\x0f\x16\xdb\x60\x76\x8a\x4d\xa5\xd8\x94\x30\xd1\xcf\x2d\x56\xac\x74\x8e\x59\x9d\x2c\xc9\xa3\x52\x6c\x4a\x98\xe8\xce\x6a\xe7\x98\xd5\xc9\x92\x3c\x2a\xc5\xa6\x84\x15\x2b\x14\x1b\x1e\x64\x48\x15\x2a\x92\x47\xa5\xd8\x94\xb0\x62\x95\x62\xc3\x83\x0c\x49\xb1\x65\xd7\x14\x2b\x56\xba\x86\x99\x47\x37\xf4\xbb\x93\xcc\x39\x2b\x5d\xc3\xcc\x14\x9b\x4a\xb1\x46\x56\xb1\xc2\x1a\xff\x06\x83\x5c\xea\x64\xa4\xd8\x54\x8a\x35\xb2\x8a\x15\xd6\xd8\x78\x90\x4b\x9d\x8c\x9f\x2d\x26\xbb\x58\xad\x38\xed\x1c\x0f\x72\xa9\x17\xc3\x06\xa3\x52\xac\x11\x92\x62\x8d\xe3\x41\x2e\xbd\xb5\x98\x55\xd6\xf8\xea\x60\xb1\x22\xd2\x06\xa3\x52\xac\x11\x92\x62\x8d\xc7\x41\x2e\x75\x22\x6d\x30\xea\x17\x16\x2b\x2e\x1f\xfc\x13\x49\xb1\xc6\xf0\x14\x6b\x3c\xee\xe4\xf2\xbd\xc5\x0c\xed\x64\x5a\x61\x8d\xaf\x4e\x16\x43\x72\x8a\xf9\xdd\x60\x68\x27\xcc\x0a\x6b\x7c\x95\x62\x53\x2e\x17\xc9\x37\x15\x13\x52\xac\x30\xe7\xed\xc1\x42\x73\x86\xd9\x20\xc5\xfc\x6e\x30\x34\xc5\x1a\x8f\x27\x19\x66\x83\x9b\x8b\xc9\x5c\xe1\xa3\x9d\xc5\x5c\xfa\x76\xcb\xb1\x0d\x52\xcc\xef\x06\x43\x53\xac\xf1\x38\xc8\xa5\x14\x7b\x2d\xc5\x1a\x21\x29\xb6\x87\x4b\xbf\xb2\x98\xb7\xc5\xe9\x20\x97\x52\xec\xb5\x14\x6b\x84\x3c\xb7\xd8\x1b\xd8\x20\xc5\xfc\x6e\x30\x34\xc5\x6e\x60\x83\x9b\x8a\xad\x90\xbd\xf5\x6f\xf1\xfa\x2b\x6f\x7b\x83\x15\x36\xda\x92\x62\x2f\x59\x29\xc5\x16\x79\xfb\xb3\xc5\x56\xc8\xda\x2a\xe6\x6d\x71\x3a\xe7\xed\xb7\x62\xb2\x76\x4a\xb1\x45\xa2\x53\xac\x71\x3a\xe7\xed\x73\x8b\x15\xb3\x53\x6c\x2a\xc5\x16\x89\xfe\x6f\x8b\x95\x5d\x5f\xa5\xd8\xb2\x14\x5b\x24\x3a\xc5\xf6\x7c\x95\x62\xcb\x52\x6c\x91\xe8\xe7\x17\xbb\x94\xd9\x29\x36\x95\x62\x8b\x44\x3f\xb7\xd8\xad\x64\xed\x94\x62\x8b\x44\xdf\x4b\xd6\x4e\x0f\x2d\xf6\xf5\xf5\x0f\xe1\xac\xed\x45\x1e\x29\xc0\x22\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82")
        zipbuf.seek(0)
        return await ctx.send(
            f"Generated in {int((time.time() - t) * 1000)} ms.\nUnzip this into your `Baba Is You/Data/Worlds/levels/` folder to view.",
            files=[discord.File(BytesIO(zipbuf.read()), filename=f"{width - 2}x{height - 2}.zip")])


async def setup(bot: Bot):
    bot.generator = CharacterGenerator()
    await bot.add_cog(GeneratorCog(bot))
