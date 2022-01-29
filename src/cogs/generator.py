from __future__ import annotations

from typing import Literal

import discord
import numpy as np
import json
import random
import time
import re

from io import BytesIO
from discord.ext import commands
from PIL import Image
from os import urandom as _urandom

from ..types import Bot, Context
from .. import constants

class GeneratorCog(commands.Cog, name="Generation Commands"):
  def __init__(self, bot: Bot):
    self.bot = bot
  
  #Rewriting Random so I can get the seed
  #https://stackoverflow.com/a/34699351/13290530
  class Random(random.Random):
      def seed(self, a=None):
          if a is None:
              try:
                # Seed with enough bytes to span the 19937 bit
                # state space for the Mersenne Twister
                a = int.from_bytes(_urandom(8), 'big')
              except NotImplementedError:
                import time
                a = int(time.time() * 256)%(2**64) # use fractional seconds
          self._current_seed = a
          super().seed(a)
      def get_seed(self):
          return self._current_seed
  
  def recolor(self, sprite: Image.Image, color: str, palette: np.ndarray) -> Image.Image:
    '''Apply rgb color'''
    r,g,b = palette[constants.COLOR_NAMES[color][::-1]]
    arr = np.asarray(sprite, dtype='float64')
    arr[..., 0] *= r / 256
    arr[..., 1] *= g / 256
    arr[..., 2] *= b / 256
    return Image.fromarray(arr.astype('uint8'))
  
  def blacken(self, sprite: Image.Image, palette) -> np.ndarray:
    '''Apply black (convenience)'''
    return self.recolor(np.array(sprite.convert('RGBA')),'black',palette)
  
  def paste(self, src: Image.Image, dst: Image.Image, loc:tuple(int,int), snap: int = 0):
    src.paste(dst,
              tuple([int(x-(s/2)) for x,s in zip(loc,dst.size)]) if snap == 0 else 
              (int(loc[0]-(dst.width/2)),min(loc[1],24-dst.height)) if snap == 1 else
              (int(loc[0]-(dst.width/2)),loc[1]-dst.height),
              dst.convert('RGBA'))
    return src
    
  def generate_image(self,ears,legs,eyes,mouth,color,variant,type,rand):
    with Image.open(f'data/generator/sprites/{type}_{variant}.png') as im:
      palette = np.array(Image.open(f"data/palettes/default.png").convert("RGB"))
      with open('data/generator/spritedata.json') as f:
        spritedata = json.loads(f.read())
        
      if legs != 0:
        positions = spritedata[type][variant][('1leg' if legs == 1 else f'{legs}legs')] 
        for leg in positions:
          with Image.open(f'data/generator/sprites/parts/legs/{rand.randint(1,5)}.png') as i:
            im = self.paste(im,i,leg,1)
      if ears != 0:
        positions = spritedata[type][variant][('1ear' if ears == 1 else '2ears')] 
        for ear in positions:
          with Image.open(f'data/generator/sprites/parts/ears/{rand.randint(1,4)}.png') as i:
            im = self.paste(im,i,ear,2)
      if eyes != 0:
        with Image.open(f'data/generator/sprites/parts/eyes/{eyes}.png') as i:
          im = self.paste(im,self.blacken(i,palette),spritedata[type][variant]['eyes'][0])
      if mouth:
        try:
          with Image.open(f'data/generator/sprites/parts/mouth.png') as i:
            im = self.paste(im,self.blacken(i,palette),spritedata[type][variant]['mouth'][0])
        except:
          pass
          
      #Recolor after generation
      im = self.recolor(np.array(im),color,palette)
      
      #Send generated sprite
      btio = BytesIO()
      im.resize((192,192),Image.NEAREST).save(btio,'png')
      btio.seek(0)
      return btio
  
  @commands.command(aliases=['customchar','cc'])
  @commands.cooldown(4, 8, type=commands.BucketType.channel)
  async def customcharacter(self, ctx: Context, ears: str, legs: str, eyes: str, mouth: str, color: str, variant: str, type: str, name: str):
    '''Generates a specified character.'''
    try:
      assert ears == '*' or (int(ears) in range(3)), 'Invalid face! Ears has to be between `0` and `2`.'
      assert legs == '*' or (int(legs) in range(5)), 'Invalid face! Legs has to be between `0` and `4`.'
      assert eyes == '*' or (int(eyes) in range(7)), 'Invalid face! Eyes has to be between `0` and `6`.'
      assert mouth == '*' or (int(mouth) in range(2)), 'Invalid face! Mouth has to be `0` or `1`.'
      assert color == '*' or color in ['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey'], 'Invalid color!\nColor must be one of `pink, red, maroon, yellow, orange, gold, brown, lime, green, cyan, blue, purple, white, silver, grey`.'
      assert variant == '*' or variant in ['smooth','fuzzy','fluffy','polygonal','skinny','belt'], 'Invalid variant!\nVariant must be one of `smooth, fuzzy, fluffy, polygonal, skinny, belt`.'
      assert type == '*' or type in ['long','tall','curved','round'], 'Invalid type!\nType must be one of `long, tall, curved, round`.'
      assert name == '*' or re.fullmatch(r'((?:[bcdfghj-mp-tv-z]|[cpt]h|[bcdgpt]r|[bcfgp]l|s[hk-nptw])(?:(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt])?|(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt])(?:[aeiou]|[aeo]i|ea|[abo]u))|(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|ck|[cpt]h|s[hkpt]))\1?|(?:[aeiou]|[aeo]i|ea|[abo]u)(?:[bcdfghj-mp-tv-z]|[cpt]h|[bcdgpt]r|[bcfgp]l|s[hk-nptw])',
                          name.lower()), 'Invalid name!\nThe naming scheme is pretty complex, just trial and error it, sorry ¯\_(ツ)_/¯'
      #shoutouts to jony for doing the regex here
      #tysm <3
      ears = random.choice([0,0,0,1,2,2,2,2]) if ears == '*' else int(ears)
      legs = random.choice([0,0,1,2,2,2,3,4,4,4]) if legs == '*' else int(legs)
      eyes = random.choice([0,0,1,2,2,2,2,2,3,4,5,6]) if eyes == '*' else int(eyes)
      mouth = random.random() > 0.75 if mouth == '*' else int(mouth)
      color = random.choice(['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey']) if color == '*' else color
      variant = random.choice(['smooth','fuzzy','fluffy','polygonal','skinny','belt']) if variant == '*' else variant
      type = random.choice(['long','tall','curved','round']) if type == '*' else type
      if name == '*':
        a = random.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','th','ph','cr','gr','tr','br','dr','pr','bl','sl','pl','cl','gl','fl','sk','sp','st','sn','sm','sw'])
        b = random.choice(['a','e','i','o','u','ei','oi','ea','ou','ai','au','bu'])
        c = random.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','ck','th','ph','sk','sp','st'])
        name = random.choice([a+b+a+b,a+b,a+b+c,b+c,a+c+b,a+c+b+a+c+b,b+c+b+c,a+b+c+a+b+c,b+a])
    except AssertionError as e:
      return await ctx.error(e.args[0])
    name = name.title()
    embed = discord.Embed(
            color = self.bot.embed_color,
            title = name,
            description = f"{name} is a __**{color}**__, __**{variant}**__, __**{type}**__ creature with __**{eyes}**__ eye{'s' if eyes != 1 else ''}, __**{ears}**__ ear{'s' if ears != 1 else ''}{', __**a mouth**__' if mouth else ''}{f',and __**{legs}'}**__ leg{'s' if legs != 1 else ''}."
        )
    embed.set_footer(text=f'Custom-generated, no seed for you!')
    file = discord.File(self.generate_image(ears,legs,eyes,mouth,color,variant,type,self.Random()),filename=f'{name}-custom.png')
    embed.set_image(url=f'attachment://{name}-custom.png')
    #note to self: it's literally this easy what are you doing
    await ctx.send(embed=embed,file=file)
    
  @commands.command(aliases=['char'])
  @commands.cooldown(4, 8, type=commands.BucketType.channel)
  async def character(self, ctx: Context, *, seed: str = None):
    '''Generates a random character. (These are bad but I'm not a good spriter lol)'''
    rand = self.Random()
    try:
      seed = int(seed)
    except ValueError:
      seed = int.from_bytes(bytes(seed,'utf-8'),'big')%(2**64)
    except TypeError:
      seed = None
    rand.seed(seed)
    ears = rand.choice([0,0,0,1,2,2,2,2])
    legs = rand.choice([0,0,1,2,2,2,3,4,4,4])
    eyes = rand.choice([0,0,1,2,2,2,2,2,3,4,5,6])
    mouth = rand.random() > 0.75
    color = rand.choice(['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey'])
    variant = rand.choice(['smooth','fuzzy','fluffy','polygonal','skinny','belt'])
    typ = rand.choice(['long','tall','curved','round'])
    a = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','th','ph','cr','gr','tr','br','dr','pr','bl','sl','pl','cl','gl','fl','sk','sp','st','sn','sm','sw'])
    b = rand.choice(['a','e','i','o','u','ei','oi','ea','ou','ai','au','bu'])
    c = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','ck','th','ph','sk','sp','st'])
    name = rand.choice([a+b+a+b,a+b,a+b+c,b+c,a+c+b,a+c+b+a+c+b,b+c+b+c,a+b+c+a+b+c,b+a]).title()
    embed = discord.Embed(
            color = self.bot.embed_color,
            title = name,
            description = f"{name} is a __**{color}**__, __**{variant}**__, __**{typ}**__ creature with __**{eyes}**__ eye{'s' if eyes != 1 else ''}, __**{ears}**__ ear{'s' if ears != 1 else ''}{', __**a mouth**__' if mouth else ''}{f',and __**{legs}'}**__ leg{'s' if legs != 1 else ''}."
        )
    embed.set_footer(text=f'Seed: {rand.get_seed()}')
    file = discord.File(self.generate_image(ears,legs,eyes,mouth,color,variant,typ,rand),filename=f'{name}-{rand.get_seed()}.png')
    embed.set_image(url=f'attachment://{name}-{rand.get_seed()}.png')
    #note to self: it's literally this easy what are you doing
    await ctx.send(embed=embed,file=file)
  
def setup(bot: Bot):
    bot.add_cog(GeneratorCog(bot))