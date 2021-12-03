#    #ff0000 is eye placement
#    #004000 is for 1 leg
#    #006000 is for 2 legs
#    1 leg + 2 legs = 3 legs
#    #00a000 is 2 legs that get added to the other 2 legs for 4 legs
#    #0000ff is for 1 ear
#    #000080 is for 2 ears
#    #00ffff is mouth placement


from PIL import Image
import numpy as np
import os
import json
import uuid


class NoIndent:
    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(NoIndentEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwargs)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
      result = super(NoIndentEncoder, self).encode(o)
      for k, v in iter(self._replacement_map.items()):
          result = result.replace('"@@%s@@"' % (k,), v)
      return result
      
      
def ct(i,j):
  return all([a==b for a,b in zip(i,j)])
variants = ['smooth','fuzzy','fluffy','polygonal','skinny','belt']
types =    ['long','tall','curved','round']
parts =    ['eyes','1leg','2legs','3legs','4legs','1ear','2ears','mouth']
spritedata = dict()
with Image.open('data/generator/util/spritesraw.png') as im:
  sim = np.array(im)
with Image.open('data/generator/util/spritedata.png') as im:
  dim = np.array(im)

for t, type in enumerate(types):
  spritedata[type] = dict()
  for v, variant in enumerate(variants):
    spritedata[type][variant] = dict()
    for part in parts:
      spritedata[type][variant][part] = NoIndent([])
    sprite = sim[(t*25):((t+1)*25)-1,(v*25):((v+1)*25)-1]
    data = dim[(t*25):((t+1)*25)-1,(v*25):((v+1)*25)-1]
    Image.fromarray(sprite).save(f'data/generator/sprites/{type}_{variant}.png')
    for y, row in enumerate(data):
      for x, pixel in enumerate(row):
        if   ct(pixel,(0xff,0x00,0x00,0xff)):
           spritedata[type][variant]['eyes'].value.append((x,y))
        elif ct(pixel,(0x00,0x40,0x00,0xff)):
           spritedata[type][variant]['1leg'].value.append((x,y))
           spritedata[type][variant]['3legs'].value.append((x,y))
        elif ct(pixel,(0x00,0x60,0x00,0xff)):
          spritedata[type][variant]['2legs'].value.append((x,y))
          spritedata[type][variant]['3legs'].value.append((x,y))
          spritedata[type][variant]['4legs'].value.append((x,y))
        elif ct(pixel,(0x00,0xa0,0x00,0xff)):
          spritedata[type][variant]['4legs'].value.append((x,y))
        elif ct(pixel,(0x00,0x00,0xff,0xff)):
           spritedata[type][variant]['1ear'].value.append((x,y))
        elif ct(pixel,(0x00,0x00,0x80,0xff)):
          spritedata[type][variant]['2ears'].value.append((x,y))
        elif ct(pixel,(0x00,0xff,0xff,0xff)):
          spritedata[type][variant]['mouth'].value.append((x,y))
    with open('data/generator/spritedata.json',mode='w') as f:
      f.write(json.dumps(spritedata,indent=4,cls=NoIndentEncoder))
    

