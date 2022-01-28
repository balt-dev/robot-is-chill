import random as rand

def crackseed(ears:int = False,legs:int = False,eyes:int = False,mouth:bool = None,color:str = False,variant:str = False,typ:str = False,name: str = False):
    try:
      assert (type(ears) == bool and not ears) or ears in range(0,3)
      assert (type(ears) == bool and not legs) or legs in range(0,5)
      assert (type(eyes) == bool and not eyes) or eyes in range(0,7)
      assert (type(color) == bool) or color in ['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey']
      assert (type(variant) == bool) or variant in ['smooth','fuzzy','fluffy','polygonal','skinny','belt']
      assert (type(typ) == bool) or typ in ['long','tall','curved','round']
    except Exception as e:
      return None
    n = rand.randint(0,2**64)
    rand.seed(n)
    r_ears = rand.choice([0,0,0,1,2,2,2,2])
    r_legs = rand.choice([0,0,1,2,2,2,3,4,4,4])
    r_eyes = rand.choice([0,0,1,2,2,2,2,2,3,4,5,6])
    r_mouth = rand.random() > 0.75
    r_color = rand.choice(['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey'])
    r_variant = rand.choice(['smooth','fuzzy','fluffy','polygonal','skinny','belt'])
    r_typ = rand.choice(['long','tall','curved','round'])
    a = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','th','ph','cr','gr','tr','br','dr','pr','bl','sl','pl','cl','gl','fl','sk','sp','st','sn','sm','sw'])
    b = rand.choice(['a','e','i','o','u','ei','oi','ea','ou','ai','au','bu'])
    c = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','ck','th','ph','sk','sp','st'])
    r_name = rand.choice([a+b+a+b,a+b,a+b+c,b+c,a+c+b,a+c+b+a+c+b,b+c+b+c,a+b+c+a+b+c,b+a]).title()
    i = 0
    while not all([(type(ears) == bool and not ears) or (ears==r_ears),
                  (type(legs) == bool and not legs) or (legs==r_legs),
                  (type(eyes) == bool and not eyes) or (eyes==r_eyes),
                  (type(mouth) == type(None) and not mouth) or (mouth==r_mouth),
                  (type(color) == bool and not color) or (color==r_color),
                  (type(variant) == bool and not variant) or (variant==r_variant),
                  (type(variant) == bool and not typ) or (typ==r_typ),
                  name==r_name or not name]):
      i += 1
      n = rand.randint(0,2**64)
      rand.seed(n)
      r_ears = rand.choice([0,0,0,1,2,2,2,2])
      r_legs = rand.choice([0,0,1,2,2,2,3,4,4,4])
      r_eyes = rand.choice([0,0,1,2,2,2,2,2,3,4,5,6])
      r_mouth = rand.random() > 0.75
      r_color = rand.choice(['pink','red','maroon','yellow','orange','gold','brown','lime','green','cyan','blue','purple','white','silver','grey'])
      r_variant = rand.choice(['smooth','fuzzy','fluffy','polygonal','skinny','belt'])
      r_typ = rand.choice(['long','tall','curved','round'])
      a = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','th','ph','cr','gr','tr','br','dr','pr','bl','sl','pl','cl','gl','fl','sk','sp','st','sn','sm','sw'])
      b = rand.choice(['a','e','i','o','u','ei','oi','ea','ou','ai','au','bu'])
      c = rand.choice(['b','c','d','f','g','h','j','k','l','m','p','q','r','s','t','v','w','x','y','z','sh','ch','ck','th','ph','sk','sp','st'])
      r_name = rand.choice([a+b+a+b,a+b,a+b+c,b+c,a+c+b,a+c+b+a+c+b,b+c+b+c,a+b+c+a+b+c,b+a]).title()
    return n

print(crackseed())

# ears=2,legs=4,eyes=2,mouth=False,color='white',variant='smooth',typ='long',name='Baba'