from __future__ import annotations
import random as rand

# limits
MAX_STACK = 2584
MAX_META_DEPTH = 24
MAX_TILES = 2584
MAX_TEXT_LENGTH = 32

# variants
DIRECTION_TILINGS = {
	0, 2, 3
}

DIRECTION_REPRESENTATION_VARIANTS = {
	"r": "`right` / `r` (Facing right)",
	"u": "`up` / `u` (Facing up)",
	"l": "`left` / `l` (Facing left)",
	"d": "`down` / `d` (Facing down)",
}

DIRECTION_VARIANTS = {
	"right": 0,
	"r": 0,
	"up": 8,
	"u": 8,
	"left": 16,
	"l": 16,
	"down": 24,
	"d": 24,
}

ANIMATION_TILINGS = {
	2, 3, 4
}

ANIMATION_REPRESENTATION_VARIANTS = {
	"a0": "`a0` (Animation frame 0)",
	"a1": "`a1` (Animation frame 1)",
	"a2": "`a2` (Animation frame 2)",
	"a3": "`a3` (Animation frame 3)",
}

ANIMATION_VARIANTS = {
	"a0": 0,
	"a1": 1,
	"a2": 2,
	"a3": 3,
}

SLEEP_TILINGS = {
	2
}

SLEEP_REPRESENTATION_VARIANTS = {
	"s": "`sleep` / `s`"
}

SLEEP_VARIANTS = {
	"sleep": -1,
	"s": -1,
}

AUTO_TILINGS = {
	1
}

AUTO_REPRESENTATION_VARIANTS = {
	"tr": "`tileright` / `tr` (Connects right)",
	"tu": "`tileup` / `tu` (Connects up)",
	"tl": "`tileleft` / `tl` (Connects left)",
	"td": "`tiledown` / `td` (Connects down)",
}

AUTO_VARIANTS = {
	"tr": 1,
	"tileright": 1,
	"tu": 2,
	"tileup": 2,
	"tl": 4,
	"tileleft": 4,
	"td": 8,
	"tiledown": 8,
}

# colors
COLOR_NAMES: dict[str, tuple[int, int]] = {
	"maroon": (2, 1), # Not actually a word in the game
	"gold":   (6, 2), # Not actually a word in the game
	"red":    (2, 2),
	"orange": (2, 3),
	"yellow": (2, 4),
	"lime":   (5, 3),
	"green":  (5, 2),
	"cyan":   (1, 4),
	"blue":   (1, 3),
	"purple": (3, 1),
	"pink":   (4, 1),
	"rosy":   (4, 2),
	"grey":   (0, 1),
	"gray":   (0, 1), # alias
	"black":  (0, 4),
	"silver": (0, 2),
	"white":  (0, 3),
	"brown":  (6, 1),
}

CUSTOM_COLOR_NAMES: dict[str,tuple[int,int,int]] = {
	"mint":         [0x00,0xee,0x8a],
	"blueberry":    [0x8f,0x94,0xc5],
	"night":        [0x13,0x14,0x57],
	"haten":        [0x43,0x3b,0xff],
	"apple":        [0xb1,0x3e,0x53],
	"lemon":        [0xff,0xcd,0x75], 
	"grape":        [0x5d,0x27,0x5d],
	"magenta":      [0xff,0x00,0xff],
	"cherry":       [0xFF,0x47,0x50],
	"rose":         [0xFF,0x84,0xB9],
	"azure":        [0x00,0x7f,0xff],
	"mud":          [0x5C,0x47,0x42],
	"dreamvoyager": [0xdf,0x4f,0xe1],
	"cobalt":       [0x20,0x66,0x94],
	"digin":        [0x4C,0xFF,0x00],
	"adr":          [0xFC,0xEC,0x94],
	"fullest":      [0x00,0xC0,0x00],
	"mrld":         [0x82,0xFF,0x97],
	"ping":         [0xED,0x42,0x45],
	"ocean":        [0xb0,0xe0,0xd6],
	"violet":       [0x44,0x22,0xff],
	"blurple":      [0x72,0x89,0xda]
}

COLOR_REPRESENTATION_VARIANTS = {
	"foobarbaz": ", ".join(f"{color}" for color in COLOR_NAMES) + " (Color names)"
}

CUSTOM_COLOR_REPRESENTATION_VARIANTS = {
	"coom": "\n".join(f"{color}: #{''.join([hex(h)[2:].zfill(2) for h in hx])}" for color,hx in list(CUSTOM_COLOR_NAMES.items())) + "\n(Custom color names)"
}

INACTIVE_COLORS: dict[tuple[int, int], tuple[int, int]] = {
	(0, 0): (0, 4),
	(1, 0): (0, 4),
	(2, 0): (1, 1),
	(3, 0): (0, 4),
	(4, 0): (0, 4),
	(5, 0): (6, 3),
	(6, 0): (6, 3),
	(0, 1): (1, 1),
	(1, 1): (1, 0),
	(2, 1): (2, 0),
	(3, 1): (3, 0),
	(4, 1): (4, 0),
	(5, 1): (5, 0),
	(6, 1): (6, 0),
	(0, 2): (0, 1),
	(1, 2): (1, 1),
	(2, 2): (2, 1),
	(3, 2): (1, 1),
	(4, 2): (4, 1),
	(5, 2): (5, 1),
	(6, 2): (6, 1),
	(0, 3): (0, 1),
	(1, 3): (1, 2),
	(2, 3): (2, 2),
	(3, 3): (4, 3),
	(4, 3): (1, 0),
	(5, 3): (5, 1),
	(6, 3): (0, 4),
	(0, 4): (6, 4),
	(1, 4): (1, 2),
	(2, 4): (6, 1),
	(3, 4): (6, 0),
	(4, 4): (3, 2),
	(5, 4): (5, 2),
	(6, 4): (6, 4),
}

# other constants
DIRECTIONS = {
	0: "right",
	8: "up",
	16: "left",
	24: "down"
}

# for n in [[0,1],[-1,0],[0,-1],[1,0],[-1,1],[-1,-1],[1,-1],[]] 
TILING_VARIANTS: dict[tuple[bool, bool, bool, bool, bool, bool, bool, bool], int] = {
	#R, U, L, D, E, Q, Z, C
	# Straightforward so far, easy to compute with a bitfield
	(False, False, False, False, False, False, False, False): 0,
	(True,  False, False, False, False, False, False, False): 1,
	(False, True,  False, False, False, False, False, False): 2,
	(True,  True,  False, False, False, False, False, False): 3,
	(False, False, True,  False, False, False, False, False): 4,
	(True,  False, True,  False, False, False, False, False): 5,
	(False, True,  True,  False, False, False, False, False): 6,
	(True,  True,  True,  False, False, False, False, False): 7,
	(False, False, False, True,  False, False, False, False): 8,
	(True,  False, False, True,  False, False, False, False): 9,
	(False, True,  False, True,  False, False, False, False): 10,
	(True,  True,  False, True,  False, False, False, False): 11,
	(False, False, True,  True,  False, False, False, False): 12,
	(True,  False, True,  True,  False, False, False, False): 13,
	(False, True,  True,  True,  False, False, False, False): 14,
	(True,  True,  True,  True,  False, False, False, False): 15,
	# Messy from here on, requires hardcoding
	(True,  True,  False, False, True,  False, False, False): 16,
	(True,  True,  True,  False, True,  False, False, False): 17,
	(True,  True,  False, True,  True,  False, False, False): 18,
	(True,  True,  True,  True,  True,  False, False, False): 19,
	(False, True,  True,  False, False, True,  False, False): 20,
	(True,  True,  True,  False, False, True,  False, False): 21,
	(False, True,  True,  True,  False, True,  False, False): 22,
	(True,  True,  True,  True,  False, True,  False, False): 23,
	(True,  True,  True,  False, True,  True,  False, False): 24,
	(True,  True,  True,  True,  True,  True,  False, False): 25,
	(False, False, True,  True,  False, False, True,  False): 26,
	(True,  False, True,  True,  False, False, True,  False): 27,
	(False, True,  True,  True,  False, False, True,  False): 28,
	(True,  True,  True,  True,  False, False, True,  False): 29,
	(True,  True,  True,  True,  True,  False, True,  False): 30,
	(False, True,  True,  True,  False, True,  True,  False): 31,
	(True,  True,  True,  True,  False, True,  True,  False): 32,
	(True,  True,  True,  True,  True,  True,  True,  False): 33,
	(True,  False, False, True,  False, False, False, True ): 34,
	(True,  True,  False, True,  False, False, False, True ): 35,
	(True,  False, True,  True,  False, False, False, True ): 36,
	(True,  True,  True,  True,  False, False, False, True ): 37,
	(True,  True,  False, True,  True,  False, False, True ): 38,
	(True,  True,  True,  True,  True,  False, False, True ): 39,
	(True,  True,  True,  True,  False, True,  False, True ): 40,
	(True,  True,  True,  True,  True,  True,  False, True ): 41,
	(True,  False, True,  True,  False, False, True,  True ): 42,
	(True,  True,  True,  True,  False, False, True,  True ): 43,
	(True,  True,  True,  True,  True,  False, True,  True ): 44,
	(True,  True,  True,  True,  False, True,  True,  True ): 45,
	(True,  True,  True,  True,  True,  True,  True,  True ): 46,
}

# While not all of these are nouns, their appearance is very noun-like
TEXT_TYPES = {
	0: "noun",
	1: "noun",
	2: "property",
	3: "noun",
	4: "noun",
	5: "letter",
	6: "noun",
	7: "noun",
}  
DEFAULT_SPRITE_SIZE = 24
PALETTE_PIXEL_SIZE = 32
SEARCH_RESULT_UNITS_PER_PAGE = 10 # roughtly half the number of lines
OTHER_LEVELS_CUTOFF = 5

BABA_WORLD = "baba"
EXTENSIONS_WORLD = "baba-extensions"

COMBINE_MAX_FILESIZE = 5242880 #in bytes