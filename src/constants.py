from __future__ import annotations


# limits
MAX_STACK = 2584
MAX_META_DEPTH = 48
MAX_META_SIZE = 10
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
    "tr": 0b10000000,
    "tileright": 0b10000000,
    "tu": 0b01000000,
    "tileup": 0b01000000,
    "tl": 0b00100000,
    "tileleft": 0b00100000,
    "td": 0b00010000,
    "tiledown": 0b00010000,
    "tur": 0b11001000,
    "tileupright": 0b11001000,
    "tul": 0b01100100,
    "tileupleft": 0b01100100,
    "tdl": 0b00110010,
    "tiledownleft": 0b00110010,
    "tdr": 0b10010001,
    "tiledownright": 0b10010001,
}

# colors
COLOR_NAMES: dict[str, tuple[int, int]] = {
    "maroon": (2, 1),  # Not actually a word in the game
    "gold": (6, 2),  # Not actually a word in the game
    "teal": (1, 2), # Also not actually a word in the game
    "red": (2, 2),
    "orange": (2, 3),
    "yellow": (2, 4),
    "lime": (5, 3),
    "green": (5, 2),
    "cyan": (1, 4),
    "blue": (3, 2),
    "purple": (3, 1),
    "pink": (4, 1),
    "rosy": (4, 2),
    "grey": (0, 1),
    "gray": (0, 1),  # alias
    "black": (0, 4),
    "silver": (0, 2),
    "white": (0, 3),
    "brown": (6, 1),
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
    None: "blank",
    0: "right",
    8: "up",
    16: "left",
    24: "down",
    -1: "turn",
    -2: "deturn",
    -3: "soft"
}

# for n in [[0,1],[-1,0],[0,-1],[1,0],[-1,1],[-1,-1],[1,-1],[]]
TILING_VARIANTS: dict[int, int] = {
    # R, U, L, D, E, Q, Z, C
    # Straightforward so far, easy to compute with a bitfield
    0b00000000: 0,
    0b10000000: 1,
    0b01000000: 2,
    0b11000000: 3,
    0b00100000: 4,
    0b10100000: 5,
    0b01100000: 6,
    0b11100000: 7,
    0b00010000: 8,
    0b10010000: 9,
    0b01010000: 10,
    0b11010000: 11,
    0b00110000: 12,
    0b10110000: 13,
    0b01110000: 14,
    0b11110000: 15,
    # Messy from here on, requires hardcoding
    0b11001000: 16,
    0b11101000: 17,
    0b11011000: 18,
    0b11111000: 19,
    0b01100100: 20,
    0b11100100: 21,
    0b01110100: 22,
    0b11110100: 23,
    0b11101100: 24,
    0b11111100: 25,
    0b00110010: 26,
    0b10110010: 27,
    0b01110010: 28,
    0b11110010: 29,
    0b11111010: 30,
    0b01110110: 31,
    0b11110110: 32,
    0b11111110: 33,
    0b10010001: 34,
    0b11010001: 35,
    0b10110001: 36,
    0b11110001: 37,
    0b11011001: 38,
    0b11111001: 39,
    0b11110101: 40,
    0b11111101: 41,
    0b10110011: 42,
    0b11110011: 43,
    0b11111011: 44,
    0b11110111: 45,
    0b11111111: 46
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
SEARCH_RESULT_UNITS_PER_PAGE = 20  # roughtly half the number of lines
OTHER_LEVELS_CUTOFF = 20

VANILLA_WORLDS = ("baba", "vanilla", "new_adv", "museum")

COMBINE_MAX_FILESIZE = 5242880  # in bytes

TIMEOUT_DURATION = 20

MAX_TILE_SIZE = 15 * DEFAULT_SPRITE_SIZE
MAX_IMAGE_SIZE = (33 * DEFAULT_SPRITE_SIZE * 2, 18 * DEFAULT_SPRITE_SIZE * 2)

NEWLINE = "\n"

VAR_POSITIONAL_MAX = 64

BLENDING_MODES = (
    "normal",
    "add",
    "subtract",
    "sub",
    "multiply",
    "divide",
    "max",
    "min",
    "screen",
    "softlight",
    "hardlight",
    "overlay",
    "mask",
    "dodge",
    "burn",
    "cut"
)

FILTER_MAX_SIZE = 524288

MAX_SIGN_TEXTS = 128
MAX_SIGN_TEXT_LENGTH = 256

FONT_MULTIPLIERS = {
    "ui": 4 / 3,
    "icon": 4 / 3,
    "offset": 7 / 10
}

MESSAGE_LIMIT = 10

CHARACTER_SHAPES = "long", "tall", "curved", "round", "segmented"
CHARACTER_VARIANTS = "smooth", "fluffy", "fuzzy", "polygonal", "skinny", "belt-like"


MACRO_LIMIT = 5000
MACRO_ARG_LIMIT = 100

LETTER_IGNORE = [
    "text_you2",
    "text_nudgeright",
    "text_nudgeup",
    "text_nudgeleft",
    "text_nudgedown",
    "text_fallright",
    "text_fallup",
    "text_fallleft",
    "text_falldown",
    "text_cash",
    "text_p1",
    "text_p2",
    "text_ab",
    "text_ba",
    "text_besideleft",
    "text_besideright",
    "text_enter",
    "text_3d"
]


