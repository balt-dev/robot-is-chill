from __future__ import annotations

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
    "gold": (6, 2), # Not actually a word in the game
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

COLOR_REPRESENTATION_VARIANTS = {
    "red": ", ".join(f"`{color}`" for color in COLOR_NAMES) + " (Color names)"
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