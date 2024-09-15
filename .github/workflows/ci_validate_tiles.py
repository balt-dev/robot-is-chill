# HACK: This makes us able to import types from the bot
import sys
sys.path.append('.')

import re
from pathlib import Path
import typing
import tomlkit
import tomlkit.items
from src.types import TilingMode

VALID_FRAMES = range(1, 3)

CUSTOM_PATH = Path("data/custom")
SPRITES_PATH = Path("data/sprites")

# Keys that are allowed or needed in a tile
REQUIRED_KEYS = {"sprite", "tiling", "color"}
# Some of these are used by basegame tiles
OPTIONAL_KEYS = {"author", "extra_frames", "type", "tags", "active", "text_direction", "source"}

# Blacklist of allowed characters in tilenames
# TODO: This should probably be larger
TILE_NAME_BLACKLIST = ("&", ":", ";", ">")
# Blacklist of allowed characters in sprite filenames
# This should be fine, it encompasses all disallowed chars on both Unix and Windows
SPRITE_NAME_BLACKLIST = ("<", ">", ":", "\"", "/", "\\", "|", "?", "*")

def is_valid_name(tile_name: str) -> bool:
    return all(char not in tile_name for char in TILE_NAME_BLACKLIST)

def main():
    failure = False

    # Check that everything in data/custom is a toml
    custom_tomls = set()
    for path in CUSTOM_PATH.glob("*"):
        if path.is_dir() or path.suffix != ".toml":
            print(f"Non-TOML file found at {path}")
            failure = True
        else:
            custom_tomls.add(path)
    
    # Check that everything in data/sprites is a directory
    sprite_dirs = set()
    for path in SPRITES_PATH.glob("*"):
        if not path.is_dir():
            print(f"Non-directory found at {path}")
            failure = True
        else:
            sprite_dirs.add(path)
    
    # Check that each directory has a corresponding toml and vice versa
    sprites_as_tomls = {(CUSTOM_PATH / path.stem).with_suffix(".toml") for path in sprite_dirs}

    for extra_toml in custom_tomls - sprites_as_tomls:
        print(f"TOML file {extra_toml.stem} has no corresponding directory")
        failure = True
    
    for extra_dir in sprites_as_tomls - custom_tomls:
        print(f"Directory at {extra_dir.stem} has no corresponding TOML file")
        failure = True
    
    # Only check directories that passed the first step
    correct_pairs = ((toml, SPRITES_PATH / toml.stem) for toml in sprites_as_tomls & custom_tomls)
    for toml, directory in correct_pairs:
        toml: Path
        directory: Path
        # Read the TOML file
        try:
            with open(toml, "r") as f:
                doc = tomlkit.load(f)
        except Exception as e:
            print(f"Failed to read TOML at {toml}: {e}")
            failure = True
            continue
        
        # We could treat it like a dict with Python's builtin toml library,
        # but iterating over syntactical items gives us finer control over style
        processed_paths = set()

        for index, (name, item) in enumerate(doc.body):
            this_failed = False
            def fail(reason: str):
                nonlocal this_failed
                if not this_failed:
                    print(f"Failures in {toml}:")
                    this_failed = True
                print(f"    Failure at item {index}: {reason}")
                failure = True

            if isinstance(item, (tomlkit.items.Comment, tomlkit.items.Whitespace)):
                # Skip comments and whitespace
                continue

            if not item.is_table() and not item.is_inline_table():
                fail(f"Item is not a table: {item.as_string()}")
                continue

            if not item.is_inline_table():
                fail(f"Item is not an inline table: {item.as_string()}")

            data: dict[str, typing.Any] = item.value
            name: str = name.key # Extract the key from the escaped string

            if not is_valid_name(name):
                fail(f"Tile has an invalid name: {name}")
            
            # Check for missing/extraneous keys
            keys = set(data.keys())
            missing_keys = REQUIRED_KEYS - (keys - OPTIONAL_KEYS)
            extraneous_keys = (keys - OPTIONAL_KEYS) - REQUIRED_KEYS
            if len(extraneous_keys):
                fail(f"Tile {name} has extraneous key(s): {extraneous_keys}")
            if len(missing_keys):
                fail(f"Tile {name} is missing required key(s): {missing_keys}")
                continue # This could cause errors

            sprite = data["sprite"]
            tiling = data["tiling"]
            if not isinstance(sprite, str):
                fail(f"Tile {name} has a non-string sprite: {sprite}")
            if not isinstance(tiling, str):
                fail(f"Tile {name} has a non-string tiling mode: {tiling}")

            sprite = str(sprite)
            tiling_mode: TilingMode | None = TilingMode.parse(str(tiling))

            if tiling_mode is None:
                fail(f"Tile {name} has an invalid tiling mode: {tiling}")
                continue

            if "author" in data:
                if not isinstance(data["author"], str):
                    fail(f"Tile {name} has a non-string author: {data['author']}")
            
            extra_frames = data.get("extra_frames")
            if extra_frames is None:
                extra_frames = set()
            elif not isinstance(extra_frames, tomlkit.items.Array):
                fail(f"Tile {name} has a non-array extra_frames field: {extra_frames}")
                extra_frames = set()
            else:
                extra_frames = set(extra_frames)
            
            expected_frames = tiling_mode.expected() | extra_frames
            found_frames = set()
            
            for sprite_path in directory.glob(f"{sprite}_*_*.png"):
                # Make sure this is *actually* the sprite we want, as
                # glob confuses things like "bird_0_1" and "bird_old_0_1"
                match = re.match(rf"^{re.escape(sprite)}_(-?\d+)_(\d+).png$", sprite_path.name)
                if match is None:
                    continue
                processed_paths.add(sprite_path)
                frame, wobble = int(match.groups(1)[0]), int(match.groups(1)[1])
                if wobble not in range(1, 4):
                    fail(f"Tile {name} has an out-of-bounds wobble frame of {wobble} on animation frame {frame}")
                found_frames.add(frame)
            
            extraneous_frames = found_frames - expected_frames
            if len(extraneous_frames):
                fail(f"Tile {name} has extraneous frames: {extraneous_frames}")
            missing_frames = expected_frames - found_frames
            if len(missing_frames):
                fail(f"Tile {name} has missing frames: {missing_frames}")
            
        paths = set(directory.glob("*"))
        extraneous_files = processed_paths - paths
        if len(extraneous_files):
            fail("Directory {directory} has extra files: {extraneous_files}")
    
    if failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
else:
    raise Exception("Tried to load CI script as a module")