import os
import json
import sys
import re
import typing

failed_test = False

tiling_mode_names: dict[str, str] = {
    "-1": "None",
    "0": "Directions",
    "1": "Tile",
    "2": "Character",
    "3": "Animated Directions",
    "4": "Animated",
}

tiling_modes: dict[str, set[int]] = {
    "-1": set([0]), # None
    "0": set([0, 8, 16, 24]), # Directions
    "1": set(range(16)), # Tile
    "2": set([ # Character
        31,  0,  1,  2,  3,
         7,  8,  9, 10, 11,
        15, 16, 17, 18, 19,
        23, 24, 25, 26, 27,
    ]),
    "3": set([ # Animated Directions
         0,  1,  2,  3,
         8,  9, 10, 11,
        16, 17, 18, 19,
        24, 25, 26, 27,
    ]),
    "4": set([0, 1, 2, 3]), # Animated
}

valid_frames = set([1, 2, 3])

def check_lowercase(string: str):
    return string.islower() or string.isnumeric()

def assert_fn(boolean: bool, message: str):
    global failed_test
    if not boolean:
        print("TEST FAILED:", message)#, file=sys.stderr)
        failed_test = True

def assert_index(list_: list, index: typing.Any, message: str) -> typing.Any:
    if index in list_:
        return list_[index]
    assert_fn(False, message)

def check_blacklist(path: str, root: str, blacklist: list[str]):
    for blacklist_item in blacklist:
        if root.startswith(os.path.join(path, blacklist_item)):
            return True
    return False

def check_json(path: str, sprite_root: str, blacklisted: bool):
    with open(path) as json_file:
        json_data = json.load(json_file)
    if type(json_data) != list:
        assert_fn(False, f"Invalid JSON file at {path}; not an array")
        return
    assert_fn(len(json_data) > 0, f"Empty JSON file at {path}")
    for tilen, tile in enumerate(json_data):
        tile_name: str = assert_index(tile, "name", f"Item `{tilen}` in `{path}` is missing a name")
        if tile_name == None: continue
        tile_sprite: str = assert_index(tile, "sprite", f"Tile `{tile_name}` in `{path}` is missing a sprite name")
        if tile_sprite == None: continue
        assert_fn(check_lowercase(tile_name), f"The tile name of the tile `{tile_name}` in `{path}` should be lowercase.")
        if not blacklisted:
            tile_mode: str = assert_index(tile, "tiling", f"Tiling mode for tile `{tile_name}` in `{path}` is missing.")
            if tile_mode == None: continue
            if type(tile_mode) != str:
                tile_mode = str(tile_mode)
                assert_fn(False, f"Tiling mode for tile `{tile_name}` in `{path}` is not a string.")
            if not tile_mode in tiling_mode_names:
                assert_fn(False, f"Tiling mode for tile `{tile_name}` in `{path}` does not exist (`{repr(tile_mode)}` is not a real tiling mode).")
                return
            regex = re.compile(rf"{re.escape(tile_sprite.lower())}_(\d+)_(\d)\.png")
            found_tiles: set[int] = set()
            found_frames: dict[int, set[int]] = {}
            for file in os.listdir(os.path.join(sprite_root)):
                matched = regex.match(file.lower())
                if not matched: continue
                assert_fn(file.startswith(tile_sprite), f"Sprite name (`{tile_sprite}`, found in `{path}`) casing is mismatched with file name (`{file}`, found in `{sprite_root}`)")
                found_tile, found_frame = matched.groups()
                found_tile, found_frame = int(found_tile), int(found_frame)
                found_tiles.add(found_tile)
                if found_tile not in found_frames:
                    found_frames[found_tile] = set()
                found_frames[found_tile].add(found_frame)
            missing_tiles = tiling_modes[tile_mode].difference(found_tiles)
            assert_fn(len(missing_tiles) == 0, f"Sprite `{tile_sprite}` is missing tiles for its specified tiling mode ({tiling_mode_names[tile_mode]}): {missing_tiles}")
            excess_tiles = found_tiles.difference(tiling_modes[tile_mode])
            assert_fn(len(excess_tiles) == 0, f"Sprite `{tile_sprite}` has tiles not appropriate for its specified tiling mode ({tiling_mode_names[tile_mode]}): {excess_tiles}")
            for found_tile in found_frames:
                found_frames_for_tile = found_frames[found_tile]
                missing_frames = valid_frames.difference(found_frames_for_tile)
                assert_fn(len(missing_frames) == 0, f"Sprite `{tile_sprite}` is missing frames: {missing_frames}")
                excess_frames = found_frames_for_tile.difference(valid_frames)
                assert_fn(len(excess_frames) == 0, f"Sprite `{tile_sprite}` has excess frames: {excess_frames}")

def check_folder(path: str, sprite_path: str, blacklist: list[str], sprite_blacklist: list[str]):
    for root, _, files in os.walk(path):
        if check_blacklist(path, root, blacklist):
            continue
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".json"):
                if full_path.startswith(sprite_path):
                    assert_fn(False, f"Stray JSON file in sprite directory at {full_path}")
                    continue
                assert_fn(check_lowercase(file), f"File name `{file}` at `{full_path}` should be lowercase.")
                sprite_root = os.path.join(sprite_path, os.path.splitext(file)[0])
                check_json(
                    full_path,
                    sprite_root,
                    check_blacklist(
                        sprite_path,
                        sprite_root,
                        sprite_blacklist
                    )
                )
            assert_fn(not file.endswith(".zip"), f"Stray zip file at {full_path}")

if __name__ == "__main__":
    check_folder("data", "data/sprites", ["generator"], ["baba", "new_adv"])
    if failed_test:
        sys.exit(1)
