# Contributing

This file details how to add new sprites to the bot.
If you're making code changes, simply add them to a PR.
Code and sprite changes must be kept in separate pull requests.

---

Each sprite group, called a "source", has:
- a file under `data/custom/<name>.toml`
- a folder under `data/sprites/<name>`

The sprites for each tile are grouped into the `sprites` folder, stored exactly how Baba Is You stores them.

Each entry in the `custom` file is an inline TOML table.
_Do not make these tables multiline. This will cause merge conflicts._

The entry's key is the tile's name - 
the string you enter in the bot to make that tile show up.

The entry's values are as follows:
- `sprite`: Required. Must be a `string`. Specifies the filename that the sprites have, sans the animation and wobble frames, in the `sprites` directory.
- `color`: Required. Must be a `list` of exactly two `integer`s. Specifies the palette index color that this sprite has.
- `tiling`: Required. Specifies which frames the tile has by default. Must be a string of value:
  - `custom`: No frames, all must be specified in `extra_frames`
  - `none`: `0`
  - `direcctional`: `0, 8, 16, 24`
  - `tiling`: `0-15`
  - `character`: `0-3, 7-11, 15-19, 23-27, 31`
  - `animated_directional`: `0-3, 8-11, 16-19, 24-27`
  - `animated`: `0-3`
  - `static_character`: `0-3, 31`
  - `diagonal_tiling`: `0-46`
- `extra_frames`: Optional. Specifies which frames the tile has outside of its tiling mode, if any. Must be a `list` of `integer`s.
- `author`: Recommended. Must be a `string`. Specifies the creator of a tile.
- `tags`: Optional. List of attributes the sprite has to allow for easier searching. Mostly seen within imported base-game sprites. Must be a `list` of `strings`.
- `active`: Not recommended. Specifies the "active color" of text when activated inside Baba Is You - should only be seen within imported files. Same format as `color`.
- `source`: Not recommended. Overrides the directory to look for sprites in. Must be a `string`.

**DO NOT ADD UNFINISHED SPRITES.**
If a sprite does not have all of its frames, do not add it to the bot until it does.
