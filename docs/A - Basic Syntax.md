Tiles are laid out on the grid with their names on a x, y plane, separating by spaces and new lines. `-` denotes an empty tile. 
```
baba keke
me - jiji
```
would give you a 3x2 grid with baba, keke, and an empty tile on the top row, and  me, an empty tile, and jiji on the bottom row.

Stacking tiles is done with `&`, so `level&cursor` would show a level with a cursor on top of it.

All variants are applied using `:` (or `;`, but we'll get to that later). So, if you wanted a `baba` tile that has the variant `left`, yout would type `baba:left`.

Variants apply in the order you specify them in. So, for example, `baba:scale2:rot45` scales the tile before rotating it, while `baba:rot45:scale2` rotates the tile before scaling it.

If you want to display a text object in a `tile` render, prefix the object's name with `text_`. 
```
- baba 

text_baba text_is text_you
```
turns into a render with the baba tile on top, and the sentence "BABA IS YOU" on bottom.
This can be shortened with commas: 
```
text_baba,is,you
```
If you're rendering using `text`, do the opposite, and prefix any tiles with `tile_`. 
A good shorthand for `text_`/`tile_` is using `$`. `$` prefixing a tile will swap it for its counterpart, no matter if you're using `text` or `tile`. This can also be done by enclosing the tiles in quotes.
```
wall wall wall - text_baba,is,you
tile&baba tile tile&flag - $wall,is,stop
wall wall wall - "flag is win"
```