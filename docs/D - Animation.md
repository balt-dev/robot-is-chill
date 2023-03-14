Animation is, well, animation. It allows you to create renders with more than one frame.
With animation, tiles are split after stacking, meaning
```
baba>keke&me
```
has one frame with a `me` on top of a `baba`, and the next with a `me` on top of a `keke`.
Tiles can be held over frames with multiple `>`s, so `baba>>>keke` has 3 frames with a `baba`, and one frame with a `keke`. Tiles can be removed with `>-`, aka turning into an empty tile.
The `-c` flag appends the current render to a render you reply to while sending the command, which can make animation much easier, since it can be partitioned.