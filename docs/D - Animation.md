Animation is, well, animation. It allows you to create renders with more than one frame.
With animation, tiles are split after stacking, meaning
```
baba>keke&me
```
has one frame with a `me` on top of a `baba`, and the next with a `me` on top of a `keke`.
Tiles can be held over frames with multiple `>`s, so `baba>>>keke` has 3 frames with a `baba`, and one frame with a `keke`. Tiles can be removed with `>-`, aka turning into an empty tile.
The `-c` flag appends the current render to a render you reply to while sending the command, which can make animation much easier, since it can be partitioned.

With animation, variants can be applied without respecifying the name, so
```
baba>:red>
```
has one frame of baba being white, and two of red.
Variants in this state are overwritten, so
```
fofo:rot45>:rot30>:rot15
```
doesn't stack the rotations.
Useful for this is a no-operation variant, which can be done by putting nothing after the variant splitter.
So:
```
baba:red>>:>
```
clears the variant after 3 frames.
However, if you want a variant to persist between applying variants, you can use `;` to split them, so
```
jiji;red>:rot45>:px2>;blue
```
has frames `:red`, `:red:rot45`, `:red:px2`, and `:blue`.
The no-op variant still works here as well.