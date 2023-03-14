Remember that asterisk from Variants I? Yeah, this is why that was there.
Sign texts are a way of drawing text to the render after everything else, not unlike the text on signs in-game.
They're denoted with `{}`, and any characters with any special meanings (e.g. , `&`, `>`, `$`) can be escaped with `\`.
A newline in a sign text can be inserted using `\n`. 
Keep in mind that sign texts _aren't real tiles_, so most variants will not work on them. 
There are a few that do, however: 
- `:scale`, `:<Color>`, and `:disp<x>/<y>` all work as expected
- `:font!<name>` - Draws using an alternative font. The `icon` font is used internally for level icons, and as such may not look great. 
- `:anchor!<anchor>` - Sets the anchor of the text. [Read more about anchors here.](https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors "Read more about anchors here.")
- `:alignment!<left|center|right>` - Sets the alignment of the text. 
- `:stroke<Color>/size` - Adds an outline around the text. 
With renders that don't have a background and are in the GIF format, text will not be smoothed.
