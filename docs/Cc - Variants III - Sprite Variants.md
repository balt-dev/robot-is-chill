Sprite variants are a way to manipulate the _sprite_ of the tile, meaning after the initial sprite has been decided, but before the sprite is placed on the grid. This is by far the largest section of variants, and listing all of them here would fill the character cap and then some, so here's a few of the most common ones:

> `:scale<x>[/<y>]` - Scales the sprite by a given magnitude. If only one is specified, it scales both axes, else it scales them individually.
> `:rot<angle>` - Rotates the sprite by a given angle. Be careful to account for lost precision, as images can't be rotated perfectly.
> `:glitch<distance>[/<chance>[/<seed>]]` - "Randomly" displaces the sprite's pixels to at most the given distance, with a optional chance to displace them. A seed, if not supplied, is calculated through a hash of all of the tile's attributes - that is, two identical tiles will be altered in the same way.
> `:warp<top left>/<top right>/<bottom right>/<bottom left>` - Maps the corners of a sprite to an arbitrary quadrilateral, given a list of corner offsets from the original sprite's bounding box. This doesn't manually recenter the sprite, so if needed, use the `:displace` variant to get it back in the right place.
> `:<Color>` - After the initial color is applied, all subsequent colors are applied during this step.
> `:o!<overlay>` - Applies an image, as opposed to a color, to a sprite. You can see the list of available overlays with the `overlays` command.
> `:fi!<url or name>[/<mode>]` - Applies a filterimage. These will be touched on later in the docs.

This is in no way a comprehensive list, so please check the `variants` command to get a gist of what is possible.