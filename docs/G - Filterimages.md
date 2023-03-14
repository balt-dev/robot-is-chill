Filterimages (or filters for short) are a way of creating custom ways to mess with tiles! 
The database of them is public, and anyone can add to it.
Filters are useful for all sorts of things! Here's how they work:

- A mode is specified while creating the filter, of `relative` or `absolute`, which determines if displacement is relative per-pixel or absolute.
- On the red channel of the filter, there's the X displacement, which ranges from -128 at `00` to 127 at `FF`.
- On the green channel of the filter, it's the same, but along the Y axis.
- The result's brightness is multiplied by the blue channel, and alpha by the alpha channel.

Filters can be added, removed, and messed with using the `filter` command.
A tile can have a filter applied using `:fi!<name>/<absolute?>`, where `<absolute?>` is a boolean value.

