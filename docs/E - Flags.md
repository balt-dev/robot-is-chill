Flags are a way of altering the _entire_ render. These are much less organized than variants, but there are less of them. They can be specified anywhere in the command, but by convention, it's best to put all flags at the beginning of the command, immediately after the command name. A list of common flags is below:

- `-c` - Combines a render with a replied message's render.
- `-co` - Removes the variation in wobble between tiles.
- `-speed=<n>[%]` - Speeds up or slows down the render.
- `-f=<n><n><n>.....` - Sets the wobble frames that will be rendered. These should be a list of numbers between 1 and 3.
- `-f=<gif|png>` - Alters the destination format of the render. PNGS support true transparency and more than 256 colors, but do not animate inside of Discord.
- `-b=<color>` - Sets the background color of a render, in place of transparency. 
- `-m=<n>` - Sets the post-rendering upscale.

Again, this is not a comprehensive list, so make sure to look at the `flags` command for a full list of flags. 