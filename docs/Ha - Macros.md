Macros are a way of combining many variants into one to make rendering complex scenes easier.
They can be thought of as shorthands.

Macros, not unlike filters, are in a public database, and can be manipulated through the `macro` command.
Any macros in the database can be used with `:m!<name>`.

Macros can have arguments! If you add `$<number>` to your macro, any values will substitute that.
As an example, if a macro with the value `rot$1:rot-$1` existed under the name `bug`, 
calling `:m!bug/45` would be substituted for `:rot45:rot-45`. 
To allow for easier macro creation, double negatives function correctly, so `--45` is the same as `45`.

Macros can also be specified locally through the `--macro` or `-mc` flag.
The syntax for this is `-mc=<name>|<value>`.
For example, `-mc=bug|rot$1:rot-$1` would allow for `baba:m!bug/45` to work, even if `bug` isn't in the macro database.