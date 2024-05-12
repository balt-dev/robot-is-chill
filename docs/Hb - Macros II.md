## The following assumes prior knowledge of how macros work.
### _Page written by @someone1065 - thank you!_

Previously macros were just sets of variants.
`=m i a`
```=t keke:m!a```

Now macros can pretty much be anything and work like text replacement.
`=m i limbo_r`
```=t -am=5/1 -speed=40 - - -
key[limbo_r] - -
- - -```
```=t -am=5/1 -speed=40 - - -
key:disp5/0>:disp10/0>:disp14/0>:disp19/0>:disp24/0>:disp29/0>:disp34/0>:disp38/0>:disp43/0>:disp48/0> - -
- - -```
These 2 renders are identical, but one uses a macro as text replacement.

Macros can also call eachother.
`=m i co`
`=m i tiffcolor`
```=t it:m!co/tiff```
And as you can see they can use arguments in the form of $n.

The other new things are mostly variables and math.
Also, you can execute macro code by itself using `=m x [macroname]` or `=m x --debug [macro]` (if you need to troubleshoot it).
`=m x [add/2/1]`
`=m x [multiply/6/7]`

Since macros can call eachother, you can also nest them.
For example, this is the equation 6*(3+4):
`=m x [multiply/6/[add/3/4]]`

Now for the variables.
You can store to a variable with [store]. And then load it with [load].
`=m x [store/x/3][store/y/14][multiply/[load/x]/[load/y]]`

However, loading a blank variable will just error.
`=m x [load/x]`
`=m x -d [load/x]`
Which is why you can use [get].
`=m x [get/x/1]...[load/x]`
[get] stores a value unless there already is one, it also outputs it, so it's more of a [load] with an edge case fix than a [store].
If there already is one, it will just load it.
`=m x [store/x/2]...[get/x/1]...[load/x]`

All the variables reset when the execution ends.
`=m x [store/x/exists][load/x]`
`=m x [get/x/existsn't]`

If you have more than 16 variables attempted to be stored, it errors too.
`=m x -d [store/a/1][store/b/2][store/c/3][store/d/4][store/e/5][store/f/6][store/g/7][store/h/8][store/i/9][store/j/10][store/k/11][store/l/12][store/m/13][store/n/14][store/o/15][store/p/16][store/q/errors][store/r/?]`