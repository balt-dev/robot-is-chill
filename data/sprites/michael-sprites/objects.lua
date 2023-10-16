-- Here we add two new objects to the object list

table.insert(editor_objlist_order, "her")
table.insert(editor_objlist_order, "text_her")
table.insert(editor_objlist_order, "rithere")
table.insert(editor_objlist_order, "text_rithere")
table.insert(editor_objlist_order, "stubbn")
table.insert(editor_objlist_order, "text_stubbn")

-- This defines the exact data for them (note that since the sprites are specific to this levelpack, sprite_in_root must be false!)

editor_objlist["her"] = 
{
	name = "her",
	sprite_in_root = false,
	unittype = "object",
	tags = {"abstract"},
	tiling = 2,
	type = 1,
	layer = 17,
	colour = {1, 3},
}
editor_objlist["text_her"] = 
{
	name = "text_her",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {1, 2},
	colour_active = {1, 3},
}
editor_objlist["rithere"] = 
{
	name = "rithere",
	sprite_in_root = false,
	unittype = "object",
	tags = {"abstract"},
	tiling = 2,
	type = 1,
	layer = 17,
	colour = {3, 1},
}
editor_objlist["text_rithere"] = 
{
	name = "text_rithere",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {3, 0},
	colour_active = {3, 1},
}
editor_objlist["stubbn"] = 
{
	name = "stubbn",
	sprite_in_root = false,
	unittype = "object",
	tags = {"abstract"},
	tiling = 2,
	type = 1,
	layer = 17,
	colour = {2, 2},
}
editor_objlist["text_stubbn"] = 
{
	name = "text_stubbn",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {2, 1},
	colour_active = {2, 2},
}

-- After adding new objects to the list, formatobjlist() must be run to setup everything correctly.

formatobjlist()

-- Here we load a sound to memory so that it can be played during runtime.
