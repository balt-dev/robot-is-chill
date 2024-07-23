table.insert(editor_objlist_order, "text_why?")
editor_objlist["text_why?"] = {
	name = "text_why?",
	sprite_in_root = false,
	unittype = "text",
	tags = {"animal", "abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {0, 0},
	colour_active = {0, 1},
	sprite = "text_why"
}

table.insert(editor_objlist_order, "why?")
editor_objlist["why?"] = {
	name = "why?",
	sprite_in_root = false,
	unittype = "object",
	tags = {"animal", "abstract"},
	tiling = 2,
	type = 0,
	layer = 18,
	colour = {0, 1},
	sprite = "why"
}

formatobjlist()