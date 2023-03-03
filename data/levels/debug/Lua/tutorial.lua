-- Here we add two new objects to the object list

table.insert(editor_objlist_order, "tuto")
table.insert(editor_objlist_order, "text_tuto")

-- This defines the exact data for them (note that since the sprites are specific to this levelpack, sprite_in_root must be false!)

editor_objlist["tuto"] = 
{
	name = "tuto",
	sprite_in_root = false,
	unittype = "object",
	tags = {"abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {0, 3},
}

editor_objlist["text_tuto"] = 
{
	name = "text_tuto",
	sprite_in_root = false,
	unittype = "text",
	tags = {"text","abstract"},
	tiling = -1,
	type = 0,
	layer = 20,
	colour = {0, 2},
	colour_active = {0, 3},
}

-- After adding new objects to the list, formatobjlist() must be run to setup everything correctly.

formatobjlist()

-- Here we load a sound to memory so that it can be played during runtime.

MF_loadsound("example")

-- To demonstrate how modsupport.lua works, here we add a function that displays a simple message and plays a custom sound every time a level starts.

table.insert(mod_hook_functions["level_start"],
	function()
		timedmessage("Starting a new level!")
		MF_playsound("example")
	end
)

-- You could use the following to give the example completion icon:
-- MF_alert("save",generaldata.strings[WORLD],"test","1")