dirs = {{1,0},{0,1},{-1,0},{0,-1},{0,0}}
ndirs = {{1,0},{0,-1},{-1,0},{0,1},{0,0}}
dirs_diagonals = {{1,0},{0,-1},{-1,0},{0,1},{1,1},{1,-1},{-1,-1},{-1,1},{0,0}}
dirs_diagonals_ = {{1,0},{0,-1},{-1,0},{0,1},{1,-1},{-1,-1},{-1,1},{1,1},{0,0}}

colours = {
	default = {0, 3},
	background = {0, 4},
	edge = {1, 0},
	backparticles = {6, 4},
	editorui = {3, 2},
	selector = {1, 4},
	level = {0, 3},
	path = {4, 1},
	menu_background = {1, 0},
	toggle_on = {4,1},
	blocked = {2,2},
	intro = {
		presents = {1, 4},
		b = {4, 1},
		a = {4, 1},
		y = {4, 1},
		o = {4, 1},
		u = {4, 1},
		i = {0, 3},
		s = {0, 3},
		dash = {0, 3},
		dim_purple = {4, 0},
		dim_white = {0, 2},
		editorhype = {2, 3},
	},
	flowers =
	{
		{1, 4},
		{2, 4},
		{2, 3},
		{1, 3},
		{4, 2},
		{5, 3},
	},
}

conditions = {}

keys = 
{
	right = 0,
	up = 1,
	left = 2,
	down = 3,
	idle = 4,
	restart = 5,
	undo = 6,
	quit = 7,
	restart_instant = 8,
}

binds =
{
	raw =
	{
	b0 = {1, 0, 0},
	b1 = {2, 0, 1},
	b2 = {3, 0, 2},
	b3 = {4, 0, 3},
	b4 = {5, 0, 4},
	b5 = {6, 0, 5},
	b6 = {7, 0, 6},
	b7 = {8, 0, 7},
	b8 = {9, 0, 8},
	b9 = {10, 0, 9},
	b10 = {11, 0, 10},
	b11 = {12, 0, 11},
	b12 = {13, 0, 12},
	b13 = {14, 0, 13},
	b14 = {15, 0, 14},
	b15 = {16, 0, 15},
	b16 = {17, 0, 16},
	b17 = {18, 0, 17},
	b18 = {19, 0, 18},
	b19 = {20, 0, 19},
	b20 = {21, 0, 20},
	b21 = {22, 0, 21},
	b22 = {23, 0, 22},
	b23 = {24, 0, 23},
	b24 = {25, 0, 24},
	a0 = {26, 0, 25},
	a1 = {27, 0, 25},
	a2 = {28, 0, 26},
	a3 = {29, 0, 26},
	a4 = {30, 0, 27},
	a5 = {31, 0, 27},
	a6 = {32, 0, 28},
	a7 = {33, 0, 28},
	a8 = {34, 0, 29},
	["h0.0"] = {35, 2, 13},
	["h0.1"] = {36, 2, 17},
	["h0.2"] = {37, 2, 15},
	["h0.3"] = {38, 2, 13},
	["h0.4"] = {39, 2, 19},
	["h0.5"] = {40, 2, 13},
	["h0.6"] = {41, 2, 13},
	["h0.7"] = {42, 2, 13},
	["h0.8"] = {43, 2, 18},
	["h0.9"] = {44, 2, 13},
	["b-1"] = {45, 2, 14},
	},
	named =
	{
	a = {1, 2, 0},
	b = {2, 2, 1},
	x = {3, 2, 2},
	y = {4, 2, 3},
	start = {5, 2, 4},
	back = {6, 2, 5},
	guide = {7, 2, 6},
	leftshoulder = {8, 2, 7},
	rightshoulder = {9, 2, 8},
	lefttrigger = {10, 2, 9},
	righttrigger = {11, 2, 10},
	leftstick = {12, 2, 11},
	rightstick = {13, 2, 12},
	leftx = {14, 2, 11},
	lefty = {15, 2, 11},
	rightx = {16, 2, 12},
	righty = {17, 2, 12},
	dpup = {18, 2, 17},
	dpright = {19, 2, 15},
	dpdown = {20, 2, 19},
	dpleft = {21, 2, 18},
	},
	
	keyboard =
	{
		"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
		"1","2","3","4","5","6","7","8","9","0",
		"Right","Up","Left","Down",
		".",",","-","+","´","'","¨","§","<",
		"Control","Shift","Return","Esc","Space","Backspace",
	},
}

soundnames =
{
	{
		name = "pop",
		count = 5,
	},
	{
		name = "",
	},
	{
		name = "plop",
		count = 4,
	},
	{
		name = "turn",
	},
	{
		name = "move_hi",
		count = 6,
	},
	{
		name = "tele",
		count = 4,
	},
	{
		name = "lock",
		count = 4,
	},
	{
		name = "move",
		count = 6,
	},
	{
		name = "burn",
		count = 6,
	},
	{
		name = "done",
		count = 4,
	},
	{
		name = "silent",
	},
}

word_names =
{
	turn = "turn right",
	deturn = "turn left",
	fallright = "fall (right)",
	fallup = "fall (up)",
	fallleft = "fall (left)",
	nudgeright = "nudge (right)",
	nudgeup = "nudge (up)",
	nudgeleft = "nudge (left)",
	nudgedown = "nudge (down)",
	lockedright = "locked (right)",
	lockedup = "locked (up)",
	lockedleft = "locked (left)",
	lockeddown = "locked (down)",
	besideright = "beside (right)",
	besideleft = "beside (left)",
	nextto = "next to",
}

decor_offsets =
{
	baba = {0,0},
	keke = {{-2,-4},{0,0},{2,-4},{0,-6}},
	robot = {-3,-6},
	ghost = {{-1,-5},{0,0},{1,-5},{-1,-8}},
}

tileslist =
{
	edge =
	{
		-- SPECIAL CASE
		colour = {0, 2},
		tile = {0, 0},
	},
	object000 =
	{
		name = "baba",
		sprite = "baba",
		sprite_in_root = true,
		unittype = "object",
		tiling = 2,
		type = 0,
		colour = {0, 3},
		tile = {1, 0},
		grid = {0, 1},
		layer = 18,
	},
	object001 =
	{
		name = "keke",
		sprite = "keke",
		sprite_in_root = true,
		unittype = "object",
		tiling = 2,
		type = 0,
		colour = {2, 2},
		tile = {2, 0},
		grid = {0, 2},
		layer = 18,
	},
	object002 =
	{
		name = "rock",
		sprite = "rock",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {6, 2},
		tile = {3, 0},
		grid = {0, 4},
		layer = 16,
	},
	object003 =
	{
		name = "text_grass",
		sprite = "text_grass",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {5, 1},
		active = {5, 3},
		tile = {4, 0},
		grid = {4, 4},
		layer = 20,
	},
	object004 =
	{
		name = "tile",
		sprite = "tile",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {1, 0},
		tile = {5, 0},
		grid = {3, 7},
		layer = 4,
	},
	object005 =
	{
		name = "text_and",
		sprite = "text_and",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 6,
		colour = {0, 1},
		active = {0, 3},
		tile = {6, 0},
		grid = {2, 0},
		layer = 20,
	},
	object006 =
	{
		name = "text_hide",
		sprite = "text_hide",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {3, 2},
		active = {3, 3},
		tile = {7, 0},
		grid = {8, 6},
		layer = 20,
	},
	object007 =
	{
		name = "text_follow",
		sprite = "text_follow",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		colour = {5, 0},
		active = {5, 2},
		tile = {8, 0},
		grid = {10, 5},
		layer = 20,
	},
	object008 =
	{
		name = "text_float",
		sprite = "text_float",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 2},
		active = {1, 4},
		tile = {9, 0},
		grid = {10, 2},
		layer = 20,
	},
	object009 =
	{
		name = "text_lonely",
		sprite = "text_lonely",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 3,
		colour = {2, 1},
		active = {2, 2},
		tile = {10, 0},
		grid = {9, 4},
		layer = 20,
	},
	object010 =
	{
		name = "lava",
		sprite = "water",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {2, 3},
		tile = {11, 0},
		grid = {9, 9},
		layer = 2,
	},
	object011 =
	{
		name = "water",
		sprite = "water",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {1, 3},
		tile = {0, 1},
		grid = {0, 6},
		layer = 2,
	},
	object012 =
	{
		name = "wall",
		sprite = "wall",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {1, 1},
		tile = {1, 1},
		grid = {0, 5},
		layer = 14,
	},
	object013 =
	{
		name = "text_empty",
		sprite = "text_empty",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 1},
		active = {0, 3},
		tile = {2, 1},
		grid = {6, 0},
		layer = 20,
	},
	object014 =
	{
		name = "text_tile",
		sprite = "text_tile",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 1},
		active = {0, 1},
		tile = {3, 1},
		grid = {4, 7},
		layer = 20,
	},
	object015 =
	{
		name = "text_weak",
		sprite = "text_weak",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 1},
		active = {1, 2},
		tile = {4, 1},
		grid = {7, 3},
		layer = 20,
	},
	object016 =
	{
		name = "text_near",
		sprite = "text_near",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 7,
		colour = {0, 1},
		active = {0, 3},
		tile = {5, 1},
		grid = {8, 4},
		layer = 20,
	},
	object017 =
	{
		name = "cloud",
		sprite = "cloud",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {1, 4},
		tile = {6, 1},
		grid = {5, 5},
		layer = 14,
	},
	object018 =
	{
		name = "pillar",
		sprite = "pillar",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {0, 1},
		tile = {7, 1},
		grid = {5, 6},
		layer = 16,
	},
	object019 =
	{
		name = "text_fungus",
		sprite = "text_fungus",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 0},
		active = {6, 1},
		tile = {8, 1},
		grid = {8, 7},
		layer = 20,
	},
	object020 =
	{
		name = "text_baba",
		sprite = "text_baba",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 0},
		active = {4, 1},
		tile = {9, 1},
		grid = {1, 1},
		layer = 20,
	},
	object021 =
	{
		name = "text_keke",
		sprite = "text_keke",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 1},
		active = {2, 2},
		tile = {10, 1},
		grid = {1, 2},
		layer = 20,
	},
	object022 =
	{
		name = "text_flag",
		sprite = "text_flag",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 1},
		active = {2, 4},
		tile = {11, 1},
		grid = {1, 3},
		layer = 20,
	},
	object023 =
	{
		name = "flag",
		sprite = "flag",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 4},
		tile = {0, 2},
		grid = {0, 3},
		layer = 17,
	},
	object024 =
	{
		name = "ice",
		sprite = "ice",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {1, 2},
		tile = {1, 2},
		grid = {3, 3},
		layer = 9,
	},
	object025 =
	{
		name = "text_shift",
		sprite = "text_shift",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 2},
		active = {1, 3},
		tile = {2, 2},
		grid = {2, 9},
		layer = 20,
	},
	object026 =
	{
		name = "rose",
		sprite = "rose",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 2},
		tile = {3, 2},
		grid = {9, 10},
		layer = 16,
	},
	object027 =
	{
		name = "text_all",
		sprite = "text_all",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 1},
		active = {0, 3},
		tile = {4, 2},
		grid = {5, 0},
		layer = 20,
	},
	object028 =
	{
		name = "text_right",
		sprite = "text_right",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 3},
		active = {1, 4},
		tile = {5, 2},
		grid = {7, 1},
		layer = 20,
	},
	object029 =
	{
		name = "text_cloud",
		sprite = "text_cloud",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 3},
		active = {1, 4},
		tile = {6, 2},
		grid = {6, 5},
		layer = 20,
	},
	object030 =
	{
		name = "text_pillar",
		sprite = "text_pillar",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 0},
		active = {0, 1},
		tile = {7, 2},
		grid = {6, 6},
		layer = 20,
	},
	object031 =
	{
		name = "fungus",
		sprite = "fungus",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {6, 1},
		tile = {8, 2},
		grid = {7, 7},
		layer = 16,
	},
	object032 =
	{
		name = "text_rock",
		sprite = "text_rock",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 0},
		active = {6, 1},
		tile = {9, 2},
		grid = {1, 4},
		layer = 20,
	},
	object033 =
	{
		name = "text_lava",
		sprite = "text_lava",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 2},
		active = {2, 3},
		tile = {10, 2},
		grid = {10, 9},
		layer = 20,
	},
	object034 =
	{
		name = "text_wall",
		sprite = "text_wall",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 1},
		active = {0, 1},
		tile = {11, 2},
		grid = {1, 5},
		layer = 20,
	},
	object035 =
	{
		name = "text_ice",
		sprite = "text_ice",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 2},
		active = {1, 3},
		tile = {0, 3},
		grid = {4, 3},
		layer = 20,
	},
	object036 =
	{
		name = "text_is",
		sprite = "text_is",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		argtype = {0, 2},
		colour = {0, 1},
		active = {0, 3},
		tile = {1, 3},
		grid = {1, 0},
		layer = 20,
	},
	object037 =
	{
		name = "text_rose",
		sprite = "text_rose",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 1},
		active = {2, 2},
		tile = {7, 3},
		grid = {10, 10},
		layer = 20,
	},
	object038 =
	{
		name = "text_more",
		sprite = "text_more",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {4, 0},
		active = {4, 1},
		tile = {3, 3},
		grid = {10, 3},
		layer = 20,
	},
	object039 =
	{
		name = "text_safe",
		sprite = "text_safe",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {0, 1},
		active = {0, 3},
		tile = {4, 3},
		grid = {7, 6},
		layer = 20,
	},
	object040 =
	{
		name = "text_up",
		sprite = "text_up",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 3},
		active = {1, 4},
		tile = {5, 3},
		grid = {6, 1},
		layer = 20,
	},
	object041 =
	{
		name = "star",
		sprite = "star",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 4},
		tile = {6, 3},
		grid = {5, 9},
		layer = 16,
	},
	object042 =
	{
		name = "text_word",
		sprite = "text_word",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {0, 2},
		active = {0, 3},
		tile = {0, 11},
		grid = {8, 0},
		layer = 20,
	},
	object043 =
	{
		name = "text_fruit",
		sprite = "text_fruit",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 1},
		active = {2, 2},
		tile = {8, 3},
		grid = {4, 10},
		layer = 20,
	},
	object044 =
	{
		name = "text_water",
		sprite = "text_water",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 2},
		active = {1, 3},
		tile = {9, 3},
		grid = {1, 6},
		layer = 20,
	},
	object045 =
	{
		name = "text_win",
		sprite = "text_win",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {6, 1},
		active = {2, 4},
		tile = {10, 3},
		grid = {2, 3},
		layer = 20,
	},
	object046 =
	{
		name = "text_push",
		sprite = "text_push",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {6, 0},
		active = {6, 1},
		tile = {11, 3},
		grid = {2, 4},
		layer = 20,
	},
	object047 =
	{
		name = "text_stop",
		sprite = "text_stop",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {5, 0},
		active = {5, 1},
		tile = {0, 4},
		grid = {2, 5},
		layer = 20,
	},
	object048 =
	{
		name = "text_move",
		sprite = "text_move",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {5, 1},
		active = {5, 3},
		tile = {1, 4},
		grid = {2, 2},
		layer = 20,
	},
	object049 =
	{
		name = "text_best",
		sprite = "text_best",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {2, 3},
		active = {2, 4},
		tile = {2, 4},
		grid = {9, 2},
		layer = 20,
	},
	object050 =
	{
		name = "text_tele",
		sprite = "text_tele",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 2},
		active = {1, 4},
		tile = {3, 4},
		grid = {8, 1},
		layer = 20,
	},
	object051 =
	{
		name = "hand",
		sprite = "hand",
		sprite_in_root = true,
		unittype = "object",
		tiling = 0,
		type = 0,
		colour = {0, 3},
		tile = {4, 4},
		grid = {7, 8},
		layer = 17,
	},
	object052 =
	{
		name = "text_left",
		sprite = "text_left",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 3},
		active = {1, 4},
		tile = {5, 4},
		grid = {6, 2},
		layer = 20,
	},
	object053 =
	{
		name = "text_star",
		sprite = "text_star",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 1},
		active = {2, 4},
		tile = {6, 4},
		grid = {6, 9},
		layer = 20,
	},
	object054 =
	{
		name = "text_red",
		sprite = "text_red",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {2, 1},
		active = {2, 2},
		tile = {7, 4},
		grid = {10, 6},
		layer = 20,
	},
	object055 =
	{
		name = "fruit",
		sprite = "fruit",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 2},
		tile = {8, 4},
		grid = {3, 10},
		layer = 16,
	},
	object056 =
	{
		name = "text_melt",
		sprite = "text_melt",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 2},
		active = {1, 3},
		tile = {9, 4},
		grid = {5, 3},
		layer = 20,
	},
	object057 =
	{
		name = "text_hot",
		sprite = "text_hot",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {2, 2},
		active = {2, 3},
		tile = {10, 4},
		grid = {6, 3},
		layer = 20,
	},
	object058 =
	{
		name = "text_you",
		sprite = "text_you",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {4, 0},
		active = {4, 1},
		tile = {11, 4},
		grid = {2, 1},
		layer = 20,
	},
	object059 =
	{
		name = "text_not",
		sprite = "text_not",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 4,
		colour = {2, 1},
		active = {2, 2},
		tile = {0, 5},
		grid = {4, 0},
		layer = 20,
	},
	object060 =
	{
		name = "text_sink",
		sprite = "text_sink",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 2},
		active = {1, 3},
		tile = {1, 5},
		grid = {2, 6},
		layer = 20,
	},
	object061 =
	{
		name = "love",
		sprite = "love",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {4, 2},
		tile = {2, 5},
		grid = {3, 5},
		layer = 16,
	},
	object062 =
	{
		name = "door",
		sprite = "door",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 2},
		tile = {3, 5},
		grid = {3, 2},
		layer = 15,
	},
	object063 =
	{
		name = "text_hand",
		sprite = "text_hand",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 1},
		active = {0, 3},
		tile = {4, 5},
		grid = {8, 8},
		layer = 20,
	},
	object064 =
	{
		name = "text_down",
		sprite = "text_down",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 3},
		active = {1, 4},
		tile = {5, 5},
		grid = {7, 2},
		layer = 20,
	},
	object065 =
	{
		name = "dust",
		sprite = "dust",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {6, 2},
		tile = {6, 5},
		grid = {9, 8},
		layer = 12,
	},
	object066 =
	{
		name = "text_flower",
		sprite = "text_flower",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {3, 2},
		active = {3, 3},
		tile = {7, 5},
		grid = {6, 4},
		layer = 20,
	},
	object067 =
	{
		name = "text_tree",
		sprite = "text_tree",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {5, 1},
		active = {5, 2},
		tile = {8, 5},
		grid = {4, 9},
		layer = 20,
	},
	object068 =
	{
		name = "ghost",
		sprite = "ghost",
		sprite_in_root = true,
		unittype = "object",
		tiling = 0,
		type = 0,
		colour = {4, 2},
		tile = {10, 10},
		grid = {5, 8},
		layer = 17,
	},
	object069 =
	{
		name = "text_defeat",
		sprite = "text_defeat",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {2, 0},
		active = {2, 1},
		tile = {10, 5},
		grid = {2, 7},
		layer = 20,
	},
	object070 =
	{
		name = "skull",
		sprite = "skull",
		sprite_in_root = true,
		unittype = "object",
		tiling = 0,
		type = 0,
		colour = {2, 1},
		tile = {11, 5},
		grid = {0, 7},
		layer = 16,
	},
	object071 =
	{
		name = "grass",
		sprite = "grass",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {5, 0},
		tile = {0, 6},
		grid = {3, 4},
		layer = 10,
	},
	object072 =
	{
		name = "text_skull",
		sprite = "text_skull",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 0},
		active = {2, 1},
		tile = {1, 6},
		grid = {1, 7},
		layer = 20,
	},
	object073 =
	{
		name = "text_love",
		sprite = "text_love",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 1},
		active = {4, 2},
		tile = {2, 6},
		grid = {4, 5},
		layer = 20,
	},
	object074 =
	{
		name = "text_door",
		sprite = "text_door",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {2, 1},
		active = {2, 2},
		tile = {3, 6},
		grid = {4, 2},
		layer = 20,
	},
	object075 =
	{
		name = "text_text",
		sprite = "text_text",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 0},
		active = {4, 1},
		tile = {4, 6},
		grid = {7, 0},
		layer = 20,
	},
	object076 =
	{
		name = "text_sleep",
		sprite = "text_sleep",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 3},
		active = {1, 4},
		tile = {5, 6},
		grid = {8, 3},
		layer = 20,
	},
	object077 =
	{
		name = "text_dust",
		sprite = "text_dust",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 2},
		active = {2, 4},
		tile = {6, 6},
		grid = {10, 8},
		layer = 20,
	},
	object078 =
	{
		name = "text_blue",
		sprite = "text_blue",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {3, 2},
		active = {3, 3},
		tile = {7, 6},
		grid = {9, 6},
		layer = 20,
	},
	object079 =
	{
		name = "tree",
		sprite = "tree",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {5, 2},
		tile = {8, 6},
		grid = {3, 9},
		layer = 15,
	},
	object080 =
	{
		name = "key",
		sprite = "key",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 4},
		tile = {9, 6},
		grid = {3, 1},
		layer = 16,
	},
	object081 =
	{
		name = "text_key",
		sprite = "text_key",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 1},
		active = {2, 4},
		tile = {10, 6},
		grid = {4, 1},
		layer = 20,
	},
	object082 =
	{
		name = "text_open",
		sprite = "text_open",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {6, 1},
		active = {2, 4},
		tile = {11, 6},
		grid = {5, 1},
		layer = 20,
	},
	object083 =
	{
		name = "text_shut",
		sprite = "text_shut",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {2, 1},
		active = {2, 2},
		tile = {0, 7},
		grid = {5, 2},
		layer = 20,
	},
	object084 =
	{
		name = "text_has",
		sprite = "text_has",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		colour = {0, 1},
		active = {0, 3},
		tile = {1, 7},
		grid = {3, 0},
		layer = 20,
	},
	object085 =
	{
		name = "box",
		sprite = "box",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {6, 2},
		tile = {2, 7},
		grid = {0, 8},
		layer = 16,
	},
	object086 =
	{
		name = "text_box",
		sprite = "text_box",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 0},
		active = {6, 1},
		tile = {3, 7},
		grid = {1, 8},
		layer = 20,
	},
	object087 =
	{
		name = "belt",
		sprite = "belt",
		sprite_in_root = true,
		unittype = "object",
		tiling = 3,
		type = 0,
		colour = {1, 1},
		tile = {4, 7},
		grid = {0, 9},
		layer = 11,
	},
	object088 =
	{
		name = "text_make",
		sprite = "text_make",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		colour = {0, 1},
		active = {0, 3},
		tile = {5, 7},
		grid = {7, 5},
		layer = 20,
	},
	object089 =
	{
		name = "text_fall",
		sprite = "text_fall",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {5, 1},
		active = {5, 3},
		tile = {6, 7},
		grid = {9, 3},
		layer = 20,
	},
	object090 =
	{
		name = "flower",
		sprite = "flower",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {3, 3},
		tile = {7, 7},
		grid = {5, 4},
		layer = 12,
	},
	object091 =
	{
		name = "text_fence",
		sprite = "text_fence",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 0},
		active = {6, 1},
		tile = {8, 7},
		grid = {10, 7},
		layer = 20,
	},
	object092 =
	{
		name = "text_belt",
		sprite = "text_belt",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {1, 2},
		active = {1, 3},
		tile = {9, 7},
		grid = {1, 9},
		layer = 20,
	},
	object093 =
	{
		name = "me",
		sprite = "me",
		sprite_in_root = true,
		unittype = "object",
		tiling = 2,
		type = 0,
		colour = {3, 1},
		tile = {10, 7},
		grid = {3, 6},
		layer = 18,
	},
	object094 =
	{
		name = "text_me",
		sprite = "text_me",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {3, 0},
		active = {3, 1},
		tile = {11, 7},
		grid = {4, 6},
		layer = 20,
	},
	object095 =
	{
		name = "text_swap",
		sprite = "text_swap",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {3, 0},
		active = {3, 1},
		tile = {2, 8},
		grid = {8, 2},
		layer = 20,
	},
	object096 =
	{
		name = "text_pull",
		sprite = "text_pull",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {6, 1},
		active = {6, 2},
		tile = {3, 8},
		grid = {2, 8},
		layer = 20,
	},
	object097 =
	{
		name = "text_on",
		sprite = "text_on",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 7,
		colour = {0, 1},
		active = {0, 3},
		tile = {4, 8},
		grid = {7, 4},
		layer = 20,
	},
	object098 =
	{
		name = "moon",
		sprite = "moon",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {2, 4},
		tile = {6, 8},
		grid = {5, 10},
		layer = 16,
	},
	object099 =
	{
		name = "text_ghost",
		sprite = "text_ghost",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 1},
		active = {4, 2},
		tile = {11, 10},
		grid = {6, 8},
		layer = 20,
	},
	object100 =
	{
		name = "fence",
		sprite = "fence",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {6, 1},
		tile = {8, 8},
		grid = {9, 7},
		layer = 14,
	},
	object101 =
	{
		name = "hedge",
		sprite = "hedge",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {5, 1},
		tile = {0, 9},
		grid = {7, 9},
		layer = 14,
	},
	object102 =
	{
		name = "text_hedge",
		sprite = "text_hedge",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {5, 0},
		active = {5, 1},
		tile = {1, 9},
		grid = {8, 9},
		layer = 20,
	},
	object103 =
	{
		name = "text_level",
		sprite = "text_level",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 0},
		active = {4, 1},
		tile = {2, 9},
		grid = {9, 0},
		layer = 20,
	},
	object104 =
	{
		name = "text_orb",
		sprite = "text_orb",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {4, 0},
		active = {4, 1},
		tile = {3, 9},
		grid = {1, 10},
		layer = 20,
	},
	object105 =
	{
		name = "orb",
		sprite = "orb",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {4, 1},
		tile = {4, 9},
		grid = {0, 10},
		layer = 17,
	},
	object106 =
	{
		name = "text_bonus",
		sprite = "text_bonus",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {4, 0},
		active = {4, 1},
		tile = {5, 9},
		grid = {2, 10},
		layer = 20,
	},
	object107 =
	{
		name = "text_moon",
		sprite = "text_moon",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 1},
		active = {2, 4},
		tile = {6, 9},
		grid = {6, 10},
		layer = 20,
	},
	object108 =
	{
		name = "text_group",
		sprite = "text_group",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {3, 2},
		active = {3, 3},
		tile = {7, 9},
		grid = {10, 0},
		layer = 20,
	},
	object109 =
	{
		name = "text_line",
		sprite = "text_line",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 2},
		active = {0, 3},
		tile = {8, 9},
		grid = {4, 8},
		layer = 20,
	},
	object110 =
	{
		name = "brick",
		sprite = "brick",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {6, 3},
		tile = {9, 9},
		grid = {7, 10},
		layer = 10,
	},
	object111 =
	{
		name = "text_brick",
		sprite = "text_brick",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {6, 0},
		active = {6, 1},
		tile = {10, 9},
		grid = {8, 10},
		layer = 20,
	},
	object112 =
	{
		name = "text_wonder",
		sprite = "text_wonder",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {0, 1},
		active = {0, 3},
		tile = {11, 9},
		grid = {9, 1},
		layer = 20,
	},
	object113 =
	{
		name = "text_eat",
		sprite = "text_eat",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		colour = {2, 1},
		active = {2, 2},
		tile = {0, 10},
		grid = {8, 5},
		layer = 20,
	},
	object114 =
	{
		name = "text_statue",
		sprite = "text_statue",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 0,
		colour = {0, 0},
		active = {0, 1},
		tile = {1, 10},
		grid = {6, 7},
		layer = 20,
	},
	object115 =
	{
		name = "statue",
		sprite = "statue",
		sprite_in_root = true,
		unittype = "object",
		tiling = 0,
		type = 0,
		colour = {0, 1},
		tile = {2, 10},
		grid = {5, 7},
		layer = 17,
	},
	object116 =
	{
		name = "text_facing",
		sprite = "text_facing",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 7,
		argextra = {"right","up","left","down"},
		colour = {0, 2},
		active = {0, 3},
		tile = {9, 10},
		grid = {10, 4},
		layer = 20,
	},
	object117 =
	{
		name = "line",
		sprite = "line",
		sprite_in_root = true,
		unittype = "object",
		tiling = 1,
		type = 0,
		colour = {0, 3},
		tile = {8, 10},
		grid = {3, 8},
		layer = 20,
	},
	object118 =
	{
		name = "text_fear",
		sprite = "text_fear",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 1,
		colour = {2, 1},
		active = {2, 2},
		tile = {7, 10},
		grid = {9, 5},
		layer = 20,
	},
	object119 =
	{
		name = "text_sad",
		sprite = "text_sad",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		colour = {1, 1},
		active = {3, 2},
		tile = {6, 10},
		grid = {10, 1},
		layer = 20,
	},
	object120 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {0, 12},
		grid = {10, 1},
		layer = 20,
	},
	object121 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {1, 12},
		grid = {10, 1},
		layer = 20,
	},
	object122 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {2, 12},
		grid = {10, 1},
		layer = 20,
	},
	object123 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {3, 12},
		grid = {10, 1},
		layer = 20,
	},
	object124 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {4, 12},
		grid = {10, 1},
		layer = 20,
	},
	object125 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {5, 12},
		grid = {10, 1},
		layer = 20,
	},
	object126 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {6, 12},
		grid = {10, 1},
		layer = 20,
	},
	object127 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {7, 12},
		grid = {10, 1},
		layer = 20,
	},
	object128 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {8, 12},
		grid = {10, 1},
		layer = 20,
	},
	object129 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {9, 12},
		grid = {10, 1},
		layer = 20,
	},
	object130 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {10, 12},
		grid = {10, 1},
		layer = 20,
	},
	object131 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {11, 12},
		grid = {10, 1},
		layer = 20,
	},
	object132 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {12, 12},
		grid = {10, 1},
		layer = 20,
	},
	object133 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {0, 13},
		grid = {10, 1},
		layer = 20,
	},
	object134 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {1, 13},
		grid = {10, 1},
		layer = 20,
	},
	object135 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {2, 13},
		grid = {10, 1},
		layer = 20,
	},
	object136 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {3, 13},
		grid = {10, 1},
		layer = 20,
	},
	object137 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {4, 13},
		grid = {10, 1},
		layer = 20,
	},
	object138 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {5, 13},
		grid = {10, 1},
		layer = 20,
	},
	object139 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {6, 13},
		grid = {10, 1},
		layer = 20,
	},
	object140 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {7, 13},
		grid = {10, 1},
		layer = 20,
	},
	object141 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {8, 13},
		grid = {10, 1},
		layer = 20,
	},
	object142 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {9, 13},
		grid = {10, 1},
		layer = 20,
	},
	object143 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {10, 13},
		grid = {10, 1},
		layer = 20,
	},
	object144 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {11, 13},
		grid = {10, 1},
		layer = 20,
	},
	object145 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {12, 13},
		grid = {10, 1},
		layer = 20,
	},
	object146 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {13, 13},
		grid = {10, 1},
		layer = 20,
	},
	object147 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {14, 13},
		grid = {10, 1},
		layer = 20,
	},
	object148 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {15, 13},
		grid = {10, 1},
		layer = 20,
	},
	object149 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {16, 13},
		grid = {10, 1},
		layer = 20,
	},
	object150 =
	{
		name = "cursor",
		sprite = "cursor",
		sprite_in_root = true,
		unittype = "object",
		tiling = -1,
		type = 0,
		colour = {4, 2},
		tile = {5, 10},
		grid = {11, 0},
		layer = 20,
	},
	object151 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {0, 14},
		grid = {10, 1},
		layer = 20,
	},
	object152 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {1, 14},
		grid = {10, 1},
		layer = 20,
	},
	object153 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {2, 14},
		grid = {10, 1},
		layer = 20,
	},
	object154 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {3, 14},
		grid = {10, 1},
		layer = 20,
	},
	object155 =
	{
		name = "default",
		sprite = "default",
		sprite_in_root = true,
		unittype = "text",
		tiling = -1,
		type = 2,
		does_not_exist = true,
		colour = {1, 1},
		active = {3, 2},
		tile = {4, 14},
		grid = {10, 1},
		layer = 20,
	},
}

particletypes =
{
	bubbles =
	{
		amount = 30,
		animation = 0,
		colour = {1, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[YVEL] = math.random(-3,-1)
				
				unit.scaleX = unit.values[YVEL] * -0.33
				unit.scaleY = unit.values[YVEL] * -0.33
			end,
	},
	soot =
	{
		amount = 30,
		animation = 1,
		colour = {0, 1},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[YVEL] = math.random(-3,-1)
				
				unit.scaleX = unit.values[YVEL] * -0.33
				unit.scaleY = unit.values[YVEL] * -0.33
			end,
	},
	sparks =
	{
		amount = 40,
		animation = 1,
		colour = {2, 3},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[YVEL] = math.random(-3,-1)
				
				unit.scaleX = unit.values[YVEL] * -0.23
				unit.scaleY = unit.values[YVEL] * -0.23
				
				local coloury = math.random(2,4)
				
				MF_setcolour(unitid,2,coloury)
				unit.strings[COLOUR] = tostring(2) .. "," .. tostring(coloury)
			end,
	},
	dust =
	{
		amount = 50,
		animation = 1,
		colour = {1, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[YVEL] = math.random(-3,-1)
				
				unit.scaleX = unit.values[YVEL] * -0.33 * 1.1
				unit.scaleY = unit.values[YVEL] * -0.33 * 1.1
			end,
	},
	snow =
	{
		amount = 30,
		animation = 1,
		colour = {0, 3},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[XVEL] = math.random(-50,-10) * 0.1
				unit.values[YVEL] = math.abs(unit.values[XVEL]) * (math.random(5,15) * 0.1)
				
				unit.scaleX = math.abs(unit.values[XVEL]) * 0.2
				unit.scaleY = math.abs(unit.values[XVEL]) * 0.2
				unit.flags[INFRONT] = true
			end,
	},
	clouds =
	{
		amount = 90,
		animation = {2, 12},
		colour = {4, 3},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.scaleX = 1 + math.random(-30,30) * 0.01
				unit.scaleY = unit.scaleX * 0.9
				
				unit.values[YVEL] = 0 - unit.scaleX
				unit.values[XVEL] = 0 - unit.scaleX
			end,
	},
	smoke =
	{
		amount = 30,
		animation = 3,
		colour = {1, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.angle = math.random(0,359)
				
				unit.scaleX = 1 + math.random(-30,30) * 0.01
				unit.scaleY = unit.scaleX
				
				unit.values[YVEL] = -1
				unit.values[DIR] = math.random(-25,25) * 0.05
			end,
	},
	pollen =
	{
		amount = 20,
		animation = 5,
		colour = {1, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[XVEL] = math.random(-20,20) * 0.1
				unit.values[YVEL] = math.random(40,80) * 0.05
				
				local size = math.random(2,5)
				unit.scaleX = size * 0.2
				unit.scaleY = size * 0.2
			end,
	},
	stars =
	{
		amount = 40,
		animation = {6, 7},
		colour = {3, 2},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				unit.values[XVEL] = ((unit.direction - 6) + math.random(0,5) * 0.1) + 0.05
				--unit.values[YVEL] = math.random(40,80) * 0.05
				
				if (unit.direction == 7) then
					MF_setcolour(unitid,1,3)
					
					unit.strings[COLOUR] = tostring(1) .. "," .. tostring(3)
				end
				
				local size = math.random(2,5)
				unit.scaleX = size * 0.2
				unit.scaleY = size * 0.2
			end,
	},
	glitter =
	{
		amount = 60,
		animation = 8,
		colour = {3, 1},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				if (math.random(1,4) == 1) then
					MF_setcolour(unitid,4,2)
					
					unit.strings[COLOUR] = tostring(4) .. "," .. tostring(1)
				end
				
				if (math.random(1,4) == 1) then
					MF_setcolour(unitid,0,3)
					
					unit.strings[COLOUR] = tostring(0) .. "," .. tostring(3)
				end
				
				local size = math.random(2,5)
				unit.scaleX = size * 0.2
				unit.scaleY = size * 0.2
			end,
	},
	leaves =
	{
		amount = 30,
		animation = {9, 10},
		colour = {6, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				if (math.random(1,4) == 1) then
					MF_setcolour(unitid,6,3)
					
					unit.strings[COLOUR] = tostring(6) .. "," .. tostring(3)
				end
				
				local size = math.random(3,6)
				unit.scaleX = size * 0.2
				unit.scaleY = size * 0.2
				
				unit.values[XVEL] = math.random(-30,-10) * 0.1
				unit.values[YVEL] = math.random(0,10) * 0.05
			end,
	},
	rain =
	{
		amount = 50,
		animation = 11,
		colour = {3, 2},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				
				local size = math.random(3,5)
				unit.scaleX = size * 0.2
				unit.scaleY = size * 0.2
				
				unit.values[YVEL] = 80 + math.random(0,10) * 0.1
			end,
	},
	world =
	{
		amount = 40,
		animation = 13,
		colour = {4, 0},
		extra = 
			function(unitid)
				local unit = mmf.newObject(unitid)
				local scale = 0.2 + math.random(0,400) * 0.01
				
				unit.scaleX = scale
				unit.scaleY = scale
				
				unit.values[DIR] = ((math.random(0,1) * 2) - 1) * (2 - scale * 0.5)
			end,
	},
}

namegivingtitles =
{
	[1] = {"namegiving_newworldname","minlen1,maxlen24"},
	[2] = {"namegiving_changelevelname","minlen1,maxlen24"},
	[3] = {"namegiving_newlevelname","minlen1,maxlen24"},
	[4] = {"namegiving_changeunitname","lower,minlen1,maxlen24"},
	[5] = {"namegiving_changeunittype","lower,minlen1,maxlen12"},
	[6] = {"namegiving_newthemename","minlen1,maxlen16"},
	[7] = {"namegiving_objectsearchstring","lower,minlen1,maxlen12"},
	[8] = {"namegiving_newthemename","minlen1,maxlen16"},
	[9] = {"namegiving_levelauthor","maxlen16"},
	[10] = {"namegiving_worldauthor","maxlen16"},
	[11] = {"namegiving_changeworldname","minlen1,maxlen24"},
	[12] = {"namegiving_downloadcode","upper,maxlen9"},
	[13] = {"namegiving_levelsearchstring","minlen1,maxlen24"},
	[14] = {"namegiving_spritesearchstring","lower,minlen1,maxlen24"},
	[15] = {"namegiving_slotname","minlen1,maxlen16"},
	[16] = {"namegiving_subtitle","maxlen26"},
	[17] = {"namegiving_editor_autopick","lower"},
	[18] = {"namegiving_currobjlist_autoadd","lower"},
	[19] = {"namegiving_changeunittype",""},
}

bicons =
{
	controls_disablegamepad_off = 0,
	controls_disablegamepad_on = 1,
	r_arrow = 3,
	u_arrow = 4,
	l_arrow = 5,
	d_arrow = 6,
	cog = 7,
	dr_arrow = 10,
	du_arrow = 11,
	dl_arrow = 12,
	dd_arrow = 13,
	t_pen = 14,
	t_line = 15,
	t_fill = 16,
	t_rect = 17,
	t_frect = 18,
	t_erase = 19,
	t_select = 20,
	cross = 21,
	search = 22,
	rsearch = 23,
	tags = 24,
	o_add = 25,
	o_del = 26,
	o_edit = 27,
	l_separate = 28,
	l_pairs = 29,
	swap = 30,
	m_start = 31,
	m_settings = 32,
	m_credits = 33,
	m_erase = 34,
	m_return = 35,
	m_settings_grid_no = 36,
	m_settings_grid = 37,
	m_settings_wobble_no = 38,
	m_settings_wobble = 39,
	m_settings_particles_no = 40,
	m_settings_particles = 41,
	m_settings_shake_no = 42,
	m_settings_shake = 43,
	yes = 44,
	no = 45,
	m_settings_language = 46,
	m_settings_hand_right = 47,
	m_settings_hand_left = 48,
	m_settings_pointers_on = 49,
	m_settings_pointers_off = 50,
}

lookup_table =
{
	-- a-z
	97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
	-- numbers
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
	-- punctuation etc
    45, 46, 63, 33, 44, 39, 58, 95, 62, 60, 40, 41, 38, 43, 129315, 164, 163, 123, 125, 47, 36, 32, 8217, 37, 8230, 183, 161, 191,
	-- umlauts etc 
	228, 246, 269, 271, 283, 287, 328, 345, 353, 357, 365, 382, 279, 230, 339, 235, 239, 252, 224, 232, 236, 249, 226, 265, 234, 285, 293, 238, 309, 244, 349, 251, 227, 241, 245, 261, 281, 322, 225, 263, 233, 237, 243, 324, 347, 250, 253, 378, 380, 229, 367, 248, 231, 351,
	-- cyrillic letters
	1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103,
	-- misc
	1104, 1105,
	-- vietnamese
	7871, 7879,
	-- misc
	193, 268, 305, 304, 26085, 26412, 35486, 188, 190, 352, 9658, 9650, 9668, 9660, 128169, 9807, 9796, 26059, 24459, 26143, 33828, 20013, 25991, 31616, 20307,
	-- capital letters a-z
	65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 59, 34,
	-- capitalized umlauts etc
	197, 201, 217, 223, 7838, 284, 348, 346, 379, 171, 187, 124, 42,
	-- capital cyrillic letters
	1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071,
}

play_data =
{
	tunes = 
	{
		baba = "beep5",
		box = "drum_kick",
		rock = "drum_snare",
		key = "drum_hat",
		keke = "tune_blop",
		skull = "tune_short2",
		piano = "piano2",
		guitar = "guitar2",
		sax = "sax2",
		drum = "drum_snare",
		hihat = "drum_hat",
		trumpet = "trumpet",
		frog = "frog",
		fire = "burn6",
		lava = "burn5",
		crystal = "clink",
		gem = "clink",
		shell = "clink",
		vase = "clink",
		mirror = "clink",
		car = "honk",
		water = "splash",
		fish = "splash",
		bubble = "splash",
		crab = "splash",
		bottle = "splash",
		bunny = "boing",
		arm = "drum_kick",
	},
	freqs = 
	{
	},
}

controlnames =
{
	-- Keyboard ingame controls
	{"right","up","left","down","right2","up2","left2","down2","idle","undo","restart","confirm","pause","idle2","undo2","restart2","confirm2"},
	-- Gamepad ingame controls
	{"move","move2","idle","undo","restart","confirm","pause","idle2","undo2","restart2","confirm2"},
	-- Editor gamepad controls
	{"move","rotate","place","copy","drag","undo","scrollright_hotbar","scrollleft_hotbar","scrollright_tool","scrollleft_tool","currobjlist","quickmenu","swap","scrollright_layer","scrollleft_layer","moveall","altpress","emptytile","showdir","cut","autopick","pickempty","lock","empty_hotbar","save","test"},
	-- Currobjlist gamepad controls
	{"move","select","swap","drag","tooltip","scrollleft","scrollright","closemenu","tags","remove","edit","addnew","search","autoadd","altsave","settings","menu"},
}