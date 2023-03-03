MF_winbackup = MF_win
enddata = nil
persists = {}
levelpersist = {}
persistreverts = {}
persistrevert = nil
persistbaserules = {}
persistbaserules["42level"] = {{"text","is","push"},{"level","is","stop"},{"cursor","is","select"}}
persistbaserulestoadd = {}
bgimages={"persist2nd", "persist3rd", "persistinception", "prison", "persistmortality", "fleeting", "wug", "persist2", "changeless", "heat", "christmas", "persistbrick", "invert", "frankenstein", "mirror", "starseeker", "welcome", "turing", "creation", "beta", "banana", "concentrate", "babsmall", "lastresort", "tower", "DONOTSPEAKHISNAME", "sticky", "word", "alphababa", "pond", "daybreak2"}
bgloaded = bgloaded or 1

prevpersists = {}
prevlevelpersist = {}

exitedlevel = false

function getprevpersists()
	persists = {}
	if prevpersists[currentlevel] ~= nil then
		for i,v in pairs(prevpersists[currentlevel]) do
			persists[i] = {}
			for j,k in pairs(v) do
				if type(k) == "table" then
					local ktable = {}
					for l,m in ipairs(k) do
						ktable[l] = m
					end
					persists[i][j] = ktable
				else
					persists[i][j] = k
				end
			end
		end
	end

	levelpersist = {}
	if prevlevelpersist[currentlevel] ~= nil then
		for i,j in ipairs(prevlevelpersist[currentlevel]) do
			levelpersist[i] = j
		end
	end
end

--for optimisation purposes, this remembers ONLY the oldest object and will not try subsequent ones if X IS NOT Y is present in future levels
--will fix this if I make a general release of this
function getreverts(unit)
	persistrevert = nil
	if persistreverts ~= nil then
		id = unit.values[ID]
		persistrevert = persistreverts[id]
	end
	
	local originalname = ""
	local oldest = 0
	local source = ""
	local baseid = -1
	local nametotest = getname(unit)
	local ingameid = unit.values[ID]
	local baseingameid = unit.values[ID]
	local unitid = unit.values[ID]
	
	for i=1,#undobuffer do
		local curr = undobuffer[i]
		
		for a,b in ipairs(curr) do
			if (b[1] == "create") and (b[3] == baseingameid) then
				oldest = i
				originalname = b[2]
				source = b[5]
				baseid = b[4]
				break
			end
		end
	end
	
	local oldestundo = undobuffer[oldest] or {}
	
	for i,v in ipairs(oldestundo) do
		if (v[1] == "remove") and ((v[6] == unit.values[ID]) or (v[7] == unit.values[ID]) or ((baseid == v[6]) and (baseid > -1))) then
			if (hasfeature(nametotest,"is","not " .. v[2],unitid,x,y) == nil) then
				originalname = v[2]
				break
			end
		end
	end
	if persistrevert ~= nil then
		originalname = persistrevert
	end
	if (string.len(originalname) > 0) then
		return originalname
	else
		return nil
	end
end

--permanently add rules made from text with the "baserule" property. for optimisation doesn't care about conditions. this is easy to fix though
function findpersistrules()
	for i,rules in ipairs(visualfeatures) do
		local conds = rules[2]
		local ids = rules[3]
		local tags = rules[4]
		
		local fullpersist = true
		for a,b in ipairs(ids) do
			local dunit = mmf.newObject(b[1])
			
			if not hasfeature(getname(dunit), "is", "baserule", dunit.fixed) then
				fullpersist = false
				break
			end
		end
		

		if (fullpersist == true) and (#conds==0) then
			--do not persist rules that are disabled
			if not hasfeature(rules[1][1],rules[1][2],"not "..rules[1][3]) and not (objectlist[rules[1][3]] ~= nil and hasfeature(rules[1][1],"is",rules[1][1])) then
				table.insert(persistbaserulestoadd,{rules[1][1],rules[1][2],rules[1][3]})
			end
		end
	end
	if #persistbaserulestoadd ~= 0 then
		if persistbaserules == nil then
			persistbaserules = {}
		end
		--if we already have rules for this level, don't add anymore; prevents duplicate entries due to hook for WIN getting called more than once with multiple YOU objects, or level being transformed into multiple things at once, etc
		persistbaserules[currentlevel] = persistbaserulestoadd
		persistbaserulestoadd = {}
	end
end

function findpersists(reason)
	if reason ~= "levelentry" then
		prevpersists = {}
		prevlevelpersist = {}
	end
	findpersistrules()
	--update persistent object info
	persists = {}
	levelpersist = {}
	if hasfeature("level","is","persist",1) then
		levelpersist = {Xoffset-Xoffsetorig,Yoffset,mapdir,maprotation}
		persistxoffset = 0
		persistyoffset = 0
	else
		persistxoffset = (Xoffset-Xoffsetorig)/tilesize
		persistyoffset = Yoffset/tilesize
	end
		
	ispersist = getunitswitheffect("persist",delthese)
	for id,unit in ipairs(ispersist) do
		x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
		name = getname(unit)
		leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
		persistobjectdata = {unit.strings[UNITNAME],x+(persistxoffset),y+(persistyoffset),dir,x+(persistxoffset),y+(persistyoffset),nil,nil,leveldata,getreverts(unit)}
		--persistobjectdata = {unit.strings[UNITNAME],x,y,dir,unit.values[ID],y,nil,nil,leveldata,unit.followed,unit.back_init}
		table.insert(persists,(persistobjectdata))
	end
end

mod_hook_functions =
{
	level_start =
	{
		levelstartfunction = function()
			currentlevel = generaldata.strings[CURRLEVEL]
			if currentlevel == "41level" then
				--preload images to reduce lag during ending
				if bgloaded <= #bgimages then
					MF_loadbackimage(bgimages[bgloaded])
					MF_movebackimage(-1000,-1000)
					bgloaded=bgloaded+1
				end
			elseif currentlevel == "42level" or currentlevel == "55level" then
				leveltree = {"41level",currentlevel}
			end
			if currentlevel == "55level" then
				ending = ending2
				generaldata2.values[ZOOM] = 2
				--preload images to reduce lag during ending
				while bgloaded <= #bgimages do
					MF_loadbackimage(bgimages[bgloaded])
					bgloaded=bgloaded+1
					MF_movebackimage(-1000,-1000)
				end
			else
				generaldata2.values[ZOOM] = 1
			end

			if exitedlevel == true then
				getprevpersists()
			end
			exitedlevel = false
			
			--check for persistent rules
			for level,v in pairs(persistbaserules) do
				for j,rule in ipairs(v) do
					--need to be able to create objects that aren't already in the level
					if (unitreference[rule[3]] ~= nil or rule[3] == "empty" or rule[3] == "text") then
						objectlist[rule[3]] = 1
					end
				end
			end
			
		
			Xoffsetorig = Xoffset
			Yoffsetorig = Yoffset
			--create persistent objects from previous level
			prevpersists[currentlevel] = {}
			if persists ~= nil then
				for i,v in pairs(persists) do
					--do not bring persistent objects if their persistence is disabled in the new level
					if hasfeature(v[1], "is","not persist") == nil and not (hasfeature("all", "is","not persist") and not (string.sub(v[1], 1, 5) == "text_")) and not ((string.sub(v[1], 1, 5) == "text_") and (hasfeature("text", "is","not persist"))) then
						create(v[1],v[2],v[3],v[4],v[5],v[6],nil,true,v[9],v[10])
						prevpersists[currentlevel][i] = {}
						--backup to prevpersists; if entry is a table (leveldata) make sure it copies the whole table
						for j,k in pairs(v) do
							if type(k) == "table" then
								local ktable = {}
								for l,m in ipairs(k) do
									ktable[l] = m
								end
								prevpersists[currentlevel][i][j] = ktable
							else
								prevpersists[currentlevel][i][j] = k
							end
						end
					end
				end
			end
			
			prevlevelpersist[currentlevel] = {}
			if levelpersist ~= nil then
				for i,j in ipairs(levelpersist) do
					prevlevelpersist[currentlevel][i] = j
				end
				if levelpersist[1] ~= nil and hasfeature("level", "is","not persist",1) == nil then
					MF_scrollroom(levelpersist[1],levelpersist[2])
					mapdir = levelpersist[3]
					maprotation = levelpersist[4]
					MF_levelrotation(maprotation)
				end
			end
			
			persistbaserulestoadd = {}
			updatecode = 1
			code(alreadyrun_) --reparse any new rules formed by persisted text
			animate()
		end
	},
	level_end =
	{	--if the player exits the level via the menu
		blankpersists = function()
			exitedlevel = true
		end
	},
	level_restart = 
	{

	},
	level_win =
	{
		findpersists
	},
	rule_update =
	{
		-- Functions added to this table will be called when rules are updated. Extra data: {is_this_a_repeated_update [false = no, true = yes]} <-- the extra variable is true e.g. when Word-related rules require the rules to be refreshed multiple times in a row.
	},
	command_given =
	{
		-- Functions added to this table will be called when a player command starts resolving. Extra data: {command, player_id}
	},
	turn_auto =
	{
		-- Functions added to this table will be called when a turn starts resolving due to 'Level Is Auto'. Extra data: {player_1_direction, player_2_direction, is_a_player_moving [false = no, true = yes]}
	},
	turn_end =
	{
		preloadbgs = function()
			if currentlevel == "41level" then
				--preload images to reduce lag during ending
				if bgloaded <= #bgimages then
					MF_loadbackimage(bgimages[bgloaded])
					MF_movebackimage(-1000,-1000)
					bgloaded=bgloaded+1
				end
			end
		end
	},
	undoed =
	{
		-- Functions added to this table will be called when the player does an undo input.
	},
}

function do_mod_hook(name,extra_)
	local extra = extra_ or {}
	
	if (mod_hook_functions[name] ~= nil) then
		for i,v in pairs(mod_hook_functions[name]) do
			if (type(v) == "function") then
				v(extra)
			end
		end
	end
end