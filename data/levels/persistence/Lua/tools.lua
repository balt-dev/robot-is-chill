function create(name,x,y,dir,oldx_,oldy_,float_,skipundo_,leveldata_,customdata)
	local oldx,oldy,float = x,y,0
	local tileid = x + y * roomsizex
	
	if (oldx_ ~= nil) then
		oldx = oldx_
	end
	
	if (oldy_ ~= nil) then
		oldy = oldy_
	end
	
	if (float_ ~= nil) then
		float = float_
	end
	
	local skipundo = skipundo_ or false
	
	local unitname = unitreference[name]
	
	if (unitname == nil) then
		if name == "level" then --unitreference for level sometimes shows as nil, not sure why
			unitname = "error"
		else
			unitname = "error"
			MF_alert("Couldn't find object for " .. tostring(name) .. "!")
		end
	end
	
	local newunitid = MF_emptycreate(unitname,oldx,oldy)
	local newunit = mmf.newObject(newunitid)
	
	local id = newid()
	
	newunit.values[ONLINE] = 1
	newunit.values[XPOS] = x
	newunit.values[YPOS] = y
	newunit.values[DIR] = dir
	newunit.values[ID] = id
	newunit.values[FLOAT] = float
	newunit.flags[CONVERTED] = true


	persistreverts[id]=customdata

	
	if (leveldata_ ~= nil) and (#leveldata_ > 0) then
		newunit.strings[U_LEVELFILE] = leveldata_[1]
		newunit.strings[U_LEVELNAME] = leveldata_[2]
		newunit.flags[MAPLEVEL] = leveldata_[3]
		newunit.values[VISUALLEVEL] = leveldata_[4]
		newunit.values[VISUALSTYLE] = leveldata_[5]
		newunit.values[COMPLETED] = leveldata_[6]
		
		newunit.strings[COLOUR] = leveldata_[7]
		newunit.strings[CLEARCOLOUR] = leveldata_[8]
		
		if (newunit.className == "level") then
			if (#leveldata_[1] > 0) then
				newunit.values[COMPLETED] = math.max(leveldata_[6], 2)
			else
				newunit.values[COMPLETED] = math.max(leveldata_[6], 1)
			end
			
			if (#leveldata_[7] == 0) or (#leveldata_[8] == 0) then
				newunit.strings[COLOUR] = "1,2"
				newunit.strings[CLEARCOLOUR] = "1,3"
				MF_setcolour(newunitid,1,2)
			else
				local c = MF_parsestring(leveldata_[7])
				MF_setcolour(newunitid,c[1],c[2])
			end
		elseif (#leveldata_[7] > 0) then
			local c = MF_parsestring(leveldata_[7])
			MF_setcolour(newunitid,c[1],c[2])
		end
	end
	
	newunit.flags[9] = true
	
	if (skipundo == false) then
		addundo({"create",name,id,-1,"create",x,y,dir})
	end
	
	addunit(newunitid)
	addunitmap(newunitid,x,y,newunit.strings[UNITNAME])
	dynamic(newunitid)
	
	local testname = getname(newunit)
	if (hasfeature(testname,"is","word",newunitid,x,y) ~= nil) then
		updatecode = 1
	end
	
	return newunit.fixed,id
end