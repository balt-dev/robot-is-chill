function mapcursor_enter(varsunitid)
	local cursors = getunitswitheffect("select",true)
	local varsunit = mmf.newObject(varsunitid)
	local entering = {}
	
	for i,unit in ipairs(cursors) do
		local targetfound = MF_findunit_fixed(unit.values[CURSOR_ONLEVEL])
		
		if targetfound then
			local allhere = findallhere(unit.values[XPOS],unit.values[YPOS],unit.fixed)
			
			for a,b in ipairs(allhere) do
				local lunit = mmf.newObject(b)
				
				if (string.len(lunit.strings[U_LEVELFILE]) > 0) and (string.len(lunit.strings[U_LEVELNAME]) > 0) and (generaldata.values[IGNORE] == 0) and (lunit.values[COMPLETED] > 1) then
					local valid = true
					
					for c,d in ipairs(cursors) do
						if (d.fixed == b) then
							valid = false
							break
						end
					end
					
					if valid then
						table.insert(entering, {b, lunit.strings[U_LEVELNAME], lunit.strings[U_LEVELFILE]})
					end
				end
			end
		end
	end
	
	if (#entering > 0) then
		dolog("end")
	end
	
	if (#entering == 1) then
		findpersists("levelentry")
		generaldata2.values[UNLOCK] = 0
		generaldata2.values[UNLOCKTIMER] = 0
		varsunit.values[1] = entering[1][1]
		MF_loop("enterlevel", 1)
	elseif (#entering > 0) then
		findpersists("levelentry")
		MF_menuselector_hack(1)
		submenu("enterlevel_multiple",entering)
		print("Trying to enter multiple levels!")
	end
end