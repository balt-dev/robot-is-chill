--The only things I've edited here is for the ending
function block(small_)
	local delthese = {}
	local doned = {}
	local unitsnow = #units
	local removalsound = 1
	local removalshort = ""
	
	local small = small_ or false
	
	local doremovalsound = false
	
	if (small == false) then
		if (generaldata2.values[ENDINGGOING] == 0) then
			local isdone = getunitswitheffect("done",false,delthese)
			
			for id,unit in ipairs(isdone) do
				table.insert(doned, unit)
			end
			
			if (#doned > 0) then
				setsoundname("turn",10)
			end
		end
		
		local ismore = getunitswitheffect("more",false,delthese)
		
		for id,unit in ipairs(ismore) do
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local name = getname(unit)
			local doblocks = {}
			
			for i=1,4 do
				local drs = ndirs[i]
				ox = drs[1]
				oy = drs[2]
				
				local valid = true
				local obs = findobstacle(x+ox,y+oy)
				local tileid = (x+ox) + (y+oy) * roomsizex
				
				if (#obs > 0) then
					for a,b in ipairs(obs) do
						if (b == -1) then
							valid = false
						elseif (b ~= 0) and (b ~= -1) then
							local bunit = mmf.newObject(b)
							local obsname = getname(bunit)
							
							local obsstop = hasfeature(obsname,"is","stop",b,x+ox,y+oy)
							local obspush = hasfeature(obsname,"is","push",b,x+ox,y+oy)
							local obspull = hasfeature(obsname,"is","pull",b,x+ox,y+oy)
							
							if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) or (obsname == name) then
								valid = false
								break
							end
						end
					end
				else
					local obsstop = hasfeature("empty","is","stop",2,x+ox,y+oy)
					local obspush = hasfeature("empty","is","push",2,x+ox,y+oy)
					local obspull = hasfeature("empty","is","pull",2,x+ox,y+oy)
					
					if (obsstop ~= nil) or (obspush ~= nil) or (obspull ~= nil) then
						valid = false
					end
				end
				
				if valid then
					local newunit = copy(unit.fixed,x+ox,y+oy)
				end
			end
		end
	end
	
	local isplay = getunitswithverb("play",delthese)
	
	for id,ugroup in ipairs(isplay) do
		local sound_freq = ugroup[1]
		local sound_units = ugroup[2]
		local sound_name = ugroup[3]
		
		if (#sound_units > 0) then
			local ptunes = play_data.tunes
			local pfreqs = play_data.freqs
			local multisample = false
			
			local tune = "beep"
			local freq = pfreqs[sound_freq] or 24000
			
			if (ptunes[sound_name] ~= nil) then
				if (type(ptunes[sound_name]) == "string") then
					tune = ptunes[sound_name]
				elseif (type(ptunes[sound_name]) == "table") then
					multisample = true
				end
			end
			
			if multisample then
				if (tonumber(string.sub(sound_freq, -1)) ~= nil) then
					local octave = math.min(math.max(tonumber(string.sub(sound_freq, -1)) - base_octave, 1), #ptunes[sound_name])
					local truefreq = string.sub(sound_freq, 1, #sound_freq-1)
					tune = ptunes[sound_name][octave]
					freq = pfreqs[truefreq]
				else
					tune = ptunes[sound_name][2]
				end
			end
			
			-- MF_alert(sound_name .. " played at " .. tostring(freq) .. " (" .. sound_freq .. ")")
			
			MF_playsound_freq(tune,freq)
			setsoundname("turn",11,nil)
			
			if (sound_name ~= "empty") then
				for a,unit in ipairs(sound_units) do
					local x,y = unit.values[XPOS],unit.values[YPOS]
					
					MF_particles("music",unit.values[XPOS],unit.values[YPOS],1,0,3,3,1)
				end
			end
		end
	end
	
	local issink = getunitswitheffect("sink",false,delthese)
	
	for id,unit in ipairs(issink) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local tileid = x + y * roomsizex
		
		if (unitmap[tileid] ~= nil) then
			local water = findallhere(x,y)
			local sunk = false
			
			if (#water > 0) then
				for a,b in ipairs(water) do
					if floating(b,unit.fixed,x,y) then
						if (b ~= unit.fixed) then
							local dosink = true
							
							for c,d in ipairs(delthese) do
								if (d == unit.fixed) or (d == b) then
									dosink = false
								end
							end
							
							local safe1 = issafe(b)
							local safe2 = issafe(unit.fixed)
							
							if safe1 and safe2 then
								dosink = false
							end
							
							if dosink then
								generaldata.values[SHAKE] = 3
								
								if (safe1 == false) then
									table.insert(delthese, b)
								end
								
								local pmult,sound = checkeffecthistory("sink")
								removalshort = sound
								removalsound = 3
								local c1,c2 = getcolour(unit.fixed)
								MF_particles("destroy",x,y,15 * pmult,c1,c2,1,1)
								
								if (b ~= unit.fixed) and (safe2 == false) then
									sunk = true
								end
							end
						end
					end
				end
			end
			
			if sunk then
				table.insert(delthese, unit.fixed)
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local isweak = getunitswitheffect("weak",false,delthese)
	
	for id,unit in ipairs(isweak) do
		if (issafe(unit.fixed) == false) and (unit.new == false) then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			local stuff = findallhere(x,y)
			
			if (#stuff > 0) then
				for i,v in ipairs(stuff) do
					if floating(v,unit.fixed,x,y) then
						local vunit = mmf.newObject(v)
						local thistype = vunit.strings[UNITTYPE]
						if (v ~= unit.fixed) then
							local pmult,sound = checkeffecthistory("weak")
							MF_particles("destroy",x,y,5 * pmult,0,3,1,1)
							removalshort = sound
							removalsound = 1
							generaldata.values[SHAKE] = 4
							table.insert(delthese, unit.fixed)
							break
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local ismelt = getunitswitheffect("melt",false,delthese)
	
	for id,unit in ipairs(ismelt) do
		local hot = findfeature(nil,"is","hot")
		local x,y = unit.values[XPOS],unit.values[YPOS]
		
		if (hot ~= nil) then
			for a,b in ipairs(hot) do
				local lava = findtype(b,x,y,0)
			
				if (#lava > 0) and (issafe(unit.fixed) == false) then
					for c,d in ipairs(lava) do
						if floating(d,unit.fixed,x,y) then
							local pmult,sound = checkeffecthistory("hot")
							MF_particles("smoke",x,y,5 * pmult,0,1,1,1)
							generaldata.values[SHAKE] = 5
							removalshort = sound
							removalsound = 9
							table.insert(delthese, unit.fixed)
							break
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local isyou = getunitswitheffect("you",false,delthese)
	local isyou2 = getunitswitheffect("you2",false,delthese)
	local isyou3 = getunitswitheffect("3d",false,delthese)
	
	for i,v in ipairs(isyou2) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou3) do
		table.insert(isyou, v)
	end
	
	for id,unit in ipairs(isyou) do
		local x,y = unit.values[XPOS],unit.values[YPOS]
		local defeat = findfeature(nil,"is","defeat")
		
		if (defeat ~= nil) then
			for a,b in ipairs(defeat) do
				if (b[1] ~= "empty") then
					local skull = findtype(b,x,y,0)
					
					if (#skull > 0) and (issafe(unit.fixed) == false) then
						for c,d in ipairs(skull) do
							local doit = false
							
							if (d ~= unit.fixed) then
								if floating(d,unit.fixed,x,y) then
									local kunit = mmf.newObject(d)
									local kname = getname(kunit)
									
									local weakskull = hasfeature(kname,"is","weak",d)
									
									if (weakskull == nil) or ((weakskull ~= nil) and issafe(d)) then
										doit = true
									end
								end
							else
								doit = true
							end
							
							if doit then
								local pmult,sound = checkeffecthistory("defeat")
								MF_particles("destroy",x,y,5 * pmult,0,3,1,1)
								generaldata.values[SHAKE] = 5
								removalshort = sound
								removalsound = 1
								table.insert(delthese, unit.fixed)
							end
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local isshut = getunitswitheffect("shut",false,delthese)
	
	for id,unit in ipairs(isshut) do
		local open = findfeature(nil,"is","open")
		local x,y = unit.values[XPOS],unit.values[YPOS]
		
		if (open ~= nil) then
			for i,v in ipairs(open) do
				local key = findtype(v,x,y,0)
				
				if (#key > 0) then
					local doparts = false
					for a,b in ipairs(key) do
						if (b ~= 0) and floating(b,unit.fixed,x,y) then
							if (issafe(unit.fixed) == false) then
								generaldata.values[SHAKE] = 8
								table.insert(delthese, unit.fixed)
								doparts = true
								online = false
							end
							
							if (b ~= unit.fixed) and (issafe(b) == false) then
								table.insert(delthese, b)
								doparts = true
							end
							
							if doparts then
								local pmult,sound = checkeffecthistory("unlock")
								setsoundname("turn",7,sound)
								MF_particles("unlock",x,y,15 * pmult,2,4,1,1)
							end
							
							break
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	local iseat = getunitswithverb("eat",delthese)
	local iseaten = {}
	
	for id,ugroup in ipairs(iseat) do
		local v = ugroup[1]
		
		if (ugroup[3] ~= "empty") then
			for a,unit in ipairs(ugroup[2]) do
				local x,y = unit.values[XPOS],unit.values[YPOS]
				local things = findtype({v,nil},x,y,unit.fixed)
				
				if (#things > 0) then
					for a,b in ipairs(things) do
						if (issafe(b) == false) and floating(b,unit.fixed,x,y) and (b ~= unit.fixed) and (iseaten[b] == nil) then
							generaldata.values[SHAKE] = 4
							table.insert(delthese, b)
							
							iseaten[b] = 1
							
							local pmult,sound = checkeffecthistory("eat")
							MF_particles("eat",x,y,5 * pmult,0,3,1,1)
							removalshort = sound
							removalsound = 1
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	if (small == false) then
		local ismake = getunitswithverb("make",delthese)
		
		for id,ugroup in ipairs(ismake) do
			local v = ugroup[1]
			
			for a,unit in ipairs(ugroup[2]) do
				local x,y,dir,name = 0,0,4,""
				
				local leveldata = {}
				
				if (ugroup[3] ~= "empty") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					name = getname(unit)
					leveldata = {unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.flags[MAPLEVEL],unit.values[VISUALLEVEL],unit.values[VISUALSTYLE],unit.values[COMPLETED],unit.strings[COLOUR],unit.strings[CLEARCOLOUR]}
				else
					x = math.floor(unit % roomsizex)
					y = math.floor(unit / roomsizex)
					name = "empty"
					dir = emptydir(x,y)
				end
				
				if (dir == 4) then
					dir = fixedrandom(0,3)
				end
				
				local exists = false
				
				if (v ~= "text") and (v ~= "all") then
					for b,mat in pairs(objectlist) do
						if (b == v) then
							exists = true
						end
					end
				else
					exists = true
				end
				
				if exists then
					local domake = true
					
					if (name ~= "empty") then
						local thingshere = findallhere(x,y)
						
						if (#thingshere > 0) then
							for a,b in ipairs(thingshere) do
								local thing = mmf.newObject(b)
								local thingname = thing.strings[UNITNAME]
								
								if (thing.flags[CONVERTED] == false) and ((thingname == v) or ((thing.strings[UNITTYPE] == "text") and (v == "text"))) then
									domake = false
								end
							end
						end
					end
					
					if domake then
						if (findnoun(v,nlist.short) == false) then
							create(v,x,y,dir,x,y,nil,nil,leveldata)
						elseif (v == "text") then
							if (name ~= "text") and (name ~= "all") then
								create("text_" .. name,x,y,dir,x,y,nil,nil,leveldata)
								updatecode = 1
							end
						elseif (string.sub(v, 1, 5) == "group") then
							local mem = findgroup(v)
							
							for c,d in ipairs(mem) do
								create(d,x,y,dir,x,y,nil,nil,leveldata)
							end
						end
					end
				end
			end
		end
		
		for i,unit in ipairs(doned) do
			addundo({"done",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.fixed,unit.values[FLOAT]})
			updateundo = true
			
			unit.values[FLOAT] = 2
			unit.values[EFFECTCOUNT] = math.random(-10,10)
			unit.values[POSITIONING] = 7
			unit.flags[DEAD] = true
			
			delunit(unit.fixed)
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	isyou = getunitswitheffect("you",false,delthese)
	isyou2 = getunitswitheffect("you2",false,delthese)
	isyou3 = getunitswitheffect("3d",false,delthese)
	
	for i,v in ipairs(isyou2) do
		table.insert(isyou, v)
	end
	
	for i,v in ipairs(isyou3) do
		table.insert(isyou, v)
	end
	
	for id,unit in ipairs(isyou) do
		if (unit.flags[DEAD] == false) and (delthese[unit.fixed] == nil) then
			local x,y = unit.values[XPOS],unit.values[YPOS]
			
			if (small == false) then
				local bonus = findfeature(nil,"is","bonus")
				
				if (bonus ~= nil) then
					for a,b in ipairs(bonus) do
						if (b[1] ~= "empty") then
							local flag = findtype(b,x,y,0)
							
							if (#flag > 0) then
								for c,d in ipairs(flag) do
									if floating(d,unit.fixed,x,y) then
										local pmult,sound = checkeffecthistory("bonus")
										MF_particles("bonus",x,y,10 * pmult,4,1,1,1)
										removalshort = sound
										removalsound = 2
										MF_playsound("bonus")
										MF_bonus(1)
										addundo({"bonus",1})
										generaldata.values[SHAKE] = 5
										table.insert(delthese, d)
									end
								end
							end
						end
					end
				end
				
				local ending = findfeature(nil,"is","end")
				
				if (ending ~= nil) then
					for a,b in ipairs(ending) do
						if (b[1] ~= "empty") then
							local flag = findtype(b,x,y,0)
							
							if (#flag > 0) then
								for c,d in ipairs(flag) do
									if floating(d,unit.fixed,x,y) and (generaldata.values[MODE] == 0) then
										if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) then
											MF_particles("unlock",x,y,10,1,4,1,1)
											MF_end(unit.fixed,d)
											break
										elseif (editor.values[INEDITOR] ~= 0) then
											local pmult = checkeffecthistory("win")
									
											MF_particles("win",x,y,10 * pmult,2,4,1,1)
											MF_end_single()
											MF_win()
											break
										else
											local pmult = checkeffecthistory("win")
											
											local mods_run = do_mod_hook("levelpack_end", {})
											
											if (mods_run == false) then
												MF_particles("win",x,y,10 * pmult,2,4,1,1)
												MF_end_single()
												MF_win()
												MF_credits(1)
											end
											break
										end
									end
								end
							end
						end
					end
				end
			end
			
			local win = findfeature(nil,"is","win")
			
			if (win ~= nil) then
				for a,b in ipairs(win) do
					if (b[1] ~= "empty") then
						local flag = findtype(b,x,y,0)
						if (#flag > 0) then
							for c,d in ipairs(flag) do
								if floating(d,unit.fixed,x,y) and (hasfeature(b[1],"is","done",d,x,y) == nil) and (hasfeature(b[1],"is","end",d,x,y) == nil) then
									local pmult = checkeffecthistory("win")
									
									MF_particles("win",x,y,10 * pmult,2,4,1,1)
									if not hasfeature(getname(unit), "is", "baserule", unit.fixed) then
										MF_win()
									else
										--done effect
										addundo({"done",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.fixed,unit.values[FLOAT]})
										updateundo = true
										
										unit.values[FLOAT] = 2
										unit.values[EFFECTCOUNT] = 1
										unit.values[POSITIONING] = 7
										unit.flags[DEAD] = true
										
										delunit(unit.fixed)
										persistbaserules = {}
										persistbaserulestoadd = {}
										MF_playsound("dididi")
										MF_end(1,2)
									end
									break
								end
							end
						end
					end
				end
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	--stop out of bounds check during credits
	if not (currentlevel == "55level") then
		for i,unit in ipairs(units) do
			if (inbounds(unit.values[XPOS],unit.values[YPOS],1) == false) then
				--MF_alert("DELETED!!!")
				table.insert(delthese, unit.fixed)
			end
		end
	end
	
	delthese,doremovalsound = handledels(delthese,doremovalsound)
	
	if (small == false) then
		local iscrash = getunitswitheffect("crash",false,delthese)
		
		if (#iscrash > 0) then
			HACK_INFINITY = 200
			destroylevel("infinity")
			return
		end
	end
	
	if doremovalsound then
		setsoundname("removal",removalsound,removalshort)
	end
end

function levelblock()
	local unlocked = false
	local things = {}
	local donethings = {}
	
	local emptythings = {}
	
	if (destroylevel_check == false) then
		if (featureindex["level"] ~= nil) then
			for i,v in ipairs(featureindex["level"]) do
				table.insert(things, v)
			end
		end
		
		if (featureindex["empty"] ~= nil) then
			for i,v in ipairs(featureindex["empty"]) do
				local rule = v[1]
				
				if (rule[1] == "empty") and ((rule[2] == "is") or (rule[2] == "eat")) then
					table.insert(emptythings, v)
				end
			end
		end
		
		local lstill = isstill_or_locked(1,nil,nil,mapdir)
		local lsleep = issleep(1)
		local lsafe = issafe(1)
		local emptybonus = false
		local emptydone = false
		
		local levelteledone = 0
		
		if (#emptythings > 0) then
			for i=1,roomsizex-2 do
				for j=1,roomsizey-2 do
					local tileid = i + j * roomsizex
					
					if (unitmap[tileid] == nil) or (#unitmap[tileid] == 0) then
						local esafe = issafe(2,i,j)
						
						--MF_alert(tostring(i) .. ", " .. tostring(j))
						local keypair = ""
						local winpair = ""
						local hotpair = ""
						local defeatpair = ""
						local bonuspair = ""
						local endpair = ""
						
						local canmelt = false
						local candefeat = false
						local canwin = false
						local canbonus = false
						local canend = false
						
						local unlock = false
						local victory = false
						local melt = false
						local defeat = false
						local bonus = false
						local ending = false
						
						for a,rules in ipairs(emptythings) do
							local rule = rules[1]
							local conds = rules[2]
							
							if (rule[2] == "is") then
								if (rule[3] == "open") and testcond(conds,2,i,j) then
									if (string.len(keypair) == 0) then
										keypair = "shut"
									elseif (keypair == "open") then
										unlock = true
									end
								elseif (rule[3] == "shut") and testcond(conds,2,i,j) then
									if (string.len(keypair) == 0) then
										keypair = "open"
									elseif (keypair == "shut") then
										unlock = true
									end
								end
								
								if (rule[3] == "melt") and testcond(conds,2,i,j) then
									canmelt = true
									
									if (string.len(hotpair) == 0) then
										hotpair = "hot"
									elseif (hotpair == "melt") then
										melt = true
									end
								elseif (rule[3] == "hot") and testcond(conds,2,i,j) then
									if (string.len(hotpair) == 0) then
										hotpair = "melt"
									elseif (hotpair == "hot") then
										melt = true
									end
								end
								
								if (rule[3] == "defeat") and testcond(conds,2,i,j) then
									if (string.len(defeatpair) == 0) then
										defeatpair = "you"
									elseif (defeatpair == "defeat") then
										defeat = true
									end
								elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
									candefeat = true
									canwin = true
									
									if (string.len(defeatpair) == 0) then
										defeatpair = "defeat"
									elseif (defeatpair == "you") then
										defeat = true
									end
								end
								
								if (rule[3] == "win") and testcond(conds,2,i,j) then
									if (string.len(winpair) == 0) then
										winpair = "you"
									elseif (winpair == "win") then
										victory = true
									end
								elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
									candefeat = true
									canwin = true
									
									if (string.len(winpair) == 0) then
										winpair = "win"
									elseif (winpair == "you") then
										victory = true
									end
								end
								
								if (rule[3] == "bonus") and testcond(conds,2,i,j) then
									if (string.len(bonuspair) == 0) then
										bonuspair = "you"
									elseif (bonuspair == "bonus") then
										bonus = true
									end
									
									canbonus = true
								elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
									if (string.len(bonuspair) == 0) then
										bonuspair = "bonus"
									elseif (bonuspair == "you") then
										bonus = true
									end
								end
								
								if (rule[3] == "end") and testcond(conds,2,i,j) then
									if (string.len(endpair) == 0) then
										endpair = "you"
									elseif (bonuspair == "end") then
										ending = true
									end
									
									canend = true
								elseif ((rule[3] == "you") or (rule[3] == "you2") or (rule[3] == "3d")) and testcond(conds,2,i,j) then
									if (string.len(endpair) == 0) then
										endpair = "end"
									elseif (endpair == "you") then
										ending = true
									end
								end
								
								if (rule[3] == "done") and testcond(conds,2,i,j) then
									emptydone = true
								end
								
								if (keypair == "shut") and (hasfeature("level","is","shut",1,i,j) ~= nil) and floating_level(2,i,j) then
									unlock = true
								elseif (keypair == "open") and (hasfeature("level","is","open",1,i,j) ~= nil) and floating_level(2,i,j) then
									unlock = true
								end
								
								if canmelt and (hasfeature("level","is","hot",1,i,j) ~= nil) and floating_level(2,i,j) then
									melt = true
								end
								
								if candefeat and (hasfeature("level","is","defeat",1,i,j) ~= nil) and floating_level(2,i,j) then
									defeat = true
								end
								
								if canwin and (hasfeature("level","is","win",1,i,j) ~= nil) and floating_level(2,i,j) then
									victory = true
								end
								
								if canbonus and ((hasfeature("level","is","you",1,i,j) ~= nil) or (hasfeature("level","is","you2",1,i,j) ~= nil) or (hasfeature("level","is","3d",1,i,j) ~= nil)) and floating_level(2,i,j) then
									bonus = true
								end
								
								if canend and ((hasfeature("level","is","you",1,i,j) ~= nil) or (hasfeature("level","is","you2",1,i,j) ~= nil) or (hasfeature("level","is","3d",1,i,j) ~= nil)) and floating_level(2,i,j) then
									ending = true
								end
							elseif (rule[2] == "eat") and (rule[3] == "level") and (lsafe == false) then
								if testcond(conds,2,i,j) and floating_level(2,i,j) then
									local pmult,sound = checkeffecthistory("eat")
									setsoundname("removal",1,sound)
									destroylevel()
									return
								end
							end
						end
						
						local alive = true
						
						if unlock and (esafe == false) and alive then
							setsoundname("turn",7)
							
							if (math.random(1,4) == 1) then
								MF_particles("unlock",i,j,1,2,4,1,1)
							end
							
							alive = false
							delete(2,i,j)
						end
						
						if melt and (esafe == false) and alive then
							setsoundname("turn",9)
							
							if (math.random(1,4) == 1) then
								MF_particles("smoke",i,j,1,0,1,1,1)
							end
							
							alive = false
							delete(2,i,j)
						end
						
						if defeat and (esafe == false) and alive then
							setsoundname("turn",1)
							
							if (math.random(1,4) == 1) then
								MF_particles("destroy",i,j,1,0,3,1,1)
							end
							
							alive = false
							delete(2,i,j)
						end
						
						if bonus and (esafe == false) and alive then
							setsoundname("turn",2)
							
							if (math.random(1,4) == 1) then
								MF_particles("win",i,j,1,4,2,1,1)
							end
							
							if (emptybonus == false) then
								MF_playsound("bonus")
								MF_bonus(1)
								addundo({"bonus",1})
								emptybonus = true
							end
							
							alive = false
							delete(2,i,j)
						end
						
						if victory and alive then
							MF_win()
							return
						end
						
						if ending and alive and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
							if (editor.values[INEDITOR] ~= 0) then
								MF_end_single()
								MF_win()
								return
							else
								MF_end_single()
								MF_win()
								MF_credits(1)
								return
							end
						end
					end
				end
			end
		end
		
		if emptydone then
			local donenum = math.random(1,4)
			MF_playsound("done" .. tostring(donenum))
		end
		
		if (#things > 0) then
			for i,rules in ipairs(things) do
				local rule = rules[1]
				local conds = rules[2]
				
				--MF_alert(rule[1] .. " " .. rule[2] .. " " .. rule[3] .. ", " .. tostring(testcond(conds,1)))
				
				if testcond(conds,1) then
					if (rule[2] == "eat") then
						local target = rule[3]
						
						local eaten = {}
						
						if (target ~= "all") and (target ~= "empty") then
							local dothese = {}
							
							if (string.sub(target, 1, 5) ~= "group") then
								dothese = {target}
							else
								dothese = findgroup(target)
							end
							
							for c,d in ipairs(dothese) do
								if (unitlists[d] ~= nil) then
									if (d == "level") and (#unitlists["level"] > 0) and (lsafe == false) then
										local pmult,sound = checkeffecthistory("eat")
										setsoundname("removal",1,sound)
										destroylevel()
										return
									end
									
									for a,unitid in ipairs(unitlists[d]) do
										if (issafe(unitid) == false) then
											table.insert(eaten, unitid)
										end
									end
								end
							end
						elseif (target == "empty") then
							local empties = findempty()
							
							for a,b in ipairs(empties) do
								local x = b % roomsizex
								local y = math.floor(b / roomsizex)
								
								generaldata.values[SHAKE] = 4
							
								local pmult,sound = checkeffecthistory("eat")
								MF_particles("eat",x,y,5 * pmult,0,3,1,1)
								setsoundname("removal",1,sound)
								
								delete(2,x,y)
							end
						end
						
						for a,b in ipairs(eaten) do
							local bunit = mmf.newObject(b)
							local x,y = bunit.values[XPOS],bunit.values[YPOS]
							generaldata.values[SHAKE] = 4
							
							local pmult,sound = checkeffecthistory("eat")
							MF_particles("eat",x,y,5 * pmult,0,3,1,1)
							setsoundname("removal",1,sound)
							
							delete(b,x,y)
						end
					elseif (rule[2] == "is") then
						local action = rule[3]
						
						if (action == "you") or (action == "you2") or (action == "3d") then
							local defeats = findfeature(nil,"is","defeat")
							local wins = findfeature(nil,"is","win")
							local ends = findfeature(nil,"is","end")
							
							if (defeats ~= nil) then
								for a,b in ipairs(defeats) do
									if (b[1] ~= "level") then
										local allyous = findall(b)
										
										if (#allyous > 0) then
											for c,d in ipairs(allyous) do
												if (issafe(1) == false) and floating_level(d) then
													destroylevel()
													return
												end
											end
										end
									elseif testcond(b[2],1) and (lsafe == false) then
										destroylevel()
										return
									end
								end
							end
							
							if ((#findallfeature("empty","is","defeat") > 0) or (#findallfeature("empty","is","defeat") > 0)) and floating_level(2) and (lsafe == false) then
								destroylevel()
								return
							end
							
							local canwin = false
							local canend = false
							
							if (wins ~= nil) then
								for a,b in ipairs(wins) do
									if (b[1] ~= "level") then
										local allyous = findall(b)
										
										if (#allyous > 0) then
											for c,d in ipairs(allyous) do
												if floating_level(d) then
													canwin = true
												end
											end
										end
									elseif testcond(b[2],1) then
										canwin = true
									end
								end
							end
							
							if (ends ~= nil) then
								for a,b in ipairs(ends) do
									if (b[1] ~= "level") then
										local allyous = findall(b)
										
										if (#allyous > 0) then
											for c,d in ipairs(allyous) do
												if floating_level(d) then
													canend = true
												end
											end
										end
									elseif testcond(b[2],1) then
										canend = true
									end
								end
							end
							
							if ((#findallfeature("empty","is","win") > 0) or (#findallfeature("empty","is","win") > 0)) and floating_level(2) then
								canwin = true
							end
							
							if ((#findallfeature("empty","is","end") > 0) or (#findallfeature("empty","is","end") > 0)) and floating_level(2) then
								canend = true
							end
							
							if canwin then
								MF_win()
								return
							end
							
							if canend and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
								if (editor.values[INEDITOR] ~= 0) then
									MF_end_single()
									MF_win()
									return
								else
									MF_end_single()
									MF_win()
									MF_credits(1)
									return
								end
							end
						elseif (action == "defeat") then
							local yous = findfeature(nil,"is","you")
							local yous2 = findfeature(nil,"is","you2")
							local yous3 = findfeature(nil,"is","3d")
							
							if (yous == nil) then
								yous = {}
							end
							
							if (yous2 ~= nil) then
								for i,v in ipairs(yous2) do
									table.insert(yous, v)
								end
							end
							
							if (yous3 ~= nil) then
								for i,v in ipairs(yous3) do
									table.insert(yous, v)
								end
							end
							
							if (yous ~= nil) then
								for a,b in ipairs(yous) do
									if (b[1] ~= "level") then
										local allyous = findall(b)
										
										if (#allyous > 0) then
											for c,d in ipairs(allyous) do
												if (issafe(d) == false) and floating_level(d) then
													local unit = mmf.newObject(d)
													
													local pmult,sound = checkeffecthistory("defeat")
													MF_particles("destroy",unit.values[XPOS],unit.values[YPOS],5 * pmult,0,3,1,1)
													setsoundname("removal",1,sound)
													generaldata.values[SHAKE] = 2
													delete(d)
												end
											end
										end
									elseif testcond(b[2],1) and (lsafe == false) then
										destroylevel()
										return
									end
								end
							end
						elseif (action == "weak") then
							for i,unit in ipairs(units) do
								local name = unit.strings[UNITNAME]
								if (unit.strings[UNITTYPE] == "text") then
									name = "text"
								end
								
								if floating_level(unit.fixed) and (lsafe == false) then
									destroylevel()
								end
							end
						elseif (action == "hot") then
							local melts = findfeature(nil,"is","melt")
							
							if (melts ~= nil) then
								for a,b in ipairs(melts) do
									local allmelts = findall(b)
									
									if (#allmelts > 0) then
										for c,d in ipairs(allmelts) do
											if (issafe(d) == false) and floating_level(d) then
												local unit = mmf.newObject(d)
												
												local pmult,sound = checkeffecthistory("hot")
												MF_particles("smoke",unit.values[XPOS],unit.values[YPOS],5 * pmult,0,1,1,1)
												generaldata.values[SHAKE] = 2
												setsoundname("removal",9,sound)
												delete(d)
											end
										end
									end
								end
							end
						elseif (action == "melt") then
							local hots = findfeature(nil,"is","hot")
							
							if (hots ~= nil) and (lsafe == false) then
								for a,b in ipairs(hots) do
									local doit = false
									
									if (b[1] ~= "level") then
										local allhots = findall(b)
										
										for c,d in ipairs(allhots) do
											if floating_level(d) then
												doit = true
											end
										end
									elseif testcond(b[2],1) then
										doit = true
									end
									
									if doit then
										destroylevel()
									end
								end
							end
							
							if (#findallfeature("empty","is","hot") > 0) and floating_level(2) and (lsafe == false) then
								destroylevel()
								return
							end
						elseif (action == "open") then
							local shuts = findfeature(nil,"is","shut")
							
							local openthese = {}
							
							if (shuts ~= nil) then
								for a,b in ipairs(shuts) do
									local doit = false
									
									if (b[1] ~= "level") then
										local allshuts = findall(b)
										
										for c,d in ipairs(allshuts) do
											if floating_level(d) then
												doit = true
												
												if (issafe(d) == false) then
													table.insert(openthese, d)
												end
											end
										end
									elseif testcond(b[2],1) then
										doit = true
									end
									
									if doit then
										if (lsafe == false) then
											destroylevel()
											return
										end
									end
								end
							end
							
							if (#openthese > 0) then
								generaldata.values[SHAKE] = 8
								
								for a,b in ipairs(openthese) do
									local bunit = mmf.newObject(b)
									local bx,by = bunit.values[XPOS],bunit.values[YPOS]
									
									local pmult,sound = checkeffecthistory("unlock")
									setsoundname("turn",7,sound)
									MF_particles("unlock",bx,by,15 * pmult,2,4,1,1)
									
									delete(b)
									deleted[b] = 1
								end
							end
							
							if (#findallfeature("empty","is","shut") > 0) and floating_level(2) and (lsafe == false) then
								destroylevel()
								return
							end
						elseif (action == "shut") then
							local opens = findfeature(nil,"is","open")
							
							local openthese = {}
							
							if (opens ~= nil) then
								for a,b in ipairs(opens) do
									local doit = false
									
									if (b[1] ~= "level") then
										local allopens = findall(b)
										
										for c,d in ipairs(allopens) do
											if floating_level(d) then
												doit = true
												
												if (issafe(d) == false) then
													table.insert(openthese, d)
												end
											end
										end
									elseif testcond(b[2],1) then
										doit = true
									end
									
									if doit then
										if (lsafe == false) then
											destroylevel()
											return
										end
									end
								end
							end
							
							if (#openthese > 0) then
								generaldata.values[SHAKE] = 8
								
								for a,b in ipairs(openthese) do
									local bunit = mmf.newObject(b)
									local bx,by = bunit.values[XPOS],bunit.values[YPOS]
									
									local pmult,sound = checkeffecthistory("unlock")
									setsoundname("turn",7,sound)
									MF_particles("unlock",bx,by,15 * pmult,2,4,1,1)
									
									delete(b)
									deleted[b] = 1
								end
							end
							
							if (#findallfeature("empty","is","open") > 0) and floating_level(2) and (lsafe == false) then
								destroylevel()
								return
							end
						elseif (action == "sink") then
							local openthese = {}
							
							for a,unit in ipairs(units) do
								local name = unit.strings[UNITNAME]
								
								if (unit.strings[UNITTYPE] == "text") then
									name = "text"
								end
								
								if floating_level(unit.fixed) then
									if (lsafe == false) then
										destroylevel()
										return
									end
									
									if (issafe(unit.fixed) == false) then
										table.insert(openthese, unit.fixed)
									end
								end
							end
							
							if (#openthese > 0) then
								generaldata.values[SHAKE] = 3
								
								for a,b in ipairs(openthese) do
									local bunit = mmf.newObject(b)
									local bx,by = bunit.values[XPOS],bunit.values[YPOS]
									
									local pmult,sound = checkeffecthistory("sink")
									setsoundname("removal",3,sound)
									local c1,c2 = getcolour(b)
									MF_particles("destroy",bx,by,15 * pmult,c1,c2,1,1)
									
									delete(b)
									deleted[b] = 1
								end
							end
						elseif (action == "done") then
							local doned = {}
							for a,unit in ipairs(units) do
								table.insert(doned, unit)
							end
							
							for a,unit in ipairs(doned) do
								addundo({"done",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.fixed,unit.values[FLOAT]})
								
								unit.values[FLOAT] = 2
								unit.values[EFFECTCOUNT] = math.random(-10,10)
								unit.values[POSITIONING] = 7
								unit.flags[DEAD] = true
								
								delunit(unit.fixed)
							end
							
							MF_playsound("doneall_c")
						elseif (action == "bonus") then
							local yous = findfeature(nil,"is","you")
							local yous2 = findfeature(nil,"is","you2")
							local yous3 = findfeature(nil,"is","3d")
							
							if (yous == nil) then
								yous = {}
							end
							
							if (yous2 ~= nil) then
								for i,v in ipairs(yous2) do
									table.insert(yous, v)
								end
							end
							
							if (yous3 ~= nil) then
								for i,v in ipairs(yous3) do
									table.insert(yous, v)
								end
							end
							
							if (yous ~= nil) then
								for a,b in ipairs(yous) do
									if (b[1] ~= "level") then
										local allyous = findall(b)
										
										if (#allyous > 0) then
											for c,d in ipairs(allyous) do
												if (issafe(d) == false) and floating_level(d) then
													MF_bonus(1)
													addundo({"bonus",1})
													destroylevel("bonus")
													return
												end
											end
										end
									elseif testcond(b[2],1) then
										if (lsafe == false) then
											MF_bonus(1)
											addundo({"bonus",1})
											destroylevel("bonus")
											return
										end
									end
								end
							end
							
							if ((#findallfeature("empty","is","you") > 0) or (#findallfeature("empty","is","you2") > 0) or (#findallfeature("empty","is","3d") > 0)) and floating_level(2) and (lsafe == false) then
								destroylevel("bonus")
								return
							end
						elseif (action == "win") then
							local yous = findfeature(nil,"is","you")
							local yous2 = findfeature(nil,"is","you2")
							local yous3 = findfeature(nil,"is","3d")
							
							if (yous == nil) then
								yous = {}
							end
							
							if (yous2 ~= nil) then
								for i,v in ipairs(yous2) do
									table.insert(yous, v)
								end
							end
							
							if (yous3 ~= nil) then
								for i,v in ipairs(yous3) do
									table.insert(yous, v)
								end
							end
							
							local canwin = false
							
							if (yous ~= nil) then
								for a,b in ipairs(yous) do
									local allyous = findall(b)
									local doit = false
									
									for c,d in ipairs(allyous) do
										if floating_level(d) then
											doit = true
										end
									end
									
									if doit then
										canwin = true
										for c,d in ipairs(allyous) do
											local unit = mmf.newObject(d)
											local pmult,sound = checkeffecthistory("win")
											MF_particles("win",unit.values[XPOS],unit.values[YPOS],10 * pmult,2,4,1,1)
										end
									end
								end
							end
							
							local emptyyou = false
							if ((#findallfeature("empty","is","you") > 0) or (#findallfeature("empty","is","you2") > 0) or (#findallfeature("empty","is","3d") > 0)) and floating_level(2) then
								emptyyou = true
							end
							
							if (hasfeature("level","is","you",1) ~= nil) or (hasfeature("level","is","you2",1) ~= nil) or (hasfeature("level","is","3d",1) ~= nil) or emptyyou then
								canwin = true
							end
							
							if canwin then
								MF_win()
								return
							end
						elseif (action == "end") then
							local yous = findfeature(nil,"is","you")
							local yous2 = findfeature(nil,"is","you2")
							local yous3 = findfeature(nil,"is","3d")
							
							if (yous == nil) then
								yous = {}
							end
							
							if (yous2 ~= nil) then
								for i,v in ipairs(yous2) do
									table.insert(yous, v)
								end
							end
							
							if (yous3 ~= nil) then
								for i,v in ipairs(yous3) do
									table.insert(yous, v)
								end
							end
							
							local canend = false
							
							if (yous ~= nil) then
								for a,b in ipairs(yous) do
									local allyous = findall(b)
									local doit = false
									
									for c,d in ipairs(allyous) do
										if floating_level(d) then
											doit = true
										end
									end
									
									if doit then
										canend = true
										for c,d in ipairs(allyous) do
											local unit = mmf.newObject(d)
											local pmult,sound = checkeffecthistory("win")
											MF_particles("win",unit.values[XPOS],unit.values[YPOS],10 * pmult,2,4,1,1)
										end
									end
								end
							end
							
							local emptyyou = false
							if ((#findallfeature("empty","is","you") > 0) or (#findallfeature("empty","is","you2") > 0) or (#findallfeature("empty","is","3d") > 0)) and floating_level(2) then
								emptyyou = true
							end
							
							if (hasfeature("level","is","you",1) ~= nil) or (hasfeature("level","is","you2",1) ~= nil) or (hasfeature("level","is","3d",1) ~= nil) or emptyyou then
								canend = true
							end
							
							if canend and (generaldata.strings[WORLD] ~= generaldata.strings[BASEWORLD]) then
								if (editor.values[INEDITOR] ~= 0) then
									MF_end_single()
									MF_win()
									break
								else
									MF_end_single()
									MF_win()
									MF_credits(1)
									break
								end
							end
						elseif (action == "tele") and (levelteledone < 3) and (lstill == false) then
							levelteledone = levelteledone + 1
							
							for a,unit in ipairs(units) do
								local x,y = unit.values[XPOS],unit.values[YPOS]
								
								local tx,ty = fixedrandom(1,roomsizex-2),fixedrandom(1,roomsizey-2)
								
								if floating_level(unit.fixed) then
									update(unit.fixed,tx,ty)
									
									local pmult,sound = checkeffecthistory("tele")
									MF_particles("glow",x,y,5 * pmult,1,4,1,1)
									MF_particles("glow",tx,ty,5 * pmult,1,4,1,1)
									setsoundname("turn",6,sound)
								end
							end
						elseif (action == "move") then
							local dir = mapdir
							
							local drs = ndirs[dir + 1]
							local ox,oy = drs[1],drs[2]
							
							if (lstill == false) and (lsleep == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,dir,dir})
								MF_scrollroom(ox * tilesize,oy * tilesize)
								updateundo = true
							end
						elseif (action == "nudgeright") then
							local dir = 0
							
							local drs = ndirs[dir + 1]
							local ox,oy = drs[1],drs[2]
							
							if (lstill == false) and (lsleep == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
								MF_scrollroom(ox * tilesize,oy * tilesize)
								updateundo = true
							end
						elseif (action == "nudgeup") then
							local dir = 1
							
							local drs = ndirs[dir + 1]
							local ox,oy = drs[1],drs[2]
							
							if (lstill == false) and (lsleep == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
								MF_scrollroom(ox * tilesize,oy * tilesize)
								updateundo = true
							end
						elseif (action == "nudgeleft") then
							local dir = 2
							
							local drs = ndirs[dir + 1]
							local ox,oy = drs[1],drs[2]
							
							if (lstill == false) and (lsleep == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
								MF_scrollroom(ox * tilesize,oy * tilesize)
								updateundo = true
							end
						elseif (action == "nudgedown") then
							local dir = 3
							
							local drs = ndirs[dir + 1]
							local ox,oy = drs[1],drs[2]
							
							if (lstill == false) and (lsleep == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + ox * tilesize,Yoffset + oy * tilesize,mapdir,mapdir})
								MF_scrollroom(ox * tilesize,oy * tilesize)
								updateundo = true
							end
						elseif (action == "fall") then
							local drop = 20
							local dir = mapdir
							
							local ox = 0
							local oy = 1
							
							if (lstill == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
								MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
								updateundo = true
							end
						elseif (action == "fallright") then
							local drop = 35
							local dir = mapdir
							
							local ox = 1
							local oy = 0
							
							if (lstill == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
								MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
								updateundo = true
							end
						elseif (action == "fallup") then
							local drop = 20
							local dir = mapdir
							
							local ox = 0
							local oy = -1
							
							if (lstill == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
								MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
								updateundo = true
							end
						elseif (action == "fallleft") then
							local drop = 35
							local dir = mapdir
							
							local ox = -1
							local oy = 0
							
							if (lstill == false) then
								addundo({"levelupdate",Xoffset,Yoffset,Xoffset + tilesize * drop * ox,Yoffset + tilesize * drop * oy,dir,dir})
								MF_scrollroom(tilesize * drop * ox,tilesize * drop * oy)
								updateundo = true
							end
						elseif (rule[3] == "turn") then
							local newmapdir = (mapdir - 1 + 4) % 4
							local newmaprotation = ((mapdir + 1 + 4) % 4) * 90
							
							addundo({"maprotation",maprotation,newmaprotation,newmapdir})
							addundo({"mapdir",mapdir,newmapdir})
							maprotation = newmaprotation
							mapdir = newmapdir
							MF_levelrotation(maprotation)
						elseif (rule[3] == "deturn") then
							local newmapdir = (mapdir + 1 + 4) % 4
							local newmaprotation = ((mapdir + 1 + 4) % 4) * 90
							
							addundo({"maprotation",maprotation,newmaprotation,newmapdir})
							addundo({"mapdir",mapdir,newmapdir})
							maprotation = newmaprotation
							mapdir = newmapdir
							MF_levelrotation(maprotation)
						elseif (action == "empty") then
							destroylevel("empty")
						end
					end
				end
			end
		end
		
		if (featureindex["done"] ~= nil) then
			for i,v in ipairs(featureindex["done"]) do
				table.insert(donethings, v)
			end
		end
		
		if (#donethings > 0) and (generaldata.values[WINTIMER] == 0) then
			for i,rules in ipairs(donethings) do
				local rule = rules[1]
				local conds = rules[2]
				
				if (rule[1] == "all") and (rule[2] == "is") and (rule[3] == "done") then
					local targets = findallfeature(nil,"is","done",true)
					local found = false
					
					for i,v in ipairs(targets) do
						local unit = mmf.newObject(v)
						
						if (unit.className ~= "level") then
							found = true
							break
						end
					end
					
					if found then
						if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) and (editor.values[INEDITOR] == 0) then
							MF_playsound("doneall_c")
							MF_allisdone()
						elseif (editor.values[INEDITOR] ~= 0) then
							local pmult = checkeffecthistory("win")
							
							MF_playsound("doneall_c")
							MF_done_single()
							MF_win()
							break
						else
							local pmult = checkeffecthistory("win")
							
							local mods_run = do_mod_hook("levelpack_done", {})
							
							if (mods_run == false) then
								MF_playsound("doneall_c")
								MF_done_single()
								MF_win()
								MF_credits(2)
							end
							break
						end
					end
				end
			end
		end
		
		if (generaldata.strings[WORLD] == generaldata.strings[BASEWORLD]) and (generaldata.strings[CURRLEVEL] == "305level") then
			local numfound = false
			
			if (featureindex["image"] ~= nil) then
				for i,v in ipairs(featureindex["image"]) do
					local rule = v[1]
					local conds = v[2]
					
					if (rule[1] == "image") and (rule[2] == "is") and (#conds == 0) then
						local num = rule[3]
						
						local nums = {
							one = {1, "image_desc_1"},
							two = {2, "image_desc_2"},
							three = {3, "image_desc_3"},
							four = {4, "image_desc_4"},
							five = {5, "image_desc_5"},
							six = {6, "image_desc_6"},
							seven = {7, "image_desc_7"},
							eight = {8, "image_desc_8"},
							nine = {9, "image_desc_9"},
							ten = {10, "image_desc_10"},
							fourteen = {11, "image_desc_11"},
							sixteen = {12, "image_desc_12"},
							minusone = {13, "image_desc_13"},
							minustwo = {14, "image_desc_14"},
							minusthree = {15, "image_desc_15"},
							minusten = {16, "image_desc_16"},
							win = {0, "win"}
						}
						
						if (nums[num] ~= nil) then
							local data = nums[num]
							
							if (data[2] ~= "win") then
								MF_setart(data[1], langtext(data[2],true))
								numfound = true
							else
								local yous = findallfeature(nil,"is","you",true)
								local yous2 = findallfeature(nil,"is","you2",true)
								local yous3 = findallfeature(nil,"is","3d",true)
								
								if (#yous2 > 0) then
									for a,b in ipairs(yous2) do
										table.insert(yous, b)
									end
								end
								
								if (#yous3 > 0) then
									for a,b in ipairs(yous3) do
										table.insert(yous, b)
									end
								end
								
								for a,b in ipairs(yous) do
									local unit = mmf.newObject(b)
									local x,y = unit.values[XPOS],unit.values[YPOS]
									
									if (x > roomsizex - 16) then
										local pmult = checkeffecthistory("win")
										
										MF_particles("win",x,y,10 * pmult,2,4,1,1)
										MF_win()
										break
									end
								end
							end
						end
					end
				end
			end
				
			if (numfound == false) then
				MF_setart(0,"")
			end
		end
		
		if unlocked then
			setsoundname("turn",7)
		end
	end
end