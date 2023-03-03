function code(alreadyrun_)
	local playrulesound = false
	local alreadyrun = alreadyrun_ or false
	
	if (updatecode == 1) then
		HACK_INFINITY = HACK_INFINITY + 1
		--MF_alert("code being updated!")
		
		MF_removeblockeffect(0)
		wordrelatedunits = {}
		
		do_mod_hook("rule_update",{alreadyrun})
		
		if (HACK_INFINITY < 200) then
			local checkthese = {}
			local wordidentifier = ""
			wordunits,wordidentifier,wordrelatedunits = findwordunits()
			
			if (#wordunits > 0) then
				for i,v in ipairs(wordunits) do
					if testcond(v[2],v[1]) then
						table.insert(checkthese, v[1])
					end
				end
			end
			
			features = {}
			featureindex = {}
			condfeatureindex = {}
			visualfeatures = {}
			notfeatures = {}
			groupfeatures = {}
			local firstwords = {}
			local alreadyused = {}
			
			--add persistent rules to base rules
			for level,levelrules in pairs(persistbaserules) do
				if level ~= generaldata.strings[CURRLEVEL] then
					for j,w in ipairs(levelrules) do
						addbaserule(w[1],w[2],w[3])
					end
				end
			end
			
			formlettermap()
			
			if (#codeunits > 0) then
				for i,v in ipairs(codeunits) do
					table.insert(checkthese, v)
				end
			end
		
			if (#checkthese > 0) or (#letterunits > 0) then
				for iid,unitid in ipairs(checkthese) do
					local unit = mmf.newObject(unitid)
					local x,y = unit.values[XPOS],unit.values[YPOS]
					local ox,oy,nox,noy = 0,0
					local tileid = x + y * roomsizex

					setcolour(unit.fixed)
					
					if (alreadyused[tileid] == nil) and (unit.values[TYPE] ~= 5) then
						for i=1,2 do
							local drs = dirs[i+2]
							local ndrs = dirs[i]
							ox = drs[1]
							oy = drs[2]
							nox = ndrs[1]
							noy = ndrs[2]
							
							--MF_alert("Doing firstwords check for " .. unit.strings[UNITNAME] .. ", dir " .. tostring(i))
							
							local hm = codecheck(unitid,ox,oy,i)
							local hm2 = codecheck(unitid,nox,noy,i)
							
							if (#hm == 0) and (#hm2 > 0) then
								--MF_alert("Added " .. unit.strings[UNITNAME] .. " to firstwords, dir " .. tostring(i))
								
								table.insert(firstwords, {{unitid}, i, 1, unit.strings[UNITNAME], unit.values[TYPE]})
								
								if (alreadyused[tileid] == nil) then
									alreadyused[tileid] = {}
								end
								
								alreadyused[tileid][i] = 1
							end
						end
					end
				end
				
				--table.insert(checkthese, {unit.strings[UNITNAME], unit.values[TYPE], unit.values[XPOS], unit.values[YPOS], 0, 1, {unitid})
				
				for a,b in pairs(letterunits_map) do
					for iid,data in ipairs(b) do
						local x,y,i = data[3],data[4],data[5]
						local unitids = data[7]
						local width = data[6]
						local word,wtype = data[1],data[2]
						
						local unitid = unitids[1]
						
						local tileid = x + y * roomsizex
						
						if (alreadyused[tileid] == nil) or ((alreadyused[tileid] ~= nil) and (alreadyused[tileid][i] == nil)) then
							local drs = dirs[i+2]
							local ndrs = dirs[i]
							ox = drs[1]
							oy = drs[2]
							nox = ndrs[1] * width
							noy = ndrs[2] * width
							
							local hm = codecheck(unitid,ox,oy,i)
							local hm2 = codecheck(unitid,nox,noy,i)
							
							--MF_alert(word .. ", " .. tostring(hm) .. ", " .. tostring(hm2) .. ", " .. tostring(width))
							
							if (#hm == 0) and (#hm2 > 0) then
								table.insert(firstwords, {unitids, i, width, word, wtype})
								
								if (alreadyused[tileid] == nil) then
									alreadyused[tileid] = {}
								end
								
								alreadyused[tileid][i] = 1
							end
						end
					end
				end
				
				docode(firstwords,wordunits)
				subrules()
				grouprules()
				playrulesound = postrules(alreadyrun)
				updatecode = 0
				
				local newwordunits,newwordidentifier,wordrelatedunits = findwordunits()
				
				--MF_alert("ID comparison: " .. newwordidentifier .. " - " .. wordidentifier)
				
				if (newwordidentifier ~= wordidentifier) then
					updatecode = 1
					code(true)
				else
					--domaprotation()
				end
			end
		else
			MF_alert("Level destroyed - code() run too many times")
			destroylevel("infinity")
			return
		end
		
		if (alreadyrun == false) then
			effects_decors()
		end
	end
	
	if (alreadyrun == false) then
		local rulesoundshort = ""
		alreadyrun = true
		if playrulesound and (generaldata5.values[LEVEL_DISABLERULEEFFECT] == 0) then
			local pmult,sound = checkeffecthistory("rule")
			rulesoundshort = sound
			local rulename = "rule" .. tostring(math.random(1,5)) .. rulesoundshort
			MF_playsound(rulename)
		end
	end
end