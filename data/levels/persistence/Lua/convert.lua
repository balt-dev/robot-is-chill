function dolevelconversions()
	if (#features > 0) and (generaldata.values[WINTIMER] == 0) then
		local mats = levelconversions
		local mat1 = "level"
		local levelmats = {}
		
		local revert = false
		
		for i,matdata in pairs(mats) do
			local conds = matdata[2]
			local mat2 = matdata[1]
			
			local objectfound = false
			
			if (unitreference[mat2] ~= nil) then
				local object = unitreference[mat2]
				
				if (tileslist[object]["name"] == mat2) and ((changes[object] == nil) or (changes[object]["name"] == nil)) then
					objectfound = true
				elseif (changes[object] ~= nil) then
					if (changes[object]["name"] ~= nil) and (changes[object]["name"] == mat2) then
						objectfound = true
					end
				end
			elseif (mat2 == "error") then
				destroylevel()
			elseif (mat2 == "revert") then
				objectfound = true
			end
			
			if testcond(conds,1) and objectfound then
				if (mat2 ~= "revert") then
					table.insert(levelmats, matdata[1])
				else
					revert = true
					levelmats = {"revert"}
					break
				end
			end
		end
		
		if (#levelmats > 0) and (#levelmats < 50) then
			if (editor.values[INEDITOR] == 0) then
				if (revert == false) then
					level_to_convert = {generaldata.strings[CURRLEVEL], levelmats}
					
					local savestring = ""
					for a,b in pairs(levelmats) do
						savestring = savestring .. b .. ","
					end
					
					local upperlevel = leveltree[#leveltree - 1] or generaldata.strings[CURRLEVEL]
					local convertdata = MF_read("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert","converts")
					local levelconverts = tonumber(convertdata) or 0
					local idtostore = levelconverts
					
					if (levelconverts == 0) then
						local totalconverts = tonumber(MF_read("save",generaldata.strings[WORLD] .. "_converts","total")) or 0
						MF_store("save",generaldata.strings[WORLD] .. "_converts",tostring(totalconverts),generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert")
						totalconverts = totalconverts + 1
						MF_store("save",generaldata.strings[WORLD] .. "_converts","total",tostring(totalconverts))
					end
					
					if (levelconverts > 0) then
						for a=1,levelconverts do
							local result = string.find("___" .. MF_read("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert",tostring(a-1)), "___" .. generaldata.strings[CURRLEVEL])
							
							if (result ~= nil) then
								idtostore = a - 1
							end
						end
					end
					
					MF_store("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert",tostring(idtostore),generaldata.strings[CURRLEVEL] .. "," .. savestring)
					
					if (idtostore == levelconverts) then
						levelconverts = levelconverts + 1
						
						MF_store("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert","converts",tostring(levelconverts))
					end
				else
					level_to_convert = {generaldata.strings[CURRLEVEL], levelmats}
					
					local upperlevel = leveltree[#leveltree - 1] or generaldata.strings[CURRLEVEL]
					local convertdata = MF_read("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert","converts")
					local levelconverts = tonumber(convertdata) or 0
					
					local found = -1
					
					if (levelconverts > 0) then
						for a=1,levelconverts do
							local result = string.find("___" .. MF_read("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert",tostring(a-1)), "___" .. generaldata.strings[CURRLEVEL])
							
							if (result ~= nil) then
								found = a
							end
							
							if (found > 0) and (a > found) then
								local newa = a - 1
								local datatostore = MF_read("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert",tostring(a-1))
								MF_store("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert",tostring(newa-1),datatostore)
							end
						end
					end
					
					if (found > 0) then
						levelconverts = levelconverts - 1
						
						if (levelconverts > 0) then
							MF_store("save",generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert","converts",tostring(levelconverts))
						else
							MF_deletesave_group(generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert","converts")
							
							local totalconverts = tonumber(MF_read("save",generaldata.strings[WORLD] .. "_converts","total"))
							local found2 = -1
							
							if (totalconverts ~= nil) then
								for a=1,totalconverts do
									local result = string.find("___" .. MF_read("save",generaldata.strings[WORLD] .. "_converts",tostring(a-1)), "___" .. generaldata.strings[WORLD] .. "_" .. upperlevel .. "_convert")
							
									if (result ~= nil) then
										found2 = a
									end
									
									if (found2 > 0) and (a > found2) then
										local newa = a - 1
										local datatostore = MF_read("save",generaldata.strings[WORLD] .. "_converts",tostring(a-1))
										MF_store("save",generaldata.strings[WORLD] .. "_converts",tostring(newa-1),datatostore)
									end
								end
							end
							
							if (found2 > 0) then
								totalconverts = totalconverts - 1
								
								if (totalconverts > 0) then
									MF_store("save",generaldata.strings[WORLD] .. "_converts","total",tostring(totalconverts))
								else
									MF_deletesave_group(generaldata.strings[WORLD] .. "_converts")
								end
							end
						end
					end
				end
				findpersists()
				uplevel()
			else
				level_to_convert = {}
			end
			MF_levelconversion()
		elseif (#levelmats >= 50) then
			HACK_INFINITY = 200
			destroylevel("toocomplex")
		end
		
		levelconversions = {}
	end
end


function doconvert(data,extrarule_)
	local style = data[2]
	local mats2 = data[3]
	
	local unitid = data[1]
	local unit = {}
	local x,y,dir,name,id,completed,float,ogname = 0,0,0,"",0,0,0,""
	local delthis = false
	
	if (unitid ~= 2) then
		unit = mmf.newObject(unitid)
		x,y,dir,name,id,completed,ogname = unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.strings[UNITNAME],unit.values[ID],unit.values[COMPLETED],unit.originalname
	end
	
	local cdata = {}
	cdata[1] = name
	
	if (style == "convert") then
		for a,mats2data in ipairs(mats2) do
			local mat2 = mats2data[1]
			local ingameid = mats2data[2]
			local baseingameid = mats2data[3]
			
			local unitname = ""
			
			if (mat2 == "revert") and (unitid ~= 2) and (ogname ~= nil) then
				local originalname = ogname
				
				if persistrevert ~= nil then
					originalname = persistrevert
				end
				
				if (string.len(originalname) > 0) then
					unitname = unitreference[originalname]
					mat2 = originalname
				else
					unitname = nil
				end
				
				if (source == "emptyconvert") then
					unitname = ""
					mat2 = "empty"
				end
				
				if (unitname == unit.className) then
					MF_alert("Trying to revert object to the same thing: " .. tostring(originalname))
					return
				end
			elseif (mat2 == "revert") and (unitid == 2) then
				MF_alert("Trying to revert empty")
				return
			end
			
			if (mat2 ~= "empty") and (mat2 ~= "error") and (mat2 ~= "revert") and (mat2 ~= "createall") then
				--MF_alert(tostring(id) .. " unit, " .. tostring(data[1]) .. ", name: " .. name .. ", result: " .. tostring(ingameid))
				
				if (mats2data[1] ~= "revert") then
					unitname = unitreference[mat2]
				end
				
				if (mat2 == "level") then
					unitname = "level"
				end
				
				if (unitname == nil) then
					MF_alert("no className found for " .. mat2 .. "!")
					return
				end
				
				local newunitid = MF_emptycreate(unitname,x,y)
				local newunit = mmf.newObject(newunitid)
				
				newunit.values[ONLINE] = 1
				newunit.values[XPOS] = x
				newunit.values[YPOS] = y
				newunit.values[DIR] = dir
				newunit.values[POSITIONING] = 20
				
				newunit.values[VISUALLEVEL] = unit.values[VISUALLEVEL]
				newunit.values[VISUALSTYLE] = unit.values[VISUALSTYLE]
				newunit.values[COMPLETED] = completed
				
				newunit.strings[COLOUR] = unit.strings[COLOUR]
				newunit.strings[CLEARCOLOUR] = unit.strings[CLEARCOLOUR]
				
				if (unitname == "level") then
					newunit.values[COMPLETED] = math.max(completed, 1)
					newunit.flags[LEVEL_JUSTCONVERTED] = true
					
					if (string.len(unit.strings[LEVELFILE]) > 0) then
						newunit.values[COMPLETED] = math.max(completed, 2)
					end
					
					if (string.len(unit.strings[COLOUR]) == 0) or (string.len(unit.strings[CLEARCOLOUR]) == 0) then
						newunit.strings[COLOUR] = "1,2"
						newunit.strings[CLEARCOLOUR] = "1,3"
						MF_setcolour(newunitid,1,2)
					else
						local c = MF_parsestring(unit.strings[COLOUR])
						MF_setcolour(newunitid,c[1],c[2])
					end
					
					newunit.visible = true
				end
				
				newunit.values[ID] = ingameid
				
				newunit.strings[U_LEVELFILE] = unit.strings[U_LEVELFILE]
				newunit.strings[U_LEVELNAME] = unit.strings[U_LEVELNAME]
				newunit.flags[MAPLEVEL] = unit.flags[MAPLEVEL]
				
				newunit.values[EFFECT] = 1
				newunit.flags[9] = true
				newunit.flags[CONVERTED] = true
				
				cdata[2] = mat2
				
				addundo({"convert",cdata[1],cdata[2],ingameid,baseingameid,x,y,dir})
				addundo({"create",mat2,ingameid,baseingameid,"convert",x,y,dir})
				
				addunit(newunitid)
				addunitmap(newunitid,x,y,newunit.strings[UNITNAME])
				poscorrect(newunitid,generaldata2.values[ROOMROTATION],generaldata2.values[ZOOM],0)
				
				if (spritedata.values[VISION] == 0) or ((newunit.values[TILING] == 1) and (newunit.values[ZLAYER] <= 10) and (newunit.values[ZLAYER] >= 0)) then
					dynamic(newunitid)
				end
				
				newunit.new = false
				newunit.originalname = unit.originalname
				
				if (newunit.strings[UNITTYPE] == "text") then
					updatecode = 1
				else
					local newname = newunit.strings[UNITNAME]
					local notnewname = "not " .. newunit.strings[UNITNAME]
					
					if (featureindex["word"] ~= nil) then
						for i,v in ipairs(featureindex["word"]) do
							local rule = v[1]
							local conds = v[2]
							
							if (rule[2] == "is") and (rule[3] == "word") then
								if (rule[1] == newname) then
									updatecode = 1
									break
								elseif (unitid ~= 2) then
									if (rule[1] == unitname) then
										updatecode = 1
										break
									end
								end
								
								if (#conds > 0) then
									for a,b in ipairs(conds) do
										if (b[2] ~= nil) and (#b[2] > 0) then
											for c,d in ipairs(b[2]) do
												if (d == newname) or ((string.sub(d, 1, 4) == "not ") and (string.sub(d, 5) ~= newname)) then
													updatecode = 1
													break
												elseif (unitid ~= 2) then
													if (d == unitname) or ((string.sub(d, 1, 4) == "not ") and (string.sub(d, 5) ~= unitname)) then
														updatecode = 1
														break
													end
												end
											end
										end
									end
								end
							end
						end
					end
				end
				
				delthis = true
			elseif (mat2 == "error") then
				if (unitid ~= 2) then
					local unit = mmf.newObject(unitid)
					local x,y = unit.values[XPOS],unit.values[YPOS]
					local pmult,sound = checkeffecthistory("paradox")
					local c1,c2 = getcolour(unitid)
					MF_particles("unlock",x,y,20 * pmult,c1,c2,1,1)
					--paradox[id] = 1
				end
				
				delthis = true
			elseif (mat2 == "empty") then
				if testcond(conds,unit.fixed) then
					addundo({"convert",cdata[1],"empty",ingameid,baseingameid,x,y,dir})
					updateunitmap(unitid,x,y,x,y,unit.strings[UNITNAME])
					delthis = true
					
					local tileid = x + y * roomsizex
					if (emptydata[tileid] == nil) then
						emptydata[tileid] = {}
					end
					
					emptydata[tileid]["conv"] = true
				end
			elseif (mat2 == "createall") then
				delthis = createall_single(unitid,conds)
			end
		end
		
		if delthis and (unit.flags[DEAD] == false) then
			addundo({"remove",unit.strings[UNITNAME],unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID],unit.values[ID],unit.strings[U_LEVELFILE],unit.strings[U_LEVELNAME],unit.values[VISUALLEVEL],unit.values[COMPLETED],unit.values[VISUALSTYLE],unit.flags[MAPLEVEL],unit.strings[COLOUR],unit.strings[CLEARCOLOUR],unit.followed,unit.back_init,unit.originalname,unit.strings[UNITSIGNTEXT]})
			
			if (unit.strings[UNITTYPE] == "text") then
				updatecode = 1
			end
			
			delunit(unitid)
			dynamic(unitid)
			MF_specialremove(unitid,2)
		end
	elseif (style == "emptyconvert") then
		for a,mats2data in ipairs(mats2) do
			local mat2 = mats2data[1]
			local i = mats2data[2]
			local j = mats2data[3]
			local unitname = unitreference[mat2]
			local newunitid = MF_emptycreate(unitname,i,j)
			local newunit = mmf.newObject(newunitid)
			
			cdata[1] = "empty"
			
			local id = newid()
			local dir = emptydir(i,j)
			
			if (dir == 4) then
				dir = fixedrandom(0,3)
			end
			
			newunit.values[ONLINE] = 1
			newunit.values[XPOS] = i
			newunit.values[YPOS] = j
			newunit.values[DIR] = dir
			newunit.values[ID] = id
			newunit.values[EFFECT] = 1
			newunit.flags[9] = true
			newunit.flags[CONVERTED] = true
			
			cdata[2] = mat2
			addundo({"convert",cdata[1],cdata[2],id,id,i,j,dir})
			addundo({"create",mat2,id,-1,"emptyconvert",i,j,dir})
			
			addunit(newunitid)
			addunitmap(newunitid,i,j,newunit.strings[UNITNAME])
			dynamic(newunitid)
			
			newunit.originalname = "empty"
			
			local tileid = i + j * roomsizex
			if (emptydata[tileid] == nil) then
				emptydata[tileid] = {}
			end
			
			emptydata[tileid]["conv"] = true
			
			if (newunit.strings[UNITTYPE] == "text") then
				updatecode = 1
			else
				if (featureindex["word"] ~= nil) then
					for i,v in ipairs(featureindex["word"]) do
						local rule = v[1]
						if (rule[1] == newunit.strings[UNITNAME]) then
							updatecode = 1
						elseif (unitid ~= 2) then
							if (rule[1] == unit.strings[UNITNAME]) then
								updatecode = 1
							end
						end
					end
				end
			end
		end
	end
end

function convert(stuff,mats,dolevels_)
	local layer = map[0]
	local delthese = {}
	local mat1 = stuff
	local dolevels = dolevels_ or false
	
	if (dolevels == false) then
		if (mat1 ~= "empty") then
			local targets = {}
			
			if (unitlists[mat1] ~= nil) then
				targets = unitlists[mat1]
			end
			
			if (editor2.values[CURSORSEXIST] == 1) then
				if (featureindex[mat1] ~= nil) then
					for i,v in ipairs(featureindex[mat1]) do
						local rule = v[1]
						
						if (rule[2] == "is") and (rule[3] == "select") then
							editor.values[NAMEFLAG] = 0
							break
						end
					end
				end
			end
			
			if (#targets > 0) then
				for i,unitid in pairs(targets) do
					local unit = mmf.newObject(unitid)
					local x,y,dir,id = unit.values[XPOS],unit.values[YPOS],unit.values[DIR],unit.values[ID]
					local name = getname(unit)
					
					local reverting = false
					local mats2 = {}

					if (unit.flags[CONVERTED] == false) then
						for a,matdata in pairs(mats) do
							local mat2 = matdata[1]
							local conds = matdata[2]
							local op = matdata[3]
							
							if (op == "write") then
								mat2 = "text_" .. matdata[1]
							end
							
							if (reverting == false) then
								local objectfound = false
								
								if (unitreference[mat2] ~= nil) and (mat2 ~= "level") and (mat2 ~= "createall") then
									local object = unitreference[mat2]
									
									if (tileslist[object]["name"] == mat2) and ((changes[object] == nil) or (changes[object]["name"] == nil)) then
										objectfound = true
									elseif (changes[object] ~= nil) then
										if (changes[object]["name"] ~= nil) and (changes[object]["name"] == mat2) then
											objectfound = true
										end
									end
								else
									objectfound = true
								end
								
								if testcond(conds,unit.fixed) and objectfound then
									local ingameid = 0
									if (a == 1) then
										ingameid = id
									elseif (a > 1) then
										ingameid = newid()
									end
									
									if (mat2 == "revert") then
										if (unit.strings[UNITNAME] ~= unit.originalname) then
											reverting = true
										end
									end
									
									persistrevert = nil
									if persistreverts ~= nil then
										persistrevert = persistreverts[id]
									end
									if persistrevert ~= nil then
										reverting = true
									end
									
									if (mat2 ~= "revert") or ((mat2 == "revert") and reverting) then
										table.insert(mats2, {mat2,ingameid,id})
										unit.flags[CONVERTED] = true
									end
								end
							else
								break
							end
						end
					end
					
					if (#mats2 > 0) then
						addaction(unit.fixed,{"convert",mats2})
					end
				end
			end
		elseif (mat1 == "empty") then
			local convunitmap = {}
			
			for a,unit in pairs(units) do
				local tileid = unit.values[XPOS] + unit.values[YPOS] * roomsizex
				convunitmap[tileid] = 1
			end
			
			for i=0,roomsizex-1 do
				for j=0,roomsizey-1 do
					local empty = true
					local mats2 = {}
					
					local tileid = i + j * roomsizex
					if (convunitmap[tileid] ~= nil) then
						empty = false
					end
					
					if (emptydata[tileid] ~= nil) then
						if (emptydata[tileid]["conv"] ~= nil) and emptydata[tileid]["conv"] then
							empty = false
						end
					end
					
					if (layer:get_x(i,j) ~= 255) then
						empty = false
					end
					
					if empty then
						for a,matdata in pairs(mats) do
							local mat2 = matdata[1]
							local conds = matdata[2]
							local op = matdata[3]
							
							if (op == "write") then
								mat2 = "text_" .. matdata[1]
							end
							
							local objectfound = false
							
							if (unitreference[mat2] ~= nil) and (mat2 ~= "level") then
								local object = unitreference[mat2]
								
								if (tileslist[object]["name"] == mat2) and ((changes[object] == nil) or (changes[object]["name"] == nil)) then
									objectfound = true
								elseif (changes[object] ~= nil) then
									if (changes[object]["name"] ~= nil) and (changes[object]["name"] == mat2) then
										objectfound = true
									end
								end
							elseif (mat2 == "level") then
								objectfound = true
							end

							if (mat2 ~= "empty") and objectfound then
								if testcond(conds,2,i,j) then
									table.insert(mats2, {mat2,i,j})
								end
							end
						end
					end
					
					if (#mats2 > 0) then
						addaction(2,{"emptyconvert",mats2})
					end
				end
			end
		end
	end
	
	if (mat1 == "level") and dolevels then
		for i,v in ipairs(mats) do
			table.insert(levelconversions, v)
		end
	end
end