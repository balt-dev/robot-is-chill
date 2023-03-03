--notes: timer = 60fps
function ending1(enddataid)
	local enddata = mmf.newObject(enddataid)
	local build = generaldata.strings[BUILD]
	
	enddata.values[ENDTIMER] = enddata.values[ENDTIMER] + 1
	local phase = enddata.values[ENDPHASE]
	local timer = enddata.values[ENDTIMER]
	--MF_letterclear("timer")
	--writetext(enddata.values[ENDTIMER],-1,16,12,"timer",false)
	
	if (phase == 1) then
		if (timer == 2) then
			generaldata2.values[ENDINGGOING] = 1
			generaldata.values[ONLYARROWS] = 1
			enddata.values[ENDPHASE] = 3
			enddata.values[ENDTIMER] = 0
		end
	end
	
	if (phase == 3) then
		if (timer == 30) then
			MF_playsound("dididi")
		end
		
		if (timer == 700) then
			local mx = screenw * 0.5
			local my = screenh * 0.5
			
			local letid1 = MF_effectcreate("Ending_theend")
			local letid2 = MF_effectcreate("Ending_theend")
			local let1 = mmf.newObject(letid1)
			local let2 = mmf.newObject(letid2)
			
			let1.x = mx
			let1.y = my + 4 - tilesize * 0.5
			let1.layer = 2
			MF_setcolour(letid1,1,2)
			
			let2.x = mx
			let2.y = my - tilesize * 0.5
			let2.layer = 2
			let2.values[15] = -1
		elseif (timer == 750) then
			local mx = screenw * 0.5
			local my = screenh * 0.5
			
			local letid3 = MF_effectcreate("Ending_theend_back")
			local let3 = mmf.newObject(letid3)
			let3.x = mx
			let3.y = my + 8 - tilesize * 0.5
			let3.layer = 2
			
			MF_playsound("themetune")
		elseif (timer == 1150) then
			dotransition()
		elseif (timer > 1150) and (generaldata.values[TRANSITIONED] == 1) then
			clearunits()
			MF_loop("clear",1)
			generaldata.values[MODE] = 3
			enddata.values[ENDPHASE] = 4
			enddata.values[ENDTIMER] = 0
		end
	end
	if phase == 4 then
		
		if (timer == 100) then
			generaldata.values[MODE] = 2
			enddata.values[ENDPHASE] = 5
			enddata.values[ENDTIMER] = 0
			enddata.values[ENDCREDITS] = 1
		end
	elseif (phase == 5) then
		
		local tiles =
		{
			keke = {
				dir = 24,
				colour = {2,2},
			},
			skull = {
				dir = 25,
				colour = {2,1},
			},
			flag = {
				dir = 26,
				colour = {2,4},
			},
			key = {
				dir = 27,
				colour = {2,4},
			},
			rocket = {
				dir = 28,
				colour = {1,1},
			},
		}
		
		if (timer == 100) then
			ending_load(tiles,"keke",-3.75,11.4)
			ending_load(tiles,"skull",3.5,19.4)
			ending_load(tiles,"flag",-3.5,32.5)
			ending_load(tiles,"key",3.5,39)
			ending_load(tiles,"rocket",-3.5,53.1)
		end
	elseif (phase == 6) then
		if (timer == 2) then
			local tiles =
			{
				baba = {
					dir = 21,
					colour = {0,3},
				},
			}
			
			ending_load(tiles,"baba",0,0.3)
		elseif (timer == 60) then
			writetext(langtext("ending"),0,screenw * 0.5,screenh * 0.5 - 96 * 1.6,0,true,2,nil,nil,2)
		end
	end
end

function ending2(enddataid)
	local enddata = mmf.newObject(enddataid)
	
	enddata.values[ENDTIMER] = enddata.values[ENDTIMER] + 1
	local phase = enddata.values[ENDPHASE]
	local timer = enddata.values[ENDTIMER]
	--MF_letterclear("timer")
	--writetext(enddata.values[ENDTIMER],-1,16,12,"timer",false)
	
	if (phase == 1) then
		if (timer == 2) then
			generaldata2.values[ENDINGGOING] = 1
			generaldata.values[ONLYARROWS] = 1
			enddata.values[ENDPHASE] = 2
			enddata.values[ENDTIMER] = 0
		end
	end
		
	if (phase == 2) then
		
		if (timer == 430) then
			generaldata.values[TRANSITIONREASON] = 0
			dotransition()
		elseif (timer > 430) and (generaldata.values[TRANSITIONED] == 1) then
			--clearunits()
			MF_loop("clear",1)
			
			enddata.values[ENDPHASE] = 3
			enddata.values[ENDTIMER] = 0
			generaldata.values[ONLYARROWS] = 0
			generaldata.values[IGNORE] = 1
			generaldata.values[TRANSITIONED] = 0
			generaldata.values[TRANSITIONREASON] = 0
			local xoffset = (screenw - roomsizex * tilesize * spritedata.values[TILEMULT]) * 0.5
			local yoffset = (screenh - roomsizey * tilesize * spritedata.values[TILEMULT]) * 0.5
			
			MF_setroomoffset(xoffset,yoffset)
			MF_loadbackimage("empty")
			levelborder()
		end
	end
	
	if (phase == 3) then
		generaldata2.values[ZOOM] = 1
		if (timer == 2) then
			dotransition()
		elseif (timer > 2) and (generaldata.values[TRANSITIONED] == 1) then
			--clearunits()
			MF_loop("clear",1)
			generaldata.values[MODE] = 3
			enddata.values[ENDPHASE] = 4
			enddata.values[ENDTIMER] = 0
			MF_channelvolume(1,0)
			MF_channelvolume(2,0)
			levelborder()
			MF_loadbackimage("cosmicbig")
			generaldata.values[IGNORE] = 0
			generaldata.values[ONLYARROWS] = 1
			MF_playmusic("lineend",1,1,1)
		end
	end
	
	if (phase == 4) then
		if not (hasfeature("baba","is","you")) then
			addbaserule("baba","is","you")
			addbaserule("bab","is","you")
			addbaserule("betababa","is","you")
		end
		if (timer == 2) then
			create("baba",16,9,0,16,20,nil,true,nil,nil)
		elseif (timer == 460) then
			writetext("I have transcended",0,screenw * 0.5 - 12,screenh * 0.5 + 80,"dialogue",true,2,true,{4,1})
		elseif timer == 918 then
			MF_letterclear("dialogue")
			writetext("My DNA becomes one with",0,screenw * 0.5 - 12,screenh * 0.5 + 60,"dialogue",true,2,true,{4,1})
			writetext("the rules of this universe",0,screenw * 0.5 - 12,screenh * 0.5 + 80,"dialogue",true,2,true,{4,1})
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persist2nd")
		elseif timer == 1150 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persist3rd")
		elseif timer == 1381 then
			MF_letterclear("dialogue")
			writetext("I am everywhere",0,screenw * 0.5 - 12,screenh * 0.5 + 40,"dialogue",true,2,true,{4,1})
			writetext("and nowhere at the same time",0,screenw * 0.5 - 12,screenh * 0.5 + 60,"dialogue",true,2,true,{4,1})
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persistinception")
		elseif timer == 1616 then
			MF_letterclear("dialogue")
			writetext("As the world flashes before my eyes,",0,screenw * 0.5 - 12,screenh * 0.5 + 60,"dialogue",true,2,true,{4,1})
			create("key",21,10,1,21,10,nil,true,nil,nil)
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("prison")
			editor2.strings[LEVELPARTICLES] = "smoke"
			MF_loop("spawnparticles",1)
		elseif timer == 1847 then
			MF_letterclear("dialogue")
			writetext("I reflect on its strange beauty",0,screenw * 0.5 - 12,screenh * 0.5 + 80,"dialogue",true,2,true,{4,1})
			for i,unit in ipairs(units) do
				unit.new = false
				if (getname(unit)=="key") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					delete(unit.fixed)
				end
			end
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persistmortality")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 2080 then
			MF_letterclear("dialogue")	
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("fleeting")
			editor2.strings[LEVELPARTICLES] = "snow"
			MF_loop("spawnparticles",1)
		elseif timer == 2310 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("wug")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 2538 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persist2")
		elseif timer == 2767 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("changeless")
			editor2.strings[LEVELPARTICLES] = "bubbles"
			MF_loop("spawnparticles",1)
		elseif timer == 3002 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("heat")
			editor2.strings[LEVELPARTICLES] = "sparks"
			MF_loop("spawnparticles",1)
		elseif timer == 3229 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("christmas")
			editor2.strings[LEVELPARTICLES] = "pollen"
			MF_loop("spawnparticles",1)
		elseif timer == 3462 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("persistbrick")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 3686 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("invert")
		elseif timer == 3919 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("frankenstein")
		elseif timer == 4154 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("mirror")
		elseif timer == 4381 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("starseeker")
		elseif timer == 4609 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("welcome")
		elseif timer == 4836 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("turing")
		elseif timer == 5068 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("creation")
			editor2.strings[LEVELPARTICLES] = "dust"
			MF_loop("spawnparticles",1)
		elseif timer == 5298 then
			for i,unit in ipairs(units) do
				unit.new = false
				if (getname(unit)=="baba") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					delete(unit.fixed)
				end
			end
			create("betababa",x,y,dir,x,y,nil,true,nil,nil)
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("beta")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 5524 then
			for i,unit in ipairs(units) do
				unit.new = false
				if (getname(unit)=="betababa") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					delete(unit.fixed)
				end
			end
			create("baba",x,y,dir,x,y,nil,true,nil,nil)
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("banana")
		elseif timer == 5584 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("concentrate")
		elseif timer == 5643 then
			for i,unit in ipairs(units) do
				unit.new = false
				if (getname(unit)=="baba") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					delete(unit.fixed)
				end
			end
			create("bab",x,y,dir,x,y,nil,true,nil,nil)
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("babsmall")
		elseif timer == 5754 then
			for i,unit in ipairs(units) do
				unit.new = false
				if (getname(unit)=="bab") then
					x,y,dir = unit.values[XPOS],unit.values[YPOS],unit.values[DIR]
					delete(unit.fixed)
				end
			end
			create("baba",x,y,dir,x,y,nil,true,nil,nil)
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("lastresort")
			editor2.strings[LEVELPARTICLES] = "snow"
			MF_loop("spawnparticles",1)
		elseif timer == 5872 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("tower")
			editor2.strings[LEVELPARTICLES] = "clouds"
			MF_loop("spawnparticles",1)
			writetext("dash attack",-1,240,12,"fakelevelname",false)
		elseif timer == 5995 then
			MF_letterclear("fakelevelname")
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("DONOTSPEAKHISNAME")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 6050 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("sticky")
			editor2.strings[LEVELPARTICLES] = "dust"
			MF_loop("spawnparticles",1)
		elseif timer == 6105 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("word")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif timer == 6163 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("alphababa")
		elseif timer == 6277 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("pond")
			editor2.strings[LEVELPARTICLES] = "soot"
			MF_loop("spawnparticles",1)
		elseif timer == 6452 then
			MF_movebackimage(-1000,-1000)
			MF_loadbackimage("daybreak2")
			editor2.strings[LEVELPARTICLES] = "none"
			MF_loop("spawnparticles",1)
		elseif (timer == 6900) then
			generaldata.values[MODE] = 2
			enddata.values[ENDPHASE] = 5
			enddata.values[ENDTIMER] = 0
			enddata.values[ENDCREDITS] = 1
			MF_loadbackimage("cosmicbig")
		end
	elseif (phase == 5) then
		delete = function ()
		end
		if generaldata2.values[ZOOM] < 5 then
			generaldata2.values[ZOOM] = generaldata2.values[ZOOM] + .01
		end
		if not (hasfeature("baba","is","you")) then; addbaserule("baba","is","you"); end
		generaldata.values[ONLYARROWS] = 0
		if timer == 400 then
			MF_playmusic("ending",0,1,1)
		end

	elseif (phase == 6) then
		if not (hasfeature("baba","is","you")) then; addbaserule("baba","is","you"); end
		if (timer == 2) then
			local tiles =
			{
				baba = {
					dir = 21,
					colour = {0,3},
				},
			}
			
			ending_load(tiles,"baba",0,0.3)
		elseif (timer == 60) then
			writetext(langtext("ending"),0,screenw * 0.5,screenh * 0.5 - 96 * 1.6,0,true,2,nil,nil,2)
		end
	end
end

function ending_load(database,name,x,y)
	local data = database[name]
	local unitid = MF_specialcreate("Ending_credits")
	local unit = mmf.newObject(unitid)
	
	unit.x = -96
	unit.y = -96
	
	unit.values[ONLINE] = 1
	unit.values[XPOS] = screenw * 0.5 + x * 96
	unit.values[YPOS] = screenh * 0.5 + y * 96
	unit.direction = data.dir
	unit.strings[2] = name
	
	MF_setcolour(unitid,data.colour[1],data.colour[2])
	
	return unitid
end