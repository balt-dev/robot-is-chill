levelblockAlphaBaba = levelblock
function levelblock()
	levelblockAlphaBaba()
	if (generaldata.strings[CURRLEVEL] == "124level") then
		MF_loadbackimage("messagehide")
		local hiddenobject = "me"
		if (hasfeature(hiddenobject, "is","done",1) == nil) and (notdone ~= true) then
			MF_done_single()
			ending = newEndingAlphaBaba
			notdone = true
			--MF_allisdone()
			addbaserule("fungus", "is", "end2")
			addbaserule("fungus", "is", "you")
			
			writetext("all is not done",0,screenw * 0.5 - 12,screenh * 0.5 + 60,0,true,3,true,{4,1},3)
			setsoundname("removal",2)
			HACK_INFINITY = 0

			MF_playsound("intro_open_1b")
			--MF_playsound("doneall_c") this sound does not exist anymore apparently
			
			--MF_playsound("intro_disappear_1")
			
			setsoundname("removal",2)
			
			local isignid = MF_specialcreate("Special_infinity")
			local isign = mmf.newObject(isignid)
			
			isign.x = screenw * 0.5
			isign.y = screenh * 0.5 - 36
			MF_setcolour(isignid,4,1)
			isign.layer = 3
			
		elseif (hasfeature(hiddenobject, "is","done",1) ~= nil) then
			notdone = false
			notdone2 = false
		end

		
		if notdone then
			addbaserule("hedge", "is", "pink")
		end
		
		local isyou = getunitswitheffect("you",delthese)
		for id,unit in ipairs(isyou) do
			local x,y = unit.values[XPOS],unit.values[YPOS]
			
			local ending = findfeature(nil,"is","end2")
			
			if (ending ~= nil) then
				for a,b in ipairs(ending) do
					if (b[1] ~= "empty") then
						local flag = findtype(b,x,y,0)
						
						if (#flag > 0) then
							for c,d in ipairs(flag) do
								if floating(d,unit.fixed) and (generaldata.values[MODE] == 0) then
									MF_particles("unlock",x,y,10,1,4,1,1)
									MF_end(unit.fixed,d)
									break
								end
							end
						end
					end
				end
			end
		end
		
		
		
	else
		ending = endingAlphaBaba
		notdone = false
		notdone2 = false
	end
end

function newEndingAlphaBaba(enddataid)
	local enddata = mmf.newObject(enddataid)
	for i,v in ipairs(features) do
		local rule = v[1]
		
		if (rule[1] == "fungus") and (rule[2] == "is") and (rule[3] == "you") then
			v[2] = {{"never",{}}}
		end
	end
	
	enddata.values[ENDTIMER] = enddata.values[ENDTIMER] + 1
	local phase = enddata.values[ENDPHASE]
	local timer = enddata.values[ENDTIMER]
	local ender = enddata.values[4]
	local ended = enddata.values[3]
	
	if (phase == 1) then
		generaldata.values[ONLYARROWS] = 1
		if (timer == 1) and (1 == 0) then
			clearunits()
			MF_loop("clear",1)
			enddata.values[ENDPHASE] = 4
			enddata.values[ENDTIMER] = -150
			generaldata.values[IGNORE] = 1
			--MF_playmusic("ending",0,1,1)
			
		end
		
		if (timer == 2) then
			local unit = mmf.newObject(ended)
			local c1,c2 = getcolour(ended)
			MF_particles("smoke",unit.values[XPOS],unit.values[YPOS],20,c1,c2,1,1)
			
			unit.visible = false
			generaldata2.values[ENDINGGOING] = 1
			
			featureindex["text"] = {}
			
		end
		
		if (timer == 13) then
			--MF_playsound_musicvolume("ending_reverse_baba_loop_start", 4)
		end
			
		if (timer == 52) then
			local unit = mmf.newObject(ended)
			local flowerid = MF_specialcreate("Flower_center")
			local flower = mmf.newObject(flowerid)
			flower.layer = 3
			
			flower.strings[2] = "you"
			flower.values[10] = 1
			flower.values[8] = 15
			flower.x = tilesize * unit.values[XPOS] * spritedata.values[TILEMULT] + tilesize * 1.0 * spritedata.values[TILEMULT]
			flower.y = tilesize * unit.values[YPOS] * spritedata.values[TILEMULT] + tilesize * 0.5 * spritedata.values[TILEMULT]
		end
		
		if (timer > 0) then
			for i,unit in ipairs(units) do
				if (unit.strings[UNITTYPE] == "text") then
					unit.x = unit.x + math.random(-1,1)
					unit.y = unit.y + math.random(-1,1)
					unit.values[POSITIONING] = 0
				end
			end
		end
		
		if (timer == 220) then
			local textconverts = {}
			--MF_playsound_channel("confirm",5)
			
			for i,unit in ipairs(units) do
				if (unit.strings[UNITTYPE] == "text") then
					table.insert(textconverts, unit)
				end
			end
			
			for i,unit in pairs(textconverts) do
				doconvert({unit.fixed,"convert","blossom",unit.values[ID],unit.values[ID]})
				updatecode = 0
			end
		end
		
		if (timer > 220) then
			local eunit = mmf.newObject(ender)
			local changethese = {}
			local deletethese = {}
			local blossoms = {}
			
			for i,unit in ipairs(units) do
				if (unit.strings[UNITNAME] == "blossom") then
					if (unit.values[FLOAT] == 3) then
						if (unit.values[XPOS] == eunit.values[XPOS]) and (unit.values[YPOS] == eunit.values[YPOS]) then
							table.insert(changethese, unit)
						end
					elseif (unit.values[FLOAT] == 0) and (timer > 230) then
						local edunit = mmf.newObject(ended)
						
						unit.values[XPOS] = unit.values[XPOS] + (edunit.values[XPOS] - unit.values[XPOS]) * 0.2
						unit.values[YPOS] = unit.values[YPOS] + (edunit.values[YPOS] - unit.values[YPOS]) * 0.2
						unit.values[POSITIONING] = 0
						
						local dist = math.abs(edunit.values[XPOS] - unit.values[XPOS]) + math.abs(edunit.values[YPOS] - unit.values[YPOS])
						
						if (dist < 0.5) then
							table.insert(deletethese, unit)
						end
					end
					table.insert(blossoms, 1)
				end
			end
			
			if (#changethese > 0) then
				MF_playsound("intro_flower_" .. tostring(math.random(1,7)))
			end
			
			for i,unit in ipairs(deletethese) do
				MF_particles("bling",unit.values[XPOS],unit.values[YPOS],10,0,3,1,1)
				delete(unit.fixed)
				updatecode = 0
			end
			
			for i,unit in ipairs(changethese) do
				MF_particles("bling",unit.values[XPOS],unit.values[YPOS],10,0,3,1,1)
				unit.values[FLOAT] = 0
				
				local tileid = unit.values[XPOS] + unit.values[YPOS] * roomsizex
				if (unitmap[tileid] ~= nil) then
					for a,b in ipairs(unitmap[tileid]) do
						if (b == unit.fixed) then
							table.remove(unitmap[tileid], a)
						end
					end
				end
			end
			
			if (#blossoms == 0) then
				generaldata.values[ONLYARROWS] = 1
				enddata.values[ENDPHASE] = 2
				enddata.values[ENDTIMER] = 0
			end
		end
	end
		
	if (phase == 2) then
		if (timer == 30) then
			local allis = MF_findspecial("you")
			
			for i,flowerid in ipairs(allis) do
				local flower = mmf.newObject(flowerid)
				flower.values[6] = 3
			end
		end
		
		if (timer == 250) then
			local allis = MF_findspecial("you")
			
			for i,flowerid in ipairs(allis) do
				local flower = mmf.newObject(flowerid)
				flower.values[6] = 1
			end
			
			MF_playsound_musicvolume("ending_reverse_baba_loop_end", 4)
		end
		
		if (timer == 430) then
			generaldata.values[TRANSITIONREASON] = 0
			dotransition()
		elseif (timer > 430) and (generaldata.values[TRANSITIONED] == 1) then
			clearunits()
			MF_loop("clear",1)
			generaldata.values[MODE] = 3
			
			enddata.values[ENDPHASE] = 3
			enddata.values[ENDTIMER] = 0
			generaldata.values[ONLYARROWS] = 0
			generaldata.values[IGNORE] = 1
			generaldata.values[TRANSITIONED] = 0
			generaldata.values[TRANSITIONREASON] = 0
			
			local xoffset = (screenw - roomsizex * tilesize * spritedata.values[TILEMULT]) * 0.5
			local yoffset = (screenh - roomsizey * tilesize * spritedata.values[TILEMULT]) * 0.5
			
			MF_setroomoffset(xoffset,yoffset)
			--MF_loadbackimage("island,island_decor")
			MF_loadbackimage("flowersmall,inf")
			levelborder()
			
			local flowerid = MF_specialcreate("Flower_center")
			local flower = mmf.newObject(flowerid)
			
			flower.strings[2] = "endingflower"
			flower.values[10] = 1
			flower.values[8] = 2
			flower.values[6] = 1
			flower.x = tilesize * 18 * spritedata.values[TILEMULT]
			flower.y = tilesize * 10 * spritedata.values[TILEMULT]
			
			local otherareas =
			{
				{
					name = "mountain",
					xpos = 10,
					ypos = 3,
					colour = {1,4},
					level = "232level",
				},
				{
					name = "cave",
					xpos = 11,
					ypos = 5,
					colour = {2,2},
					level = "179level",
				},
				{
					name = "garden",
					xpos = 13,
					ypos = 8,
					colour = {4,1},
					level = "180level",
				},
				{
					name = "ruins",
					xpos = 18,
					ypos = 6,
					colour = {2,4},
					level = "206level",
				},
				{
					name = "space",
					xpos = 20,
					ypos = 4,
					colour = {3,2},
					level = "87level",
				},
				{
					name = "forest",
					xpos = 21,
					ypos = 6,
					colour = {5,2},
					level = "169level",
				},
				{
					name = "island",
					xpos = 18,
					ypos = 9,
					colour = {5,3},
					level = "207level",
				},
				{
					name = "lake",
					xpos = 16,
					ypos = 10,
					colour = {1,4},
					level = "177level",
				},
				{
					name = "fall",
					xpos = 21,
					ypos = 13,
					colour = {2,4},
					level = "16level",
				},
				{
					name = "abstract",
					xpos = 24,
					ypos = 10,
					colour = {0,2},
					level = "182level",
				},
			}
			
			--[[for i,v in ipairs(otherareas) do
				local status = tonumber(MF_read("save",generaldata.strings[WORLD] .. "_complete",v.level))
				
				if (status == 1) then
					local flowerid = MF_specialcreate("Flower_center")
					local flower = mmf.newObject(flowerid)
					
					flower.strings[1] = tostring(v.colour[1]) .. "," .. tostring(v.colour[2])
					flower.strings[2] = "otherarea"
					flower.values[10] = 2
					flower.values[8] = 1
					flower.values[6] = 1
					flower.x = Xoffset + (v.xpos * tilesize + tilesize * 0.5) * spritedata.values[TILEMULT]
					flower.y = Yoffset + (v.ypos * tilesize + tilesize * 0.5) * spritedata.values[TILEMULT]
				end
			end]]
		end
	end
	
	if (phase == 3) then
		if (timer == 30) then
			MF_playsound("dididi")
		end
		if (timer == 700) then
			dotransition()
		elseif (timer > 700) and (generaldata.values[TRANSITIONED] == 1) then
			clearunits()
			MF_loop("clear",1)
			generaldata.values[MODE] = 3
			enddata.values[ENDPHASE] = 4
			enddata.values[ENDTIMER] = -100
			--MF_playmusic("ending",0,1,1)
			MF_playmusic("noise",0,1,1)
			MF_channelvolume(1,0)
			MF_channelvolume(2,0)
		end
	end
	
	if (phase == 4) then
		--[[local tiles =
		{
			baba = {
				dir = 0,
				colour = {4,1},
			},
			is = {
				dir = 1,
				colour = {0,3},
			},
			you = {
				dir = 2,
				colour = {4,1},
			},
			made = {
				dir = 10,
				colour = {1,3},
			},
			by = {
				dir = 3,
				colour = {1,3},
			},
			arvi = {
				dir = 4,
				colour = {0,3},
			},
			teikari = {
				dir = 5,
				colour = {0,3},
			},
			port = {
				dir = 6,
				colour = {1,3},
			},
			mp2_logo = {
				dir = 7,
				colour = {4,1},
			},
			mp2 = {
				dir = 8,
				colour = {0,3},
			},
			games = {
				dir = 9,
				colour = {0,3},
			},
			with = {
				dir = 11,
				colour = {1,3},
			},
			mmf2 = {
				dir = 12,
				colour = {5,3},
			},
			click = {
				dir = 13,
				colour = {5,3},
			},
			team = {
				dir = 14,
				colour = {5,3},
			},
			baba_obj = {
				dir = 15,
				colour = {0,3},
			},
		}
		
		local time1 = 80
		
		if (timer == time1) then
			ending_load(tiles,"baba",-1,0)
		elseif (timer == time1 + 40) then
			ending_load(tiles,"is",0,0)
		elseif (timer == time1 + 70) then
			ending_load(tiles,"you",1,0)
		end
		
		if (timer == 190) then
			ending_load(tiles,"baba_obj",-2.5,2)
		end
		
		time1 = 310
		
		if (timer == time1) then
			MF_hack_removecredit("baba")
		elseif (timer == time1 + 20) then
			MF_hack_removecredit("is")
		elseif (timer == time1 + 40) then
			MF_hack_removecredit("you")
		end
		
		time1 = 290
		
		if (timer == time1 + 55) then
			ending_load(tiles,"made",-1.5,0)
		elseif (timer == time1 + 85) then
			ending_load(tiles,"by",-0.5,0)
		elseif (timer == time1 + 136) then
			ending_load(tiles,"arvi",0.5,0)
		elseif (timer == time1 + 175) then
			ending_load(tiles,"teikari",1.5,0)
		end
		
		time1 = 490
		
		if (timer == time1) then
			MF_hack_movecredit("baba_obj","1","0",22)
		elseif (timer == time1 + 10) then
			MF_hack_movecredit("baba_obj","1","0",19)
		elseif (timer == time1 + 20) then
			MF_hack_movecredit("baba_obj","1","0",22)
		elseif (timer == time1 + 40) then
			MF_hack_movecredit("baba_obj","0","-1",16)
		elseif (timer == time1 + 50) then
			MF_hack_movecredit("baba_obj","0","-1",16)
			MF_hack_movecredit("arvi","0","-1",-1)
			
			local stuff = MF_findspecial("arvi")
			local stuff2 = MF_findspecial("teikari")
			
			for a,b in ipairs(stuff) do
				MF_setcolour(b,0,1)
			end
			
			for a,b in ipairs(stuff2) do
				MF_setcolour(b,0,1)
			end
		end
		
		time1 = 605
		
		if (timer == time1) then
			MF_hack_removecredit("made")
		elseif (timer == time1 + 10) then
			ending_load(tiles,"port",-2,0)
		elseif (timer == time1 + 20) then
			MF_hack_movecredit("by","-0.5","0",-1)
			MF_hack_movecredit("baba_obj","-0.5","0",23)
		elseif (timer == time1 + 40) then
			MF_hack_movecredit("baba_obj","0","-1",16)
			MF_hack_removecredit("arvi")
		elseif (timer == time1 + 60) then
			MF_hack_movecredit("baba_obj","-1","0",23)
			MF_hack_removecredit("teikari")
		elseif (timer == time1 + 68) then
			ending_load(tiles,"mp2_logo",0,0)
		elseif (timer == time1 + 90) then
			ending_load(tiles,"mp2",1,0)
		elseif (timer == time1 + 120) then
			ending_load(tiles,"games",2,0)
		end
		
		time1 = 816
			
		if (timer == time1) then
			MF_hack_removecredit("port")
			MF_hack_movecredit("baba_obj","0","1",18)
			MF_hack_movecredit("by","0","1",-1)
		elseif (timer == time1 + 10) then
			ending_load(tiles,"made",-1,-1)
		elseif (timer == time1 + 24) then
			MF_hack_removecredit("mp2_logo")
			ending_load(tiles,"with",0,-1)
		elseif (timer == time1 + 38) then
			ending_load(tiles,"mmf2",1.2,-1)
			MF_hack_removecredit("mp2")
		elseif (timer == time1 + 60) then
			MF_hack_removecredit("games")
		elseif (timer == time1 + 65) then
			ending_load(tiles,"click",0.2,1)
		elseif (timer == time1 + 110) then
			ending_load(tiles,"team",1.2,1)
		end
		
		time1 = 836
		
		if (timer == time1) then
			MF_hack_movecredit("baba_obj","-1","0",23)
		elseif (timer == time1 + 10) then
			MF_hack_movecredit("baba_obj","-1","0",17)
		elseif (timer == time1 + 20) then
			MF_hack_movecredit("baba_obj","-1","0",20)
		elseif (timer == time1 + 30) then
			MF_hack_movecredit("baba_obj","-1","0",17)
		elseif (timer == time1 + 40) then
			MF_hack_movecredit("baba_obj","-1","0",23)
		elseif (timer == time1 + 70) then
			MF_hack_removecredit("baba_obj")
		end
		
		time1 = 1040
		
		if (timer == time1) then
			MF_hack_removecredit("made")
		elseif (timer == time1 + 10) then
			MF_hack_removecredit("with")
		elseif (timer == time1 + 20) then
			MF_hack_removecredit("mmf2")
		elseif (timer == time1 + 30) then
			MF_hack_removecredit("by")
		elseif (timer == time1 + 40) then
			MF_hack_removecredit("click")
		elseif (timer == time1 + 50) then
			MF_hack_removecredit("team")
		end]]
		
		if (timer == 1) then
			generaldata.values[MODE] = 2
			enddata.values[ENDPHASE] = 5
			enddata.values[ENDTIMER] = 0
			enddata.values[ENDCREDITS] = 1
		end
	elseif (phase == 5) then
		local tiles =
		{
			you = {
				dir = 2,
				colour = {4,1},
			},
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
			ending_load(tiles,"you",0.5,16.5)
			ending_load(tiles,"keke",-3,6)
			ending_load(tiles,"skull",3,12.6)
			ending_load(tiles,"flag",3.2,21.6)
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
		elseif (timer == 3) then
			MF_playsound("themetune")
		elseif (timer == 60) then
			writetext("thank you for playing",0,screenw * 0.5,screenh * 0.5 - 96 * 1.6,0,true,2,nil,nil,2)
		end
		if notdone2 ~= true then
			notdone2 = true
			
			local isignid = MF_specialcreate("Special_infinity")
			local isign = mmf.newObject(isignid)
			
			isign.x = screenw * 0.5
			isign.y = screenh * 0.5 - 64
			MF_setcolour(isignid,4,1)
			isign.layer = 3
		end

	end
	
end