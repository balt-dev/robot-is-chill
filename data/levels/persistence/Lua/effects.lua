function effects(timer)
	doeffect(timer,nil,"win","unlock",1,2,20,{2,4})
	doeffect(timer,nil,"best","unlock",6,30,2,{2,4})
	doeffect(timer,nil,"tele","glow",1,5,20,{1,4})
	doeffect(timer,nil,"hot","hot",1,80,10,{0,1})
	doeffect(timer,nil,"bonus","bonus",1,2,20,{4,1})
	doeffect(timer,nil,"wonder","wonder",1,10,5,{0,3})
	doeffect(timer,nil,"sad","tear",1,2,20,{3,2})
	doeffect(timer,nil,"sleep","sleep",1,2,60,{3,2})
	doeffect(timer,nil,"power","electricity",2,5,8,{2,4})
	doeffect(timer,nil,"broken","error",3,10,8,{2,2})
	--doeffect(timer,"play",nil,"music",1,2,30,{0,3})
	
	local rnd = math.random(2,4)
	doeffect(timer,nil,"end","unlock",1,1,10,{1,rnd},"inwards")
	--rnd = math.random(0,2)
	--doeffect(timer,"melt","unlock",1,1,10,{4,rnd},"inwards")
	
	if currentlevel == "42level" then
		doeffect(timer,nil,"baserule","infinity",1,2,100,{0,3})
	else
		doeffect(timer,nil,"baserule","infinity",1,2,20,{0,3})
	end
end