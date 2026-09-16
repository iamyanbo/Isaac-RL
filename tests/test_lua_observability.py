from test_lua_events import event_runtime


def test_native_bomb_exports_age_without_using_a_nonexistent_fuse_property():
    lua = event_runtime()
    lua.execute('''
        local export=upvalue(upvalue(upvalue(callbacks.MC_POST_UPDATE,'reply'),'snapshot'),'combatEntity')
        local e={InitSeed=8,SizeMulti={X=1,Y=1},EntityCollisionClass=4,CollisionDamage=1,FrameCount=17}
        local data={}; e.GetData=function() return data end
        local result=export(e,4)
        assert(result.bomb_age==17 and result.countdown==nil)
    ''')


def test_native_laser_export_uses_shape_specific_geometry_and_marks_invalid_curves():
    lua = event_runtime()
    lua.execute('''
        local reply = upvalue(callbacks.MC_POST_UPDATE,'reply')
        local snapshot = upvalue(reply,'snapshot')
        local export = upvalue(snapshot,'combatEntity')
        local e = {InitSeed=7,SizeMulti={X=1,Y=1},EntityCollisionClass=4,
            CollisionDamage=1,Position={X=20,Y=30}}
        local data={}; e.GetData=function() return data end
        local laser = {AngleDegrees=90,Radius=40,Timeout=20,
            IsCircleLaser=function() return true end,
            IsSampleLaser=function() return false end,
            GetEndPoint=function() error('circle has no line endpoint') end,
            GetSamples=function() error('circle samples contain native NaNs') end}
        e.ToLaser=function() return laser end
        local ring=export(e,7)
        assert(ring.circle and ring.radius==40 and ring.geometry_valid)
        assert(#ring.samples==0 and ring.endpoint[1]==20)
        laser.IsCircleLaser=function() return false end
        laser.GetEndPoint=function() return {X=20,Y=200} end
        local line=export(e,7)
        assert(#line.samples==2 and line.samples[2][2]==200 and line.geometry_valid)
        laser.IsSampleLaser=function() return true end
        laser.GetSamples=function()
            return setmetatable({Get=function(_,i) return {X=i,Y=i*i} end},
                {__len=function() return 20 end})
        end
        local curve=export(e,7)
        assert(curve.sample_count==20 and #curve.samples==8 and curve.geometry_valid)
        assert(curve.samples[1][1]==0 and curve.samples[8][1]==19)
        laser.GetSamples=function()
            return setmetatable({Get=function() return {X=0/0,Y=0/0} end},
                {__len=function() return 2 end})
        end
        local invalid=export(e,7)
        assert(not invalid.geometry_valid and #invalid.samples==0)
        assert(invalid.sample_count==2 and invalid.endpoint[1]==0)
    ''')


def test_native_identity_is_lifetime_stable_not_seed_or_lua_wrapper_equality():
    lua = event_runtime()
    lua.execute('''
        local export=upvalue(upvalue(upvalue(callbacks.MC_POST_UPDATE,'reply'),'snapshot'),'combatEntity')
        local function wrapper(data)
            return {InitSeed=123,SizeMulti={X=1,Y=1},EntityCollisionClass=4,
                CollisionDamage=1,FrameCount=17,GetData=function() return data end}
        end
        local a,b={},{}
        local first=export(wrapper(a),4)
        local again=export(wrapper(a),4)
        local second=export(wrapper(b),4)
        assert(first.id==second.id and first.track_id~=second.track_id)
        assert(first.track_id==again.track_id)
        local replacement=export(wrapper({}),4)
        assert(replacement.track_id>second.track_id)
    ''')
