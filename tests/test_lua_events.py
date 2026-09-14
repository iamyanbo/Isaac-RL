"""Exercise the actual death callback, without starting or modifying a game."""
from pathlib import Path

from lupa import LuaRuntime


def event_runtime():
    lua = LuaRuntime()
    lua.execute('''
        callbacks = {}
        function RegisterMod()
            return {AddCallback=function(_, id, fn) callbacks[id]=fn end}
        end
        function require(_) return {} end
        function Game() return {} end
        Isaac = {DebugString=function() end}
        ModCallbacks = setmetatable({}, {__index=function(_,key) return key end})
        EntityFlag = {FLAG_FRIENDLY=1}
        function upvalue(fn, wanted, value)
            for i=1,100 do
                local name, old = debug.getupvalue(fn,i)
                if name == nil then break end
                if name == wanted then
                    if value ~= nil then debug.setupvalue(fn,i,value) end
                    return old
                end
            end
            error('Missing callback upvalue: ' .. wanted)
        end
        function deadEntity(hostile, background)
            return {
                IsActiveEnemy=function(_, includeDead) return includeDead and not background end,
                HasEntityFlags=function() return not hostile end
            }
        end
    ''')
    lua.execute((Path(__file__).resolve().parents[1]/"mod"/"main.lua").read_text())
    return lua


def test_dead_hostile_enemy_produces_kill_event_but_friendly_and_background_do_not():
    lua = event_runtime()
    lua.execute('''
        local fn=callbacks.MC_POST_NPC_DEATH
        upvalue(fn,'active',true)
        fn(nil,deadEntity(true,false))
        assert(upvalue(fn,'kills') == 1, 'dead hostile enemy must be counted')
        fn(nil,deadEntity(false,false))
        fn(nil,deadEntity(true,true))
        assert(upvalue(fn,'kills') == 1, 'friendly and background NPCs must not be counted')
        upvalue(fn,'active',false)
        fn(nil,deadEntity(true,false))
        assert(upvalue(fn,'kills') == 1, 'manual gameplay must not accrue training events')
    ''')


def test_damage_counts_resolved_hp_loss_not_rejected_hits_and_settles_death_once():
    lua = event_runtime()
    lua.execute('''
        local damage = callbacks.MC_ENTITY_TAKE_DMG
        local death = callbacks.MC_POST_NPC_DEATH
        local observe = upvalue(callbacks.MC_POST_UPDATE,'observeDamage')
        local settle = upvalue(death,'observeEnemyHealth')
        upvalue(damage,'active',true)
        local e = {InitSeed=7,HitPoints=10,exists=true,
            Exists=function(self) return self.exists end,
            ToPlayer=function() return nil end,
            IsActiveEnemy=function() return true end,
            IsVulnerableEnemy=function() return true end,
            HasEntityFlags=function() return false end}
        Isaac.GetRoomEntities=function() return e.exists and {e} or {} end
        observe()
        -- A rejected/blocked attempt does not remove HP.
        damage(nil,e,100)
        observe()
        assert(upvalue(settle,'damageDealt') == 0)
        -- Two hits in one update share the first pre-hit health baseline.
        damage(nil,e,4); e.HitPoints=6
        damage(nil,e,4); e.HitPoints=2
        observe()
        assert(upvalue(settle,'damageDealt') == 8)
        -- Lethal overkill pays only the remaining real health, once.
        damage(nil,e,100); e.HitPoints=0
        death(nil,e); observe()
        assert(upvalue(settle,'damageDealt') == 10)
        assert(upvalue(death,'kills') == 1)
        -- Despawn without a death confirmation is not damage evidence.
        e.InitSeed=8; e.HitPoints=10; e.exists=true
        observe(); damage(nil,e,100); e.exists=false; observe()
        assert(upvalue(settle,'damageDealt') == 10)
        -- Healing and inactive/manual gameplay cannot create damage reward.
        e.InitSeed=9; e.exists=true; e.HitPoints=5
        observe(); damage(nil,e,2); e.HitPoints=8; observe()
        assert(upvalue(settle,'damageDealt') == 10)
        upvalue(damage,'active',false)
        damage(nil,e,8); e.HitPoints=0; observe()
        assert(upvalue(settle,'damageDealt') == 10)
    ''')
