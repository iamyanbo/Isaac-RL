-- OPERATIONAL FIXTURE ONLY. Install in reserved instance 3, never training.
-- Creates known native entity classes to test readout, not policy competence.
local mod = RegisterMod("Isaac RL Observation Fixture",1)
local ticks = 0
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED,function() ticks=0 end)
mod:AddCallback(ModCallbacks.MC_POST_UPDATE,function()
    ticks=ticks+1
    if ticks ~= 10 then return end
    local player = Isaac.GetPlayer(0)
    local enemy = Isaac.Spawn(EntityType.ENTITY_GAPER,0,0,Vector(450,200),Vector(0,0),nil)
    local bomb = Isaac.Spawn(EntityType.ENTITY_BOMB,0,0,Vector(120,360),Vector(0,0),player):ToBomb()
    bomb:SetExplosionCountdown(120)
    -- Native Spawn accepts an explicit random seed. Sharing it must not merge
    -- these two distinct entities or crash the history validator.
    Game():Spawn(EntityType.ENTITY_PROJECTILE,0,Vector(250,180),Vector(2,1),enemy,0,581234)
    Game():Spawn(EntityType.ENTITY_PROJECTILE,4,Vector(280,180),Vector(-2,1),enemy,0,581234)
    local beam = EntityLaser.ShootAngle(1,Vector(100,160),0,60,Vector(0,0),enemy)
    beam.CollisionDamage=1
    local ring = Isaac.Spawn(EntityType.ENTITY_LASER,2,2,Vector(450,320),Vector(0,0),enemy):ToLaser()
    ring.Radius=40
    ring.Timeout=60
    player:TakeDamage(1,0,EntityRef(enemy),0)
end)
