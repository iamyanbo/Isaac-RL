-- Synchronous, localhost-only, newline-delimited JSON bridge.
-- Each step returns after exactly N game updates, or on death/floor completion.
local mod = RegisterMod("Isaac RL Bridge", 1)
local json = require("json")
local ok, socket = pcall(require, "socket")
if not ok then
    Isaac.DebugString("ISAAC_RL ERROR: launch the game with --luadebug: " .. tostring(socket))
    return
end
local game = Game()
local PORT = tonumber(os.getenv("ISAAC_RL_PORT")) or 9999
local client, pending, remaining, resetting = nil, nil, 0, false
local episode, sequence, episodeFrame = 0, 0, 0
local active, lastError = false, ""
local action, previousAction = {}, {}
local moves = {{0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}}
local visits, cleared, cellVisits = {}, {}, {}
local damageTaken, damageDealt, kills = 0, 0, 0
local enemyHealth = {}
local damageAttempted = 0
local bossSeen, bossDefeated = false, false
local lastConnect, responseId = 0, 0
local resetFromEpisode = -1

local function observeEnemyHealth(entity, current)
    local key = entity.InitSeed
    if key == nil then return end
    local previous = enemyHealth[key]
    if previous then damageDealt = damageDealt + math.max(0,previous-current) end
    enemyHealth[key] = current
end

local function observeDamage()
    if not active then return end
    -- Sample native resolved health, independent of TAKE_DMG callback timing
    -- and vulnerability predicates. Disappearance alone never means damage.
    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsActiveEnemy(true) and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
            observeEnemyHealth(entity,math.max(0,entity.HitPoints))
        end
    end
end

local function disconnect(message)
    if client then client:close() end
    client, active, pending = nil, false, nil
    action, previousAction = {}, {}
    lastError = tostring(message or "disconnected")
    Isaac.DebugString("ISAAC_RL: " .. lastError)
end

local function send(value)
    if not client then return false end
    local wire = json.encode(value) .. "\n"
    client:settimeout(5)
    local sent, err = client:send(wire)
    if not sent then disconnect(err); return false end
    return true
end

local function arrayPosition(entity)
    return {entity.Position.X, entity.Position.Y, entity.Velocity.X, entity.Velocity.Y}
end

-- Sidecar leaves the legacy 12-column entity wire format intact. These are
-- current native properties, never future states or recommended controls.
local function combatEntity(e, kind)
    local result = {id=e.InitSeed, size_multi={e.SizeMulti.X,e.SizeMulti.Y},
        collision=e.EntityCollisionClass, collision_damage=e.CollisionDamage}
    if kind == 1 then
        local npc = e:ToNPC()
        result.npc_state, result.state_frame = npc.State, npc.StateFrame
        result.animation_frame = e:GetSprite():GetFrame()
    elseif kind == 2 then
        local projectile = e:ToProjectile()
        result.height, result.falling_speed, result.falling_accel =
            projectile.Height, projectile.FallingSpeed, projectile.FallingAccel
    elseif kind == 4 then
        result.countdown = e:ToBomb().ExplosionCountdown
    elseif kind == 7 then
        local laser = e:ToLaser()
        result.circle, result.sample = laser:IsCircleLaser(), laser:IsSampleLaser()
        -- Native GetSamples() is NOT meaningful for circles (verified NaNs),
        -- and may be empty for straight lasers. Use each shape's own geometry.
        local endpoint = result.circle and e.Position or laser:GetEndPoint()
        result.endpoint = {endpoint.X,endpoint.Y}
        result.angle = result.circle and 0 or laser.AngleDegrees
        result.radius = result.circle and laser.Radius or 0
        result.timeout, result.samples = laser.Timeout, {}
        result.sample_count, result.geometry_valid = 0, true
        if not result.circle and result.sample then
            local samples = laser:GetSamples()
            result.sample_count = #samples
            -- Include both ends; bounded polyline, original count for audits.
            local count = math.min(8,#samples)
            result.geometry_valid = count >= 2
            for i=0,count-1 do
                local index = count == 1 and 0 or math.floor(i*(#samples-1)/(count-1))
                local point = samples:Get(index)
                result.samples[#result.samples+1] = {point.X,point.Y}
            end
        elseif not result.circle then
            result.samples = {{e.Position.X,e.Position.Y},result.endpoint}
            result.sample_count = 2
        end
        local function finite(value) return value == value and math.abs(value) < math.huge end
        for _, value in ipairs({result.angle,result.radius,endpoint.X,endpoint.Y}) do
            if not finite(value) then result.geometry_valid=false end
        end
        for _, point in ipairs(result.samples) do
            if not finite(point[1]) or not finite(point[2]) then result.geometry_valid=false end
        end
        if not result.geometry_valid then
            -- Explicit missing-geometry mask, never emit invalid JSON or claim
            -- that an uninitialized native curve is a valid zero-length beam.
            result.endpoint, result.angle, result.radius = {0,0},0,0
            result.samples = {}
        end
    end
    return result
end

local function snapshot()
    local player, room, level = Isaac.GetPlayer(0), game:GetRoom(), game:GetLevel()
    local roomId = level:GetCurrentRoomIndex()
    local entities, doors, grid, combat = {}, {}, {}, {}
    local enemies = 0
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        local kind = nil
        if e:IsActiveEnemy(false) and not e:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) and not e:IsDead() then
            kind = 1
            enemies = enemies + 1
            if e:IsBoss() and room:GetType() == RoomType.ROOM_BOSS then bossSeen = true end
        elseif e.Type == EntityType.ENTITY_PROJECTILE then kind = 2
        elseif e.Type == EntityType.ENTITY_PICKUP then kind = 3
        elseif e.Type == EntityType.ENTITY_BOMB then kind = 4
        elseif e.Type == EntityType.ENTITY_FIREPLACE and e.HitPoints > 0 then kind = 5
        elseif e.Type == EntityType.ENTITY_TEAR then kind = 6
        elseif e.Type == EntityType.ENTITY_LASER then kind = 7
        end
        if kind then
            local p = arrayPosition(e)
            p[5],p[6],p[7],p[8] = kind,e.Type,e.Variant,e.SubType
            p[9],p[10],p[11],p[12] = e.HitPoints,e.MaxHitPoints,e.Size,e.InitSeed
            entities[#entities+1] = p
            combat[#combat+1] = combatEntity(e,kind)
        end
    end
    local isClear = room:IsClear()
    if isClear then cleared[tostring(roomId)] = true end
    if level:GetStage() == 1 and room:GetType() == RoomType.ROOM_BOSS and
        bossSeen and isClear and enemies == 0 and not player:IsDead() then
        bossDefeated = true
    end
    for slot = 0, 7 do
        local door = room:GetDoor(slot)
        if door then
            doors[#doors+1] = {
                slot=slot, x=door.Position.X, y=door.Position.Y,
                open=door:IsOpen(), locked=door:IsLocked(), target=door.TargetRoomIndex,
                type=door.TargetRoomType, visits=visits[tostring(door.TargetRoomIndex)] or 0
            }
        end
    end
    for i = 0, room:GetGridSize()-1 do
        local g = room:GetGridEntity(i)
        if g then
            local p = room:GetGridPosition(i)
            grid[#grid+1] = {p.X,p.Y,g:GetType(),room:GetGridCollision(i),g.State}
        end
    end
    local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
    return {
        type="state", protocol=1, id=responseId, sequence=sequence, episode=episode,
        frame=game:GetFrameCount(), episode_frame=episodeFrame,
        seed=game:GetSeeds():GetStartSeedString(), stage=level:GetStage(), stage_type=level:GetStageType(),
        difficulty=game.Difficulty, character=player:GetPlayerType(),
        player={x=player.Position.X,y=player.Position.Y,vx=player.Velocity.X,vy=player.Velocity.Y,
            hearts=player:GetHearts(),soul=player:GetSoulHearts(),max_hearts=player:GetMaxHearts(),
            coins=player:GetNumCoins(),keys=player:GetNumKeys(),bombs=player:GetNumBombs(),
            damage=player.Damage,speed=player.MoveSpeed,fire_delay=player.MaxFireDelay,
            shot_speed=player.ShotSpeed,tear_range=player.TearRange,
            charge=player:GetActiveCharge(),dead=player:IsDead(),items=player:GetCollectibleCount()},
        room={id=roomId,type=room:GetType(),clear=isClear,enemies=enemies,
            bounds={tl.X,tl.Y,br.X,br.Y},visits=visits[tostring(roomId)] or 0},
        entities=entities,doors=doors,grid=grid,
        combat_schema="combat_v1",combat_entities=combat,
        combat_player={size=player.Size,size_multi={player.SizeMulti.X,player.SizeMulti.Y},
            fire_cooldown=player.FireDelay,damage_cooldown_render_frames=player:GetDamageCooldown(),
            invincible=player:HasInvincibility() or player:GetDamageCooldown()>0,
            invincibility_effect=player:HasInvincibility(),can_shoot=player:CanShoot(),can_fly=player.CanFly,
            collision=player.EntityCollisionClass},
        events={damage_taken=damageTaken,damage_dealt=damageDealt,damage_attempted=damageAttempted,kills=kills},
        visited=visits,cleared=cleared,boss_seen=bossSeen,boss_defeated=bossDefeated,
        success=bossDefeated,terminal=player:IsDead() or bossDefeated,
        mode="full_floor",bridge_version="0.1.4",bridge_port=PORT,damage_signal="hp_delta_v1"
    }
end

local function reply()
    sequence = sequence + 1
    return send(snapshot())
end

local function setAction(a)
    previousAction = action
    action = {}
    local move = moves[(a[1] or 0)+1] or moves[1]
    action[ButtonAction.ACTION_LEFT] = move[1] < 0
    action[ButtonAction.ACTION_RIGHT] = move[1] > 0
    action[ButtonAction.ACTION_UP] = move[2] < 0
    action[ButtonAction.ACTION_DOWN] = move[2] > 0
    local shoot = a[2] or 0
    action[ButtonAction.ACTION_SHOOTLEFT] = shoot == 1
    action[ButtonAction.ACTION_SHOOTRIGHT] = shoot == 2
    action[ButtonAction.ACTION_SHOOTUP] = shoot == 3
    action[ButtonAction.ACTION_SHOOTDOWN] = shoot == 4
    action[ButtonAction.ACTION_BOMB] = a[3] == 1
    action[ButtonAction.ACTION_ITEM] = a[3] == 2
    action[ButtonAction.ACTION_PILLCARD] = a[3] == 3
end

local function receiveCommand()
    -- Blocking at a step boundary freezes simulation during inference/optimization.
    -- A finite timeout restores manual control if the Python process disappears.
    client:settimeout(120)
    local line, err = client:receive("*l")
    if not line then disconnect(err); return end
    local decoded, cmd = pcall(json.decode, line)
    if not decoded or type(cmd) ~= "table" then disconnect("invalid JSON"); return end
    responseId = cmd.id or 0
    if cmd.op == "step" then
        setAction(cmd.action or {})
        remaining = math.max(1, math.min(30, tonumber(cmd.frames) or 4))
        pending = "step"
    elseif cmd.op == "reset" then
        action, previousAction = {}, {}
        resetting, pending = true, "reset"
        resetFromEpisode = episode
        -- 'restart 0' starts a normal Isaac run with a new random seed.
        -- Seeded resets are used for held-out evaluation; seed does not grant items.
        if cmd.seed and type(cmd.seed) == "string" and cmd.seed:match("^[A-Z0-9 ]+$") and #cmd.seed <= 9 then
            Isaac.ExecuteCommand("seed " .. cmd.seed)
        else
            game:GetSeeds():SetStartSeed("")
            Isaac.ExecuteCommand("restart 0")
        end
    elseif cmd.op == "close" then
        send({type="closed",id=responseId})
        disconnect("client closed")
    elseif cmd.op == "ping" then
        reply()
        receiveCommand()
    else
        disconnect("unsupported operation")
    end
end

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, button)
    if not active or not entity or not entity:ToPlayer() then return end
    if button < 0 or button > ButtonAction.ACTION_DROP then return end
    local pressed = action[button] == true
    if hook == InputHook.GET_ACTION_VALUE then return pressed and 1.0 or 0.0 end
    if hook == InputHook.IS_ACTION_TRIGGERED then
        return pressed and not previousAction[button]
    end
    return pressed
end)

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function()
    episode = episode + 1
    episodeFrame, damageTaken, damageDealt, kills = 0,0,0,0
    enemyHealth,damageAttempted = {},0
    bossSeen, bossDefeated, visits, cleared, cellVisits = false,false,{},{},{}
    visits[tostring(game:GetLevel():GetCurrentRoomIndex())] = 1
    action, previousAction = {},{}
    remaining = 0
    -- Keep the connection and pending reset across a game restart.
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    local key = tostring(game:GetLevel():GetCurrentRoomIndex())
    visits[key] = (visits[key] or 0) + 1
end)

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount)
    if not active then return end
    if entity:ToPlayer() then
        damageTaken = damageTaken + amount
    elseif entity:IsActiveEnemy(false) and entity:IsVulnerableEnemy() and
        not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
        -- Diagnostic-only legacy attempted-hit signal; never paid as reward.
        damageAttempted = damageAttempted + math.min(amount,math.max(0,entity.HitPoints))
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, entity)
    -- This callback runs AFTER death. includeDead=false filters out every kill.
    if active and entity:IsActiveEnemy(true) and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
        observeEnemyHealth(entity,0)
        kills = kills + 1
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    observeDamage()
    episodeFrame = episodeFrame + 1
    previousAction = action
    if not client then
        if socket.gettime() - lastConnect < 1 then return end
        lastConnect = socket.gettime()
        local candidate = socket.tcp()
        candidate:settimeout(0.05)
        local connected = candidate:connect("127.0.0.1", PORT)
        if not connected then candidate:close(); return end
        candidate:setoption("tcp-nodelay", true)
        client, active = candidate, true
        if reply() then receiveCommand() end
        return
    end
    if resetting and episode > resetFromEpisode then
        -- Wait until post-update so all player/room initialization is finished.
        resetting, pending = false,nil
        if reply() then receiveCommand() end
        return
    end
    if pending == "step" then
        remaining = remaining - 1
        if remaining <= 0 or Isaac.GetPlayer(0):IsDead() then
            pending = nil
            if reply() then receiveCommand() end
        end
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    Isaac.RenderText("Isaac RL :" .. PORT .. " | " .. (active and ("episode " .. episode) or "waiting for Python"), 60, 18, 1,1,1,1)
end)

mod:AddCallback(ModCallbacks.MC_PRE_MOD_UNLOAD, function() disconnect("mod unloaded") end)
Isaac.DebugString("ISAAC_RL: bridge loaded; localhost port " .. PORT)
