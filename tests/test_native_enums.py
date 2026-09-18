from pathlib import Path
import re

import pytest

from isaac_rl.observation import PLAYER_SOLID_COLLISIONS, TERRAIN_HAZARDS


def test_terrain_constants_match_this_installed_games_own_lua_enums():
    source = Path(__file__).resolve().parents[1]/"runtime/game/resources/scripts/enums.lua"
    if not source.exists():
        pytest.skip("Optional installed-game contract test; portable encoder tests run separately")
    text = source.read_text(encoding="utf-8-sig")
    def value(name):
        match = re.search(r"\b"+name+r"\s*=\s*(\d+)",text)
        assert match, f"Installed enum missing: {name}"
        return int(match.group(1))
    assert TERRAIN_HAZARDS == {value("GRID_SPIKES"),value("GRID_SPIKES_ONOFF"),value("GRID_TNT")}
    assert PLAYER_SOLID_COLLISIONS == {value("COLLISION_OBJECT"),value("COLLISION_SOLID"),value("COLLISION_WALL")}
    assert value("COLLISION_WALL_EXCEPT_PLAYER") not in PLAYER_SOLID_COLLISIONS
    assert value("GRID_WALL") not in TERRAIN_HAZARDS
    assert value("GRID_POOP") not in TERRAIN_HAZARDS
