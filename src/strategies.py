import numpy as np
from kaggle_environments.envs.halite.helpers import Ship
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.maps import AttractionMap, visit_map


class Strategy:
    def __init__(self):
        self.mine_halite = make_halite_map()
        self.mine_halite_longterm = make_halite_longterm_map(self.mine_halite)
        self.return_halite = make_return_halite_map()
        self.expand = make_expand_map(self.mine_halite)
        self.avoid_friendlies = {}
        self.attack_enemy_miners = make_attack_enemy_miners_map()
        self.attack_enemy_bases = make_attack_enemy_bases_map()
        # avoid_enemies


def make_halite_map():
    halite_map = np.array(GameState.halite).reshape(GameState.map_size())
    halite_map = gaussian_filter(halite_map, sigma=0.4, mode='wrap') / 5
    return AttractionMap(halite_map)


def make_halite_longterm_map(halite_map: AttractionMap):
    mine_halite_longterm = (np.copy(halite_map.array) - 5) * 5
    mine_halite_longterm = gaussian_filter(mine_halite_longterm, sigma=1.7, mode='wrap')
    return AttractionMap(mine_halite_longterm)


def make_return_halite_map():
    return_halite = np.empty(GameState.map_size())
    visit_map(
        return_halite,
        (s.position.norm for s in GameState.board.current_player.shipyards),
        100,
        lambda x: x * 0.85
    )
    return AttractionMap(return_halite)


def make_expand_map(halite_map: AttractionMap):
    expansion_map = np.copy(halite_map.array)
    for base in GameState.board.shipyards.values():
        expansion_map[base.position.norm] = -500
    expansion_map = gaussian_filter(expansion_map, sigma=2.5, mode='wrap') * 5
    expansion_map -= halite_map.array
    return AttractionMap(expansion_map)


def make_attack_enemy_miners_map():
    me = GameState.board.current_player_id
    attack_enemy_miners = np.empty(GameState.map_size())
    visit_map(
        attack_enemy_miners,
        (s.position.norm
         for s in GameState.board.ships.values()
         if s.player_id != me and s.halite > 100),
        100,
        lambda x: max(0, x - 10)
    )
    return AttractionMap(attack_enemy_miners)


def make_attack_enemy_bases_map():
    me = GameState.board.current_player_id
    attack_enemy_bases = np.empty(GameState.map_size())
    visit_map(
        attack_enemy_bases,
        (s.position.norm
         for s in GameState.board.shipyards.values()
         if s.player_id != me),
        100,
        lambda x: max(0, x - 10)
    )
    return AttractionMap(attack_enemy_bases)


def make_friendlies_map(ship: Ship) -> AttractionMap:
    friendly_ships_map = np.zeros(GameState.map_size())
    for other_ship in GameState.board.current_player.ships:
        if other_ship.id != ship.id:
            friendly_ships_map[other_ship.position.norm] = -250
    friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=1, mode='wrap')
    return AttractionMap(friendly_ships_map)

'''
threat_map = np.zeros(GameState.map_size())
for ship in GameState.board.ships.values():
    if ship.player_id != GameState.board.current_player_id:
        threat_map[ship.position.norm] = -600
threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
Strategies.avoid_enemies = AttractionMap(threat_map)
'''

'''
# TODO not used
friendly_ships_map = np.zeros(GameState.map_size())
for ship in GameState.board.current_player.ships:
    friendly_ships_map[ship.position.norm] = -100
#friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=0.8, mode='wrap')
Strategies.avoid_friendlies = AttractionMap(friendly_ships_map)

# TODO not used
friendly_bases = np.zeros(GameState.map_size())
for base in GameState.board.current_player.shipyards:
    friendly_bases[base.position.norm] = -180
friendly_bases = gaussian_filter(friendly_bases, sigma=0.5, mode='wrap')
Strategies.friendly_bases = AttractionMap(friendly_bases)
'''
