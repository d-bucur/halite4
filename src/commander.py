import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force


class Strategies:
    halite_map: AttractionMap = None
    expansion_map: AttractionMap = None
    threat_map: AttractionMap = None
    reward_map: AttractionMap = None

    @classmethod
    def __iter__(cls):
        yield cls.expansion_map
        yield cls.halite_map


class Commander:
    def __init__(self, configuration: Configuration):
        GameState.config = configuration

    def get_next_actions(self):
        self._calc_maps()
        self._decide_priorities()
        self._make_actions()
        self._build_ships()
        return GameState.board.current_player.next_actions

    def _calc_maps(self) -> None:
        halite_map = np.array(GameState.halite).reshape(self._map_size())
        halite_map = gaussian_filter(halite_map, sigma=0.7, mode='wrap')
        Strategies.halite_map = AttractionMap(halite_map)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -5000
        expansion_map = gaussian_filter(expansion_map, sigma=3, mode='wrap')
        Strategies.expansion_map = AttractionMap(expansion_map)

        threat_map = np.zeros(self._map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                threat_map[ship.position.norm] = -5000
        threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
        Strategies.threat_map = AttractionMap(threat_map)

        reward_map = halite_map + threat_map
        Strategies.reward_map = AttractionMap(reward_map)

    def _decide_priorities(self):
        if len(GameState.board.current_player.shipyards) == 0:
            Strategies.expansion_map.priority = 10

    def _make_actions(self):
        for ship in GameState.board.current_player.ships:
            direction = Strategies.expansion_map.flow.at(ship.position.norm)
            print(f"{ship} force {direction}")
            ship.next_action = action_from_force(direction, 0.5)
            if not ship.next_action and Strategies.expansion_map.priority > 5 and self._can_build_base():
                ship.next_action = ShipAction.CONVERT

    def _build_ships(self):
        for base in GameState.board.current_player.shipyards:
            if self._can_build_ship():
                base.next_action = ShipyardAction.SPAWN

    @staticmethod
    def _turns_remaining():
        return GameState.config.episode_steps - GameState.board.step

    @staticmethod
    def _map_size():
        return GameState.config.size, GameState.config.size

    @staticmethod
    def _can_build_base():
        return GameState.board.current_player.halite >= GameState.config.convert_cost

    @staticmethod
    def _can_build_ship():
        return GameState.board.current_player.halite >= GameState.config.spawn_cost
