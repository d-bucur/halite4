import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.maps import AttractionMap


class Commander:
    def __init__(self, configuration: Configuration):
        GameState.config = configuration

        self.halite_map: AttractionMap = None
        self.expansion_map: AttractionMap = None
        self.threat_map: AttractionMap = None
        self.reward_map: AttractionMap = None
        
    def get_next_actions(self):
        self.calc_maps()
        return GameState.board.current_player.next_actions

    def calc_maps(self) -> None:
        halite_map = np.array(GameState.halite).reshape(self._map_size())
        self.halite_map = AttractionMap(halite_map)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -5000
        expansion_map = gaussian_filter(expansion_map, sigma=3, mode='wrap')
        self.expansion_map = AttractionMap(expansion_map)

        threat_map = np.zeros(self._map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                threat_map[ship.position.norm] = -5000
        threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
        self.threat_map = AttractionMap(threat_map)

        reward_map = halite_map + threat_map
        self.halite_map = AttractionMap(reward_map)

    @staticmethod
    def _turns_remaining():
        return GameState.config.episode_steps - GameState.board.step

    @staticmethod
    def _map_size():
        return GameState.config.size, GameState.config.size
