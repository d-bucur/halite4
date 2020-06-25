import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState


class Commander:

    def __init__(self, configuration: Configuration):
        GameState.config = configuration

        self.halite_map: np.ndarray = None
        self.expansion_map: np.ndarray = None
        self.threat_map: np.ndarray = None
        self.reward_map: np.ndarray = None
        
    def get_next_actions(self):
        self.calc_maps()
        return GameState.board.current_player.next_actions

    def calc_maps(self) -> None:
        self.halite_map = np.array(GameState.halite).reshape(self._map_size())

        self.expansion_map = np.copy(self.halite_map)
        for base in GameState.board.shipyards.values():
            self.expansion_map[base.position.norm] = -5000
        self.expansion_map = gaussian_filter(self.expansion_map, sigma=3, mode='wrap')

        self.threat_map = np.zeros(self._map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                self.threat_map[ship.position.norm] = -5000
        self.threat_map = gaussian_filter(self.threat_map, sigma=1.2, mode='wrap')

        self.reward_map = self.halite_map + self.threat_map

    @staticmethod
    def _turns_remaining():
        return GameState.config.episode_steps - GameState.board.step

    @staticmethod
    def _map_size():
        return GameState.config.size, GameState.config.size
