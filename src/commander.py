from typing import List, Dict

import numpy as np
from kaggle_environments.envs.halite.helpers import ShipAction, Board, ShipyardAction
from scipy.ndimage import gaussian_filter

from src.helpers import calc_highest_in_range
from src.orders import ShipOrder, BuildShipyardOrder


class Commander:
    def __init__(self):
        self.board: Board = None
        self.halite_map: np.ndarray = None
        self.threat_map: np.ndarray = None
        self.reward_map: np.ndarray = None
        self.orders: Dict[str, ShipOrder] = {}
        self.halite: List = None
    
    def update(self, board: Board, raw_observation):
        self.board = board
        self.halite = raw_observation.halite
        
    def get_next_actions(self):
        self.calc_maps()
        me = self.board.current_player

        if len(me.shipyards) == 0:
            first_ship = me.ships[0]
            target = calc_highest_in_range(self.reward_map, first_ship.position.norm, 4)
            if first_ship.id not in self.orders:
                self.orders[first_ship.id] = BuildShipyardOrder(target)

        self.build_ship_actions()
        self.build_shipyard_actions()
        
        return self.board.current_player.next_actions

    def build_ship_actions(self) -> None:
        for ship_id in list(self.orders.keys()):
            if ship_id not in self.board.ships or self.orders[ship_id].is_done(self.board.ships[ship_id]):
                del self.orders[ship_id]
        for ship in self.board.current_player.ships:
            if ship.id in self.orders:
                ship.next_action = self.orders[ship.id].execute(ship)
            else:
                ship.next_action = ShipAction.EAST

    def build_shipyard_actions(self) -> None:
        affordable_ships = self.board.current_player.halite // self.board.configuration.spawn_cost
        for shipyard in self.board.current_player.shipyards:
            if affordable_ships > 0:
                shipyard.next_action = ShipyardAction.SPAWN
                affordable_ships -= 1

    def calc_maps(self) -> None:
        self.halite_map = np.array(self.halite).reshape(self._map_size())
        self.halite_map = gaussian_filter(self.halite_map, sigma=1, mode='wrap')
        self.threat_map = np.zeros(self._map_size())
        for ship in self.board.ships.values():
            if ship.player_id != self.board.current_player_id:
                self.threat_map[ship.position.norm] = -5000
        self.threat_map = gaussian_filter(self.threat_map, sigma=1.2, mode='wrap')
        self.reward_map = self.halite_map * 3 + self.threat_map

    def _map_size(self):
        return self.board.configuration.size, self.board.configuration.size
