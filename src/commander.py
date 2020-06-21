from typing import List, Dict

import numpy as np
from kaggle_environments.envs.halite.helpers import Board, ShipyardAction
from scipy.ndimage import gaussian_filter

from src.helpers import calc_highest_in_range
from src.orders import ShipOrder, BuildShipyardOrder, HarvestOrder


class Commander:
    def __init__(self):
        self.board: Board = None
        self.halite_map: np.ndarray = None
        self.halite_map_blurred: np.ndarray = None
        self.threat_map: np.ndarray = None
        self.reward_map: np.ndarray = None
        self.objectives_map: np.ndarray = None
        self.orders: Dict[str, ShipOrder] = {}
        self.halite: List = None
        self.need_shipyard: int = 1
    
    def update(self, board: Board, raw_observation):
        self.board = board
        self.halite = raw_observation.halite
        
    def get_next_actions(self):
        self.clear_done_orders()
        self.calc_maps()
        me = self.board.current_player

        ships_without_orders = [ship for ship in me.ships if ship.id not in self.orders]

        while self.need_shipyard > 0:
            ship = ships_without_orders.pop()
            target = calc_highest_in_range(self.reward_map, ship.position.norm, 4, 1)[0]
            self.orders[ship.id] = BuildShipyardOrder(target)
            self.need_shipyard -= 1

        self.assign_harvesters(ships_without_orders)

        self.build_ship_actions()
        self.build_shipyard_actions()
        
        return self.board.current_player.next_actions

    def build_ship_actions(self) -> None:
        for ship in self.board.current_player.ships:
            if ship.id in self.orders:
                ship.next_action = self.orders[ship.id].execute(ship)

    def clear_done_orders(self):
        for ship_id in list(self.orders.keys()):
            if ship_id not in self.board.ships or self.orders[ship_id].is_done(self.board.ships[ship_id]):
                del self.orders[ship_id]

    def build_shipyard_actions(self) -> None:
        affordable_ships = self.board.current_player.halite // self.board.configuration.spawn_cost
        for shipyard in self.board.current_player.shipyards:
            if affordable_ships > 0:
                shipyard.next_action = ShipyardAction.SPAWN
                affordable_ships -= 1

    def calc_maps(self) -> None:
        self.halite_map = np.array(self.halite).reshape(self._map_size())
        self.halite_map_blurred = gaussian_filter(self.halite_map, sigma=1, mode='wrap')

        self.threat_map = np.zeros(self._map_size())
        for ship in self.board.ships.values():
            if ship.player_id != self.board.current_player_id:
                self.threat_map[ship.position.norm] = -5000
        self.threat_map = gaussian_filter(self.threat_map, sigma=1.2, mode='wrap')

        self.objectives_map = np.zeros(self._map_size())
        for ship_id, order in self.orders.items():
            ship_target = order.next_target()
            self.objectives_map[ship_target] = -1000
        self.objectives_map = gaussian_filter(self.objectives_map, sigma=1, mode='wrap')

        self.reward_map = 3 * self.halite_map_blurred + self.threat_map + 3 * self.objectives_map

    def _map_size(self):
        return self.board.configuration.size, self.board.configuration.size

    def assign_harvesters(self, ships):
        if not ships or not self.board.current_player.shipyards:
            return
        base = self.board.current_player.shipyards[0]
        base_pos = base.position.norm
        highest = calc_highest_in_range(self.reward_map, base_pos, 7, len(ships))
        for ship, target in zip(ships, highest):
            self.orders[ship.id] = HarvestOrder(target, base_pos, self.board)
