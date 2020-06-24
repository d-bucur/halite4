from collections import defaultdict
from typing import List, Dict, Set

import numpy as np
from kaggle_environments.envs.halite.helpers import Board, ShipyardAction, Ship, ShipAction, Configuration
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.helpers import calc_highest_in_range
from src.orders import ShipOrder, BuildShipyardOrder, HarvestOrder
from src.pathing import PathPlanner


class Commander:

    def __init__(self, configuration: Configuration):
        self.config = configuration
        GameState.planner = PathPlanner(configuration.size)

        self.orders: Dict[str, ShipOrder] = {}
        self.harvesters_x_base: Dict[str, Set[str]] = defaultdict(lambda: set())
        self.need_shipyard: int = 0

        self.halite_map: np.ndarray = None
        self.expansion_map: np.ndarray = None
        self.threat_map: np.ndarray = None
        self.reward_map: np.ndarray = None
        self.objectives_map: np.ndarray = None
        
    def get_next_actions(self):
        self.clear_state()
        self.calc_maps()

        ships_without_orders = [ship for ship in GameState.board.current_player.ships if ship.id not in self.orders]

        free_bases = self.expansion_phase(ships_without_orders)
        self.assign_harvesters(ships_without_orders, free_bases)

        self.build_ship_actions()
        self.build_shipyard_actions()
        self.collision_resolution()
        
        return GameState.board.current_player.next_actions

    def clear_state(self):
        # remove done orders or from invalid ships
        for ship_id in list(self.orders.keys()):
            if ship_id not in GameState.board.ships or self.orders[ship_id].is_done(GameState.board.ships[ship_id]):
                del self.orders[ship_id]
        # purge harvesters_x_base or obsolete keys
        for base_id, ships in self.harvesters_x_base.items():
            if base_id not in GameState.board.shipyards:
                del self.harvesters_x_base[base_id]
            self.harvesters_x_base[base_id] = set(s for s in ships if s in GameState.board.ships)

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

        self.objectives_map = np.zeros(self._map_size())
        for ship_id, order in self.orders.items():
            ship_target = order.next_target()
            self.objectives_map[ship_target] = -1000
        self.objectives_map = gaussian_filter(self.objectives_map, sigma=1, mode='wrap')

        self.reward_map = self.halite_map + self.threat_map + 3 * self.objectives_map

    def expansion_phase(self, ships_without_orders: List[Ship]) -> List[str]:
        HARVESTERS_PER_BASE = 4
        EXPANSION_RAGE = 6
        BASE_EXPANSION_CUTOFF = 200

        free_bases = []
        for base in GameState.board.current_player.shipyards:
            ships = self.harvesters_x_base[base.id]
            if len(ships) < HARVESTERS_PER_BASE:
                free_bases.append(base.id)
        if self._turns_remaining() < BASE_EXPANSION_CUTOFF:
            return free_bases

        can_afford_expansion = GameState.board.current_player.halite > self.config.convert_cost

        if can_afford_expansion:
            if len(free_bases) == 0:
                if all(type(o) != BuildShipyardOrder for o in self.orders.values()):
                    self.need_shipyard += 1

            while self.need_shipyard > 0 and ships_without_orders:
                ship = ships_without_orders.pop()
                target = calc_highest_in_range(self.expansion_map, ship.position.norm, EXPANSION_RAGE, 1)[0]
                self.orders[ship.id] = BuildShipyardOrder(ship, target)
                self.need_shipyard -= 1

        return free_bases

    def assign_harvesters(self, ships: List[Ship], free_bases: List[str]):
        if not ships or not free_bases or not GameState.board.current_player.shipyards:
            return
        HARVESTER_RANGE = 7
        base = GameState.board.shipyards[free_bases[0]]  # TODO not using all bases
        base_pos = base.position.norm
        highest = calc_highest_in_range(self.reward_map, base_pos, HARVESTER_RANGE, len(ships))
        for ship, target in zip(ships, highest):
            self.harvesters_x_base[base.id].add(ship.id)
            self.orders[ship.id] = HarvestOrder(ship, target, base_pos)

    def build_ship_actions(self) -> None:
        for ship in GameState.board.current_player.ships:
            if ship.id in self.orders:
                ship.next_action = self.orders[ship.id].execute(ship)

    def build_shipyard_actions(self) -> None:
        NEW_SHIPS_CUTOFF = 150
        if self._turns_remaining() < NEW_SHIPS_CUTOFF:
            return
        affordable_ships = GameState.board.current_player.halite // self.config.spawn_cost
        for shipyard in GameState.board.current_player.shipyards:
            if affordable_ships > 0 and shipyard.cell.ship is None:
                shipyard.next_action = ShipyardAction.SPAWN
                affordable_ships -= 1

    def _turns_remaining(self):
        return self.config.episode_steps - GameState.board.step

    def _map_size(self):
        return self.config.size, self.config.size

    def collision_resolution(self):
        # if will be dead next turn, stop, iterate a maximum amount of times
        max_simulations = 10
        has_collisions = True
        while has_collisions and max_simulations > 0:
            next_board = GameState.board.next()
            max_simulations -= 1
            for ship in GameState.board.current_player.ships:
                if ship.id not in next_board.ships and ship.next_action != ShipAction.CONVERT and ship.cell.shipyard is None:
                    ship.next_action = None
                    break
            has_collisions = False
