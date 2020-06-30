import logging
import random

import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction, Ship
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force, ForceCombination, ContributingForce
from src.planner import Planner
from src.strategies import Strategy


class Commander:
    def __init__(self, configuration: Configuration):
        GameState.config = configuration
        self.planner = Planner()
        self.strategy: Strategy = None

    def get_next_actions(self):
        self.strategy = Strategy()
        self.planner = Planner()
        self._decide_priorities()
        for ship in GameState.board.current_player.ships:
            self._make_action(ship)
        self.planner.resolve_collisions()
        logging.info("Actions after collision resolution")
        for ship in GameState.board.current_player.ships:
            logging.info(f"{ship}: {ship.next_action}")
        self._build_ships()
        return GameState.board.current_player.next_actions

    def _decide_priorities(self):
        SHIP_TO_BASE_RATIO = 5
        if len(GameState.board.current_player.shipyards) == 0:
            self.strategy.expand.priority = 100
        else:
            if len(GameState.board.current_player.ships) / len(GameState.board.current_player.shipyards) > SHIP_TO_BASE_RATIO:
                self.strategy.expand.priority = 5
            else:
                self.strategy.expand.priority = 0
        if self._turns_remaining() < 100:
            self.strategy.return_halite.priority = 50
            self.strategy.expand.priority = 0

    def _make_action(self, ship):
        FORCE_CUTOFF = 1
        ship_pos = ship.position.norm
        logging.info(f"--------- {ship}")
        logging.info(f"cell halite {ship.cell.halite}, ship halite {ship.halite}")
        combination = ForceCombination()

        # Avoid colliding
        '''friendlies_map = self._calc_friendlies_map(ship)
        Strategies.avoid_friendlies_dict[ship.id] = friendlies_map
        avoid_clustering = ContributingForce(
            'avoid friendlies',
            friendlies_map.flow_at(ship_pos),
            0.2
        )'''

        if ship.cell.halite > 250:
            self.strategy.mine_halite.priority *= 2
        mine = self.strategy.mine_halite.get_force(ship_pos, 'mine')
        expand = self.strategy.expand.get_force(ship_pos, 'expand')
        mine_longterm = self.strategy.mine_halite_longterm.get_force(ship_pos, 'mine longterm')
        return_halite = self.strategy.return_halite.get_force(ship_pos, 'return halite')
        attack_enemy_miners = self.strategy.attack_enemy_miners.get_force(ship_pos, 'attack enemy miners')
        attack_enemy_bases = self.strategy.attack_enemy_bases.get_force(ship_pos, 'attack enemy bases')

        RETURN_TRESHOLD = 350
        # TODO separate phases?
        if ship.halite == 0 and len(GameState.board.current_player.ships) > 5 and self._ship_is_attacker(ship):
            if ship.halite == 0:
                combination.add(attack_enemy_miners)
            else:
                combination.add(return_halite)
            total = combination.total()
            logging.info(f"TOTAL            {total}")
            ship.next_action = action_from_force(total, 0)
        elif ship.halite == 0 and len(GameState.board.current_player.ships) > 10 and self._ship_is_kamikaze(ship):
            combination.add(attack_enemy_bases)
            total = combination.total()
            logging.info(f"TOTAL            {total}")
            ship.next_action = action_from_force(total, 0)
        else:
            if expand.weight > 0 and ship.halite == 0:
                combination.add(expand)
            if ship.cell.halite < 200 and self.strategy.mine_halite_longterm.value_at(ship_pos) < 10:
                # combination.add(avoid_clustering) # TODO
                combination.add(mine_longterm)
            elif ship.halite > RETURN_TRESHOLD:
                combination.add(return_halite)
            else:
                combination.add(mine)
            total = combination.total()
            logging.info(f"TOTAL            {total}")
            ship.next_action = action_from_force(total, FORCE_CUTOFF)

            if not ship.next_action\
                    and self.strategy.expand.priority >= 5\
                    and self._can_build_base()\
                    and self.strategy.expand.value_at(ship_pos) > 0:
                ship.next_action = ShipAction.CONVERT
                self.strategy.expand.priority = 0

            MIN_HALITE_TO_STAND = 75
            if ship.cell.halite < MIN_HALITE_TO_STAND:
                if not ship.next_action:
                    logging.info("Moving to any flow direction")
                    ship.next_action = action_from_force(total, 0)
                if not ship.next_action:
                    logging.info("Moving randomly")
                    ship.next_action = self._random_action()

        logging.info(f'Action {ship.next_action}')
        self.planner.reserve_action(ship, ship.next_action)

    def _build_ships(self):
        for base in GameState.board.current_player.shipyards:
            STOP_BUILDING_SHIPS_TURN = 150
            if self._can_build_ship() and self.strategy.expand.priority == 0 and self._turns_remaining() > STOP_BUILDING_SHIPS_TURN:
                if not self.planner.at(base.position.norm):
                    base.next_action = ShipyardAction.SPAWN

    @staticmethod
    def _turns_remaining():
        return GameState.config.episode_steps - GameState.board.step

    @staticmethod
    def _can_build_base():
        return GameState.board.current_player.halite >= GameState.config.convert_cost

    @staticmethod
    def _can_build_ship():
        return GameState.board.current_player.halite >= GameState.config.spawn_cost

    def _calc_friendlies_map(self, ship: Ship) -> AttractionMap:
        friendly_ships_map = np.zeros(GameState.map_size())
        for other_ship in GameState.board.current_player.ships:
            if other_ship.id != ship.id:
                friendly_ships_map[other_ship.position.norm] = -250
        friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=1, mode='wrap')
        friendlies_map = AttractionMap(friendly_ships_map)
        self.strategy.avoid_friendlies_dict[ship.id] = friendlies_map
        return friendlies_map

    def _random_action(self):
        return random.choice((ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST))

    def _ship_is_attacker(self, ship: Ship):
        ATTACKER_RATIO = 3
        ship_id, _ = ship.id.split('-')
        ship_id = int(ship_id)
        return ship_id % ATTACKER_RATIO == 0

    def _ship_is_kamikaze(self, ship):
        KAMIKAZE_RATIO = 5
        ship_id, _ = ship.id.split('-')
        ship_id = int(ship_id)
        return ship_id % KAMIKAZE_RATIO == 0
