import logging
import random
from typing import Dict

import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction, Ship
from scipy.ndimage import gaussian_filter

from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force, ForceCombination, ContributingForce, visit_map
from src.planner import Planner


class Strategies:
    mine_halite: AttractionMap = None
    mine_halite_longterm: AttractionMap = None
    return_halite: AttractionMap = None
    expand: AttractionMap = None
    avoid_enemies: AttractionMap = None
    avoid_friendlies: AttractionMap = None
    avoid_friendlies_dict: Dict[str, AttractionMap] = {}
    friendly_bases: AttractionMap = None
    attack_enemy_miners: AttractionMap = None
    attack_enemy_bases: AttractionMap = None

    @classmethod
    def update(cls):
        halite_map = np.array(GameState.halite).reshape(GameState.map_size())
        halite_map = gaussian_filter(halite_map, sigma=0.4, mode='wrap') / 5
        Strategies.mine_halite = AttractionMap(halite_map)

        mine_halite_longterm = (np.copy(halite_map) - 5) * 5
        mine_halite_longterm = gaussian_filter(mine_halite_longterm, sigma=1.7, mode='wrap')
        Strategies.mine_halite_longterm = AttractionMap(mine_halite_longterm)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -500
        expansion_map = gaussian_filter(expansion_map, sigma=2.5, mode='wrap') * 5
        expansion_map -= halite_map
        Strategies.expand = AttractionMap(expansion_map)

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

        me = GameState.board.current_player_id
        attack_enemy_miners = np.empty(GameState.map_size())
        visit_map(
            attack_enemy_miners,
            (s.position.norm
             for s in GameState.board.ships.values()
             if s.player_id != me and s.halite > 100),
            100,
            lambda x: max(0, x-10)
        )
        Strategies.attack_enemy_miners = AttractionMap(attack_enemy_miners)

        attack_enemy_bases = np.empty(GameState.map_size())
        visit_map(
            attack_enemy_bases,
            (s.position.norm
             for s in GameState.board.shipyards.values()
             if s.player_id != me),
            100,
            lambda x: max(0, x-10)
        )
        Strategies.attack_enemy_bases = AttractionMap(attack_enemy_bases)

        return_halite = np.empty(GameState.map_size())
        visit_map(
            return_halite,
            (s.position.norm for s in GameState.board.current_player.shipyards),
            100,
            lambda x: x * 0.85
        )
        Strategies.return_halite = AttractionMap(return_halite)


class Commander:
    def __init__(self, configuration: Configuration):
        GameState.config = configuration
        self.planner = Planner()

    def get_next_actions(self):
        Strategies.update()
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
            Strategies.expand.priority = 100
        else:
            if len(GameState.board.current_player.ships) / len(GameState.board.current_player.shipyards) > SHIP_TO_BASE_RATIO:
                Strategies.expand.priority = 5
            else:
                Strategies.expand.priority = 0
        if self._turns_remaining() < 100:
            Strategies.return_halite.priority = 50
            Strategies.expand.priority = 0

    def _make_action(self, ship):
        FORCE_CUTOFF = 1
        ship_pos = ship.position.norm
        logging.info(f"--------- {ship}")
        logging.info(f"cell halite {ship.cell.halite}, ship halite {ship.halite}")
        combination = ForceCombination()

        expand = ContributingForce(
            'expand',
            Strategies.expand.flow_at(ship_pos),
            Strategies.expand.priority
        )

        # Avoid colliding
        friendlies_map = self._calc_friendlies_map(ship)
        Strategies.avoid_friendlies_dict[ship.id] = friendlies_map
        avoid_clustering = ContributingForce(
            'avoid friendlies',
            friendlies_map.flow_at(ship_pos),
            0.2
        )

        mine_priority = Strategies.mine_halite.priority
        if ship.cell.halite > 250:
            mine_priority *= 2
        mine = ContributingForce(
            'mine',
            Strategies.mine_halite.flow_at(ship_pos),
            mine_priority
        )

        mine_longterm = ContributingForce(
            'mine longterm',
            Strategies.mine_halite_longterm.flow_at(ship_pos),
            Strategies.mine_halite_longterm.priority
        )

        RETURN_TRESHOLD = 350
        return_halite = ContributingForce(
            'return halite',
            Strategies.return_halite.flow_at(ship_pos),
            Strategies.return_halite.priority
        )

        attack_enemy_miners = ContributingForce(
            'attack enemy miners',
            Strategies.attack_enemy_miners.flow_at(ship_pos),
            Strategies.attack_enemy_miners.priority
        )

        attack_enemy_bases = ContributingForce(
            'attack enemy bases',
            Strategies.attack_enemy_bases.flow_at(ship_pos),
            Strategies.attack_enemy_bases.priority
        )

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
            if ship.cell.halite < 200 and Strategies.mine_halite_longterm.value_at(ship_pos) < 10:
                combination.add(avoid_clustering)
                combination.add(mine_longterm)
            elif ship.halite > RETURN_TRESHOLD:
                combination.add(return_halite)
            else:
                combination.add(mine)
            total = combination.total()
            logging.info(f"TOTAL            {total}")
            ship.next_action = action_from_force(total, FORCE_CUTOFF)

            if not ship.next_action\
                    and Strategies.expand.priority >= 5\
                    and self._can_build_base()\
                    and Strategies.expand.value_at(ship_pos) > 0:
                ship.next_action = ShipAction.CONVERT
                Strategies.expand.priority = 0

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
            if self._can_build_ship() and Strategies.expand.priority == 0 and self._turns_remaining() > STOP_BUILDING_SHIPS_TURN:
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
        Strategies.avoid_friendlies_dict[ship.id] = friendlies_map
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
