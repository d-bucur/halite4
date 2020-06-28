from typing import Dict
import logging

import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction, Ship
from scipy.ndimage import gaussian_filter

from src.coordinates import P
from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force, ForceCombination, ContributingForce
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

    @classmethod
    def __iter__(cls):
        yield cls.expand
        yield cls.mine_halite
        yield cls.avoid_friendlies
        yield cls.friendly_bases

    @classmethod
    def update(cls):
        halite_map = np.array(GameState.halite).reshape(GameState.map_size())
        halite_map = gaussian_filter(halite_map, sigma=0.2, mode='wrap') / 5
        Strategies.mine_halite = AttractionMap(halite_map)

        mine_halite_longterm = np.copy(halite_map) * 5
        mine_halite_longterm = gaussian_filter(mine_halite_longterm, sigma=2, mode='wrap')
        Strategies.mine_halite_longterm = AttractionMap(mine_halite_longterm)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -400
        expansion_map = gaussian_filter(expansion_map, sigma=2, mode='wrap') * 5
        expansion_map -= halite_map
        Strategies.expand = AttractionMap(expansion_map)

        threat_map = np.zeros(GameState.map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                threat_map[ship.position.norm] = -600
        threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
        Strategies.avoid_enemies = AttractionMap(threat_map)

        # TODO not used
        friendly_ships_map = np.zeros(GameState.map_size())
        for ship in GameState.board.current_player.ships:
            friendly_ships_map[ship.position.norm] = -100
        #friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=0.8, mode='wrap')
        Strategies.avoid_friendlies = AttractionMap(friendly_ships_map)

        friendly_bases = np.zeros(GameState.map_size())
        for base in GameState.board.current_player.shipyards:
            friendly_bases[base.position.norm] = -180
        friendly_bases = gaussian_filter(friendly_bases, sigma=0.5, mode='wrap')
        Strategies.friendly_bases = AttractionMap(friendly_bases)

        return_halite = np.zeros(GameState.map_size())
        for base in GameState.board.current_player.shipyards:
            return_halite[base.position.norm] = 6000
        return_halite = gaussian_filter(return_halite, sigma=3, mode='wrap')
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
        Strategies.mine_halite.priority = 1
        Strategies.mine_halite_longterm.priority = 1
        Strategies.friendly_bases.priority = 0.15
        Strategies.avoid_friendlies.priority = 1
        Strategies.avoid_enemies.priority = 1
        if self._turns_remaining() < 50:
            Strategies.return_halite.priority = 50

    def _make_action(self, ship):
        FORCE_CUTOFF = 5
        ship_pos = ship.position.norm
        logging.info(f"--------- {ship}")
        logging.info(f"cell halite {ship.cell.halite}, ship halite {ship.halite}")
        combination = ForceCombination()

        # Create new bases
        expand = ContributingForce(
            'expand',
            Strategies.expand.at(ship_pos),
            Strategies.expand.priority
        )

        '''
        # Get away from friendly shipyards
        # TODO only add if current action is none
        friendly_base_priority = Strategies.friendly_bases.priority
        #if ship.halite > 0:
        #    friendly_base_priority = -friendly_base_priority
        add_force('friendly base', Strategies.friendly_bases.at(ship_pos), friendly_base_priority)
        '''

        # Avoid colliding into friendlies
        '''
        AVOID_FRIENDLIES_REDUCTION_FACTOR = 5
        avoid_friendlies_force = self._calc_friendlies_map(ship).at(ship_pos, True) / AVOID_FRIENDLIES_REDUCTION_FACTOR
        avoid_friendlies_priority = Strategies.avoid_friendlies.priority
        avoid_friendlies_priority *= min(3, avoid_friendlies_force.magnitude / 3)
        add_force(
            'avoid friendlies',
            avoid_friendlies_force,
            avoid_friendlies_priority
        )
        '''

        # Mine
        MAX_HALITE_X_SHIP = 500
        mine_force = Strategies.mine_halite.at(ship_pos) / 5
        cell_halite_modifier = ship.cell.halite / GameState.config.max_cell_halite * 3 + 1
        carrying_halite_modifier = max(0, (MAX_HALITE_X_SHIP - ship.halite) / MAX_HALITE_X_SHIP)
        mine_priority = Strategies.mine_halite.priority * carrying_halite_modifier * cell_halite_modifier
        mine = ContributingForce(
            'mine',
            mine_force,
            mine_priority
        )

        mine_longterm = ContributingForce(
            'mine longterm',
            Strategies.mine_halite_longterm.at(ship_pos),
            Strategies.mine_halite_longterm.priority * (500 - ship.cell.halite) / 500 * carrying_halite_modifier
        )

        # Return halite
        RETURN_TRESHOLD = 500
        return_halite_priority = Strategies.return_halite.priority * (ship.halite / RETURN_TRESHOLD)
        return_halite = ContributingForce(
            'return halite',
            Strategies.return_halite.at(ship_pos),
            return_halite_priority
        )

        combination.add(expand)
        combination.add(mine)
        combination.add(mine_longterm)
        combination.add(return_halite)
        total = combination.total()
        logging.info(f"TOTAL            {total}")
        ship.next_action = action_from_force(total, FORCE_CUTOFF)

        if not ship.next_action and Strategies.expand.priority >= 5 and self._can_build_base():
            ship.next_action = ShipAction.CONVERT

        MIN_HALITE_TO_STAND = 75
        if not ship.next_action and ship.cell.halite < MIN_HALITE_TO_STAND:
            ship.next_action = action_from_force(total, 0)
            logging.info("Avoided standing still")
        logging.info(f'Action {ship.next_action}')
        self.planner.reserve_action(ship, ship.next_action)

    def _build_ships(self):
        for base in GameState.board.current_player.shipyards:
            if self._can_build_ship() and Strategies.expand.priority == 0:
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
                friendly_ships_map[other_ship.position.norm] = -150
        friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=0.5, mode='wrap')
        friendlies_map = AttractionMap(friendly_ships_map)
        Strategies.avoid_friendlies_dict[ship.id] = friendlies_map
        return friendlies_map
