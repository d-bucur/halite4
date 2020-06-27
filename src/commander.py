import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction
from scipy.ndimage import gaussian_filter

from src.coordinates import P
from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force


class Strategies:
    mine_halite: AttractionMap = None
    return_halite: AttractionMap = None
    expand: AttractionMap = None
    avoid_enemies: AttractionMap = None
    avoid_friendlies: AttractionMap = None
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
        halite_map = gaussian_filter(halite_map, sigma=0.1, mode='wrap') / 5
        Strategies.mine_halite = AttractionMap(halite_map)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -5000
        expansion_map = gaussian_filter(expansion_map, sigma=3, mode='wrap')
        Strategies.expand = AttractionMap(expansion_map)

        threat_map = np.zeros(GameState.map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                threat_map[ship.position.norm] = -600
        threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
        Strategies.avoid_enemies = AttractionMap(threat_map)

        friendly_ships_map = np.zeros(GameState.map_size())
        for ship in GameState.board.current_player.ships:
            friendly_ships_map[ship.position.norm] = -300
        friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=0.8, mode='wrap')
        Strategies.avoid_friendlies = AttractionMap(friendly_ships_map)

        friendly_bases = np.zeros(GameState.map_size())
        for base in GameState.board.current_player.shipyards:
            friendly_bases[base.position.norm] = -400
        friendly_bases = gaussian_filter(friendly_bases, sigma=0.8, mode='wrap')
        Strategies.friendly_bases = AttractionMap(friendly_bases)

        return_halite = np.zeros(GameState.map_size())
        for base in GameState.board.current_player.shipyards:
            return_halite[base.position.norm] = 3000
        return_halite = gaussian_filter(return_halite, sigma=3, mode='wrap')
        Strategies.return_halite = AttractionMap(return_halite)


class Commander:
    def __init__(self, configuration: Configuration):
        GameState.config = configuration

    def get_next_actions(self):
        Strategies.update()
        self._decide_priorities()
        for ship in GameState.board.current_player.ships:
            self._make_action(ship)
        self._build_ships()
        return GameState.board.current_player.next_actions

    def _decide_priorities(self):
        if len(GameState.board.current_player.shipyards) == 0:
            Strategies.expand.priority = 100
        else:
            Strategies.expand.priority = 0
        Strategies.mine_halite.priority = 1
        Strategies.friendly_bases.priority = 0.3
        Strategies.avoid_friendlies.priority = 1
        Strategies.avoid_enemies.priority = 1

    def _make_action(self, ship):
        FORCE_CUTOFF = 5
        total_force = P(0, 0)
        total_priorities = 0
        ship_pos = ship.position.norm
        print("--", ship)
        print(f"halite on cell: {ship.cell.halite}")

        def debug_force(name: str, f: P, prio: float):
            print(f"{name} {f/prio} * {prio} = {f}")

        def add_force(name: str, f: P, prio: float):
            nonlocal total_priorities, total_force
            print(f"{name} {f} * {prio} = {f*prio}")
            total_priorities += prio
            total_force += f * prio

        f = Strategies.expand.at(ship_pos) * Strategies.expand.priority
        debug_force('expand', f, Strategies.expand.priority)
        total_priorities += Strategies.expand.priority
        total_force += f

        MAX_HALITE_X_SHIP = 500
        mine_priority = Strategies.mine_halite.priority * (MAX_HALITE_X_SHIP - ship.halite) / MAX_HALITE_X_SHIP
        f = Strategies.mine_halite.at(ship_pos) * mine_priority
        debug_force('mine', f, mine_priority)
        total_priorities += mine_priority
        total_force += f

        BOOST_RETURN_HALITE = 5
        return_halite_priority = Strategies.return_halite.priority * (ship.halite / MAX_HALITE_X_SHIP) * BOOST_RETURN_HALITE
        f = Strategies.mine_halite.at(ship_pos)
        print(f"return halite {f} * {return_halite_priority} = {f*return_halite_priority}")
        f = f * return_halite_priority
        total_priorities += return_halite_priority
        total_force += f

        friendly_base_priority = Strategies.friendly_bases.priority
        #if ship.halite > 0:
        #    friendly_base_priority = -friendly_base_priority
        f = Strategies.friendly_bases.at(ship_pos) * friendly_base_priority
        debug_force('friendly base', f, friendly_base_priority)
        total_priorities += friendly_base_priority
        total_force += f

        f = Strategies.avoid_friendlies.at(ship_pos, False) * Strategies.avoid_friendlies.priority
        debug_force('avoid friendlies', f, Strategies.avoid_friendlies.priority)
        total_priorities += Strategies.avoid_friendlies.priority
        total_force += f

        direction = total_force / total_priorities
        debug_force('TOTAL', direction, 1)
        ship.next_action = action_from_force(direction, FORCE_CUTOFF)

        if not ship.next_action and Strategies.expand.priority > 5 and self._can_build_base():
            ship.next_action = ShipAction.CONVERT

    def _build_ships(self):
        for base in GameState.board.current_player.shipyards:
            if self._can_build_ship():
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
