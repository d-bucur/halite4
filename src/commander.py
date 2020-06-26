import numpy as np
from kaggle_environments.envs.halite.helpers import Configuration, ShipAction, ShipyardAction
from scipy.ndimage import gaussian_filter

from src.coordinates import P
from src.gamestate import GameState
from src.maps import AttractionMap, action_from_force


class Strategies:
    mine_halite: AttractionMap = None
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
        halite_map = gaussian_filter(halite_map, sigma=0.7, mode='wrap')
        Strategies.mine_halite = AttractionMap(halite_map)

        expansion_map = np.copy(halite_map)
        for base in GameState.board.shipyards.values():
            expansion_map[base.position.norm] = -5000
        expansion_map = gaussian_filter(expansion_map, sigma=3, mode='wrap')
        Strategies.expand = AttractionMap(expansion_map)

        threat_map = np.zeros(GameState.map_size())
        for ship in GameState.board.ships.values():
            if ship.player_id != GameState.board.current_player_id:
                threat_map[ship.position.norm] = -5000
        threat_map = gaussian_filter(threat_map, sigma=1.2, mode='wrap')
        Strategies.avoid_enemies = AttractionMap(threat_map)

        friendly_ships_map = np.zeros(GameState.map_size())
        for ship in GameState.board.current_player.ships:
            friendly_ships_map[ship.position.norm] = -5000
        friendly_ships_map = gaussian_filter(friendly_ships_map, sigma=0.7, mode='wrap')
        Strategies.avoid_friendlies = AttractionMap(friendly_ships_map)

        friendly_bases = np.zeros(GameState.map_size())
        for base in GameState.board.current_player.shipyards:
            friendly_bases[base.position.norm] = -5000
        friendly_bases = gaussian_filter(friendly_bases, sigma=1, mode='wrap')
        Strategies.friendly_bases = AttractionMap(friendly_bases)


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
        Strategies.friendly_bases.priority = 10

    def _make_action(self, ship):
        FORCE_CUTOFF = 6
        total_force = P(0, 0)
        total_priorities = 0
        print(ship)
        for strategy in Strategies():
            force = strategy.at(ship.position.norm)
            print(f"force {force} * {strategy.priority}")
            force *= strategy.priority
            total_priorities += strategy.priority
            total_force += force
        direction = total_force / total_priorities
        print(f"total force {direction}")
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
