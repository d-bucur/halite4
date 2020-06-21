import abc

from kaggle_environments.envs.halite.helpers import *
import numpy as np
from scipy.ndimage import gaussian_filter


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


'''
Coordinate system
  0      y
    +----->
    |
  x |
    v
'''
SimplePoint = Tuple[int, int]  # TODO subclass from Point


_vars = AttributeDict()  # TODO proper types
_vars.orders = {}
saved_config: Optional[Configuration] = None


class ShipOrder:
    @abc.abstractmethod
    def execute(self, ship: Ship):
        ...


class BuildShipyardOrder(ShipOrder):
    def __init__(self, target: SimplePoint):
        self.target = target

    def execute(self, ship: Ship):
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            return ShipAction.CONVERT
        else:
            return path_to_next(ship_pos, self.target)

    def is_done(self, ship: Ship):
        ship_pos = ship.position.norm
        return ship_pos == self.target and ship.cell.shipyard

    def __repr__(self) -> str:
        return f"Build shipyard at {self.target}"


def path_to_next(start: SimplePoint, target: SimplePoint) -> ShipAction:
    # TODO wraparound map
    tx, ty = target
    sx, sy = start
    dx = tx - sx
    dy = ty - sy
    # TODO move through diagonal or pathfind through heatmap
    if dx < 0:
        return ShipAction.NORTH
    elif dx > 0:
        return ShipAction.SOUTH
    elif dy < 0:
        return ShipAction.WEST
    else:
        return ShipAction.EAST


def agent(obs, config):
    global _vars
    global saved_config
    # add method to convert coordinate system back to topleft 0,0
    Point.norm = property(lambda self: (config.size - 1 - self.y, self.x))

    board = Board(obs, config)
    obs = Observation(obs)
    if not saved_config:
        saved_config = Configuration(config)
    me = board.current_player

    calc_maps(board, config, obs)

    if len(me.shipyards) == 0:
        first_ship = me.ships[0]
        target = calc_highest_in_range(_vars.reward_map, first_ship.position.norm, 4)
        if first_ship.id not in _vars.orders:
            _vars.orders[first_ship.id] = BuildShipyardOrder(target)

    build_ship_actions(me.ships, board.ships)
    build_shipyard_actions(board)

    return me.next_actions


def distance(p1: SimplePoint, p2: SimplePoint) -> int:
    ax, ay = p1
    bx, by = p2
    return abs(bx-ax) + abs(by-ay)


def calc_highest_in_range(arr: np.ndarray, position: SimplePoint, max_distance: int) -> SimplePoint:
    pos_max = None
    val_max = float('-inf')
    for idx, val in np.ndenumerate(arr):
        if distance(position, idx) < max_distance:
            if val_max is None or val_max < val:
                pos_max = idx
                val_max = val
    return pos_max


def calc_maps(board, config, obs) -> None:
    global _vars
    _vars.halite_map = np.array(obs.halite).reshape((config.size, config.size))
    _vars.halite_map = gaussian_filter(_vars.halite_map, sigma=1, mode='wrap')
    _vars.threat_map = np.zeros((config.size, config.size))
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            _vars.threat_map[ship.position.norm] = -5000
    _vars.threat_map = gaussian_filter(_vars.threat_map, sigma=1.2, mode='wrap')
    _vars.reward_map = _vars.halite_map * 3 + _vars.threat_map


def build_ship_actions(ships: List[Ship], ships_dict: Dict[str, Ship]) -> None:
    for ship_id in list(_vars.orders.keys()):
        if ship_id not in ships_dict or _vars.orders[ship_id].is_done(ships_dict[ship_id]):
            del _vars.orders[ship_id]
    for ship in ships:
        if ship.id in _vars.orders:
            ship.next_action = _vars.orders[ship.id].execute(ship)
        else:
            ship.next_action = ShipAction.EAST


def build_shipyard_actions(board: Board) -> None:
    affordable_ships = board.current_player.halite // saved_config.spawn_cost
    for shipyard in board.current_player.shipyards:
        if affordable_ships > 0:
            shipyard.next_action = ShipyardAction.SPAWN
            affordable_ships -= 1


def internals():
    # helper method to expose data to notebook reliably with autoreload
    return _vars
