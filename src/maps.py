from enum import IntEnum
from typing import Optional

import numpy as np
from kaggle_environments.envs.halite.helpers import ShipAction

from src.coordinates import P

FieldType = np.ndarray


class FlowField:
    def __init__(self, field: FieldType) -> None:
        self.field = field

    def at(self, pos: P):
        return P(self.field[0][pos], self.field[1][pos])

    def __repr__(self):
        return str(self.field)


class AttractionMap:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.flow = FlowField(make_field(self.array))
        self.priority = 1


def make_field(array: np.ndarray) -> FieldType:
    return np.gradient(array)


def action_from_force(f: P, cutoff: float = 0) -> Optional[ShipAction]:
    x, y = f
    abs_x = abs(x)
    abs_y = abs(y)
    if abs_x <= cutoff and abs_y <= cutoff:
        return None
    elif abs_x > abs_y:
        if x > 0:
            return ShipAction.SOUTH
        else:
            return ShipAction.NORTH
    else:
        if y > 0:
            return ShipAction.EAST
        else:
            return ShipAction.WEST


class FieldDirection(IntEnum):
    STAND = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


_neighbor_tiles = {
    P(0, 1): FieldDirection.RIGHT,
    P(1, 0): FieldDirection.DOWN,
    P(-1, 0): FieldDirection.UP,
    P(0, -1): FieldDirection.LEFT
}


def make_field_discrete(array: np.ndarray, cutoff) -> FieldType:
    gradient = np.gradient(array)
    field = gradient[0]
    for curr_pos, val in np.ndenumerate(array):
        field[curr_pos] = action_from_force(P(gradient[0][curr_pos], gradient[1][curr_pos]), cutoff)
    return field


def _make_field_iterating(array: np.ndarray) -> FieldType:
    field = np.zeros(array.shape)
    for curr_pos, val in np.ndenumerate(array):
        max_pos, max_val = None, None
        for delta_pos in _neighbor_tiles.keys():
            adj_pos = delta_pos + curr_pos
            adj_pos = adj_pos.resize(array.shape[0])
            if not max_pos or field[adj_pos] > max_val:
                max_val = field[adj_pos]
                max_pos = _neighbor_tiles[delta_pos]
        field[curr_pos] = max_pos.value if max_pos else None
    return field
