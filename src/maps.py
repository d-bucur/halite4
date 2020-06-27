import random
from enum import IntEnum
from typing import Optional

import numpy as np
from kaggle_environments.envs.halite.helpers import ShipAction

from src.coordinates import P

FieldType = np.ndarray


class AttractionMap:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.flow = make_field(self.array)
        self.priority = 1

    def at(self, pos: P):
        """ returns a force which is the direction of the maximum ascent """
        dx = self.flow[0][pos]
        dy = self.flow[1][pos]
        return P(dx, dy)


def make_field_gradient(arr: np.ndarray) -> FieldType:
    return np.gradient(arr)


def make_field(arr: np.ndarray):
    # x derivate
    dx = np.empty(arr.shape)
    for y in range(0, arr.shape[1]):
        curr = arr[(-2, y)]
        next = arr[(-1, y)]
        for x in range(-1, arr.shape[0] - 1):
            prev = curr
            curr = next
            next = arr[(x + 1, y)]
            if curr >= prev and curr >= next:
                dx[(x, y)] = 0.0
            else:
                dx[(x, y)] = next - prev

    # y derivate
    dy = np.empty(arr.shape)
    for x in range(0, arr.shape[0]):
        curr = arr[(x, -2)]
        next = arr[(x, -1)]
        for y in range(-1, arr.shape[1] - 1):
            prev = curr
            curr = next
            next = arr[(x, y + 1)]
            if curr >= prev and curr >= next:
                dy[(x, y)] = 0.0
            else:
                dy[(x, y)] = next - prev
    return [dx, dy]


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
