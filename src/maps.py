from enum import IntEnum, auto
from typing import NewType

import numpy as np

from src.coordinates import P


class FieldDirection(IntEnum):
    STAND = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


FieldType = np.ndarray


class FlowField:
    def __init__(self, field: FieldType) -> None:
        self.field = field

    def __repr__(self):
        return ""


class AttractionMap:
    def __init__(self, array: np.ndarray, cutoff: float = 0) -> None:
        self.array = array
        self.flow = make_field(self.array, cutoff)


_neighbor_tiles = {
    P(0, 1): FieldDirection.RIGHT,
    P(1, 0): FieldDirection.DOWN,
    P(-1, 0): FieldDirection.UP,
    P(0, -1): FieldDirection.LEFT
}


def make_field(array: np.ndarray, cutoff) -> FieldType:
    gradient = np.gradient(array)
    field = gradient[0]
    for curr_pos, val in np.ndenumerate(array):
        dx = abs(gradient[0][curr_pos])
        dy = abs(gradient[1][curr_pos])
        if dx <= cutoff and dy <= cutoff:
            field[curr_pos] = FieldDirection.STAND
        elif dx > dy:
            if gradient[0][curr_pos] > 0:
                field[curr_pos] = FieldDirection.DOWN
            else:
                field[curr_pos] = FieldDirection.UP
        else:
            if gradient[1][curr_pos] > 0:
                field[curr_pos] = FieldDirection.RIGHT
            else:
                field[curr_pos] = FieldDirection.LEFT
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
