import numpy as np

from src.coordinates import SimplePoint


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
