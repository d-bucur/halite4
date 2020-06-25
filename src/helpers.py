from heapq import heappushpop, heappush
from typing import List

import numpy as np

from src.coordinates import P


def distance(p1: P, p2: P) -> int:
    ax, ay = p1
    bx, by = p2
    return abs(bx-ax) + abs(by-ay)


def calc_highest_in_range(arr: np.ndarray, position: P, max_distance: int, count: int) -> List[P]:
    highest = []
    lowest_val = float('-inf')
    for idx, val in np.ndenumerate(arr):
        if distance(position, idx) < max_distance:  # TODO iterate efficiently
            if lowest_val < val:
                if len(highest) == count:
                    heappushpop(highest, (val, idx))
                else:
                    heappush(highest, (val, idx))
                lowest_val, _ = highest[0]
    return [P(i[1][0], i[1][1]) for i in highest]
