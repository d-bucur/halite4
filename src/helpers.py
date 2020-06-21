from heapq import heappop, heappushpop, heappush
from typing import List

import numpy as np

from src.coordinates import PointAlt


def distance(p1: PointAlt, p2: PointAlt) -> int:
    ax, ay = p1
    bx, by = p2
    return abs(bx-ax) + abs(by-ay)


def calc_highest_in_range(arr: np.ndarray, position: PointAlt, max_distance: int, count: int) -> List[PointAlt]:
    highest = []
    lowest_val = float('-inf')
    for idx, val in np.ndenumerate(arr):
        if distance(position, idx) < max_distance:
            if lowest_val < val:
                if len(highest) == count:
                    heappushpop(highest, (val, idx))
                else:
                    heappush(highest, (val, idx))
                lowest_val, _ = highest[0]
    return [i[1] for i in highest]
