from typing import Tuple

from kaggle_environments.envs.halite.helpers import Point

'''
Coordinate system
  0      y
    +----->
    |
  x |
    v
'''
PointAlt = Tuple[int, int]


def from_point(p: Point, size: int) -> PointAlt:
    return size - 1 - p.y, p.x
