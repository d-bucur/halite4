from typing import Tuple, Type, Optional

from kaggle_environments.envs.halite.helpers import Point, ShipAction


class PointAlt(tuple):
    # TODO copy all logic from Point, but keep separate class
    """
    Coordinate system
      0      y
        +----->
        |
      x |
        v
    """
    def __new__(cls: Type['PointAlt'], x: int, y: int):
        return super(PointAlt, cls).__new__(cls, tuple((x, y)))

    def resize(self, size: int):
        """ does not consider points that are > size*2 """
        x, y = self[0], self[1]
        if self[0] >= size:
            x = self[0] - size
        elif self[0] < 0:
            x = self[0] + size
        if self[1] >= size:
            y = self[1] - size
        elif self[1] < 0:
            y = self[1] + size
        return PointAlt(x, y)

    def __add__(self, other) -> 'PointAlt':
        return PointAlt(self[0] + other[0], self[1] + other[1])

    def action_from(self, start: 'PointAlt', size: int) -> Optional[ShipAction]:
        dx = (self[1] - start[1]) % size
        dy = (self[0] - start[0]) % size
        middle = size / 2
        if dy == 0:
            if dx > middle:
                return ShipAction.WEST
            elif dx < middle:
                return ShipAction.EAST
        elif dx == 0:
            if dy > middle:
                return ShipAction.NORTH
            elif dy < middle:
                return ShipAction.SOUTH
        return None

    def __repr__(self):
        return f"P({self[0]}, {self[1]})"


def from_point(p: Point, size: int) -> PointAlt:
    return PointAlt(size - 1 - p.y, p.x)
