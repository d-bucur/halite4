from typing import Optional, Callable, Union, Tuple

from kaggle_environments.envs.halite.helpers import Point, ShipAction


class P(Point):
    """
    Coordinate system
      0      y
        +----->
        |
      x |
        v
    """

    def map(self, f: Callable[[int], int]) -> 'P':
        return P(f(self[0]), f(self[1]))

    def map2(self, other: Union[Tuple[int, int], 'Point'], f: Callable[[int, int], int]) -> 'P':
        return P(f(self[0], other[0]), f(self[1], other[1]))

    def __truediv__(self, factor: int) -> 'Point':
        return self.map(lambda x: x / factor)

    def resize(self, size: int):
        """ does not consider points that are > size*2 """
        # TODO rewrite with modulo
        x, y = self[0], self[1]
        if self[0] >= size:
            x = self[0] - size
        elif self[0] < 0:
            x = self[0] + size
        if self[1] >= size:
            y = self[1] - size
        elif self[1] < 0:
            y = self[1] + size
        return P(x, y)

    def action_from(self, start: 'P', size: int) -> Optional[ShipAction]:
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

    @property
    def tuple(self) -> Tuple[int, int]:
        return self.x, self.y


def from_point(p: Point, size: int) -> P:
    return P(size - 1 - p.y, p.x)
