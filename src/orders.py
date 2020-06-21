import abc

from kaggle_environments.envs.halite.helpers import ShipAction, Ship

from src.coordinates import SimplePoint


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


class ShipOrder:
    @abc.abstractmethod
    def execute(self, ship: Ship):
        ...

    @abc.abstractmethod
    def is_done(self, ship: Ship):
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
