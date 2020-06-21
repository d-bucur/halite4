import abc
from typing import Optional

from kaggle_environments.envs.halite.helpers import ShipAction, Ship, Board

from src.coordinates import PointAlt


def path_to_next(start: PointAlt, target: PointAlt) -> ShipAction:
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
    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ...

    @abc.abstractmethod
    def is_done(self, ship: Ship):
        ...

    @abc.abstractmethod
    def next_target(self):
        ...


class BuildShipyardOrder(ShipOrder):
    def __init__(self, target: PointAlt, board: Board):
        self.target = target
        self.board = board

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            return ShipAction.CONVERT
        else:
            return path_to_next(ship_pos, self.target)

    def is_done(self, ship: Ship):
        ship_pos = ship.position.norm
        missing_halite_for_base = self.board.current_player.halite < self.board.configuration.convert_cost
        return ship_pos == self.target and (ship.cell.shipyard or missing_halite_for_base)

    def next_target(self):
        return self.target

    def __repr__(self) -> str:
        return f"Build shipyard at {self.target}"


class HarvestOrder(ShipOrder):
    def __init__(self, target: PointAlt, base_pos: PointAlt, board: Board):
        self.target = target
        self.board = board
        self.go_harvest = True
        self.base_pos = base_pos

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            if ship.cell.halite < 50 or ship.halite >= self.board.configuration.max_cell_halite:
                self.go_harvest = False
            else:
                return None

        return path_to_next(ship_pos, self.next_target())

    def is_done(self, ship: Ship):
        return not self.go_harvest and ship.position.norm == self.base_pos

    def next_target(self):
        if self.go_harvest:
            return self.target
        else:
            return self.base_pos

    def __repr__(self) -> str:
        return f"Harvest at {self.target}"
