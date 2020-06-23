import abc
import random
from typing import Optional

from kaggle_environments.envs.halite.helpers import ShipAction, Ship, Board

from src.coordinates import PointAlt
from src.pathing import PathPlanner


class ShipOrder:
    board: Board = None
    planner: PathPlanner = None

    # TODO pass board and ship, use ship id as hash and save in sets
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
    def __init__(self, ship: Ship, target: PointAlt):
        self.ship = ship
        self.target = target
        self.planner.reserve_path(ship.position.norm, target, self.board.step, ship.id)

    # TODO remove ship param from everywhere
    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            self.planner.remove_path(ship.id)
            return ShipAction.CONVERT
        else:
            # TODO duplicate code
            next_point = self.planner.point_at(ship.id, self.board.step+1)
            return next_point.action_from(ship.position.norm)

    def is_done(self, ship: Ship):
        ship_pos = ship.position.norm
        # TODO save config in commander class variable
        missing_halite_for_base = self.board.current_player.halite < self.board.configuration.convert_cost
        done = ship_pos == self.target and (ship.cell.shipyard or missing_halite_for_base)
        if done:
            self.planner.remove_path(ship.id)
        return done

    def next_target(self):
        return self.target

    def __repr__(self) -> str:
        return f"Build shipyard at {self.target}"


class HarvestOrder(ShipOrder):
    def __init__(self, ship: Ship, target: PointAlt, base_pos: PointAlt):
        self.target = target
        self.go_harvest = True
        self.base_pos = base_pos
        self.ship = ship
        self.planner.reserve_path(ship.position.norm, target, self.board.step, ship.id)

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            self.planner.remove_path(ship.id)
            if ship.cell.halite < 50 or ship.halite >= self.board.configuration.max_cell_halite:
                self.go_harvest = False
                self.planner.reserve_path(ship.position.norm, self.base_pos, self.board.step, ship.id)
            else:
                # TODO plan staying still for a while
                return None

        next_point = self.planner.point_at(ship.id, self.board.step + 1)
        return next_point.action_from(ship.position.norm)

    def is_done(self, ship: Ship):
        done = not self.go_harvest and ship.position.norm == self.base_pos
        if done:
            self.planner.remove_path(ship.id)
        return done

    def next_target(self):
        if self.go_harvest:
            return self.target
        else:
            return self.base_pos

    def __repr__(self) -> str:
        return f"Harvest at {self.target}"
