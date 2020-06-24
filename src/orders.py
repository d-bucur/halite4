import abc
from typing import Optional

from kaggle_environments.envs.halite.helpers import ShipAction, Ship, Board

from src.coordinates import PointAlt
from src.pathing import PathPlanner


class ShipOrder:
    board: Board = None
    planner: PathPlanner = None

    @abc.abstractmethod
    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ...

    @abc.abstractmethod
    def is_done(self, ship: Ship):
        ...

    @abc.abstractmethod
    def next_target(self):
        ...

    def _follow_next_point(self, ship):
        curr_point = self.planner.point_at(ship.id, self.board.step)
        if not curr_point or curr_point != ship.position.norm:
            # print(f"WARN: {ship.id} has deviated from path at turn {self.board.step}. Should be at {curr_point} but is at {ship.position.norm} instead")
            self.planner.remove_path(ship.id)
            self.planner.reserve_path(ship.position.norm, self.next_target(), self.board.step, ship.id)

        next_point = self.planner.point_at(ship.id, self.board.step + 1)
        if not next_point:
            return None
        return next_point.action_from(ship.position.norm, self.board.configuration.size)


class BuildShipyardOrder(ShipOrder):
    def __init__(self, ship: Ship, target: PointAlt):
        self.target = target
        self.planner.reserve_path(ship.position.norm, target, self.board.step, ship.id)

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            self.planner.remove_path(ship.id)
            return ShipAction.CONVERT
        else:
            return self._follow_next_point(ship)

    def is_done(self, ship: Ship):
        ship_pos = ship.position.norm
        missing_halite_for_base = self.board.current_player.halite < self.board.configuration.convert_cost
        done = (ship_pos == self.target and (ship.cell.shipyard or missing_halite_for_base))
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

        return self._follow_next_point(ship)

    def is_done(self, ship: Ship):
        done = (not self.go_harvest and ship.position.norm == self.base_pos)
        if done:
            self.planner.remove_path(ship.id)
        return done

    def next_target(self) -> PointAlt:
        if self.go_harvest:
            return self.target
        else:
            return self.base_pos

    def __repr__(self) -> str:
        if self.go_harvest:
            return f"Harvest at {self.target}"
        else:
            return f"Returning harvest to {self.base_pos} from {self.target}"
