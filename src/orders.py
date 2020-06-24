import abc
from typing import Optional

from kaggle_environments.envs.halite.helpers import ShipAction, Ship

from src.coordinates import PointAlt
from src.gamestate import GameState


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

    def _follow_next_point(self, ship):
        curr_point = GameState.planner.point_at(ship.id, GameState.board.step)
        if not curr_point or curr_point != ship.position.norm:
            # path is outdated, recalculate
            # print(f"WARN: {ship.id} has deviated from path at turn {GameState.board.step}. Should be at {curr_point} but is at {ship.position.norm} instead")
            GameState.planner.remove_path(ship.id)
            GameState.planner.reserve_path(ship.position.norm, self.next_target(), GameState.board.step, ship.id)

        next_point = GameState.planner.point_at(ship.id, GameState.board.step + 1)
        if not next_point:
            return None
        return next_point.action_from(ship.position.norm, GameState.board.configuration.size)


class BuildShipyardOrder(ShipOrder):
    def __init__(self, ship: Ship, target: PointAlt):
        self.target = target
        GameState.planner.reserve_path(ship.position.norm, target, GameState.board.step, ship.id)

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            GameState.planner.remove_path(ship.id)
            return ShipAction.CONVERT
        else:
            return self._follow_next_point(ship)

    def is_done(self, ship: Ship):
        ship_pos = ship.position.norm
        missing_halite_for_base = GameState.board.current_player.halite < GameState.board.configuration.convert_cost
        done = (ship_pos == self.target and (ship.cell.shipyard or missing_halite_for_base))
        if done:
            GameState.planner.remove_path(ship.id)
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
        GameState.planner.reserve_path(ship.position.norm, target, GameState.board.step, ship.id)

    def execute(self, ship: Ship) -> Optional[ShipAction]:
        ship_pos = ship.position.norm
        if ship_pos == self.target:
            GameState.planner.remove_path(ship.id)
            if ship.cell.halite < 50 or ship.halite >= GameState.board.configuration.max_cell_halite:
                self.go_harvest = False
                GameState.planner.reserve_path(ship.position.norm, self.base_pos, GameState.board.step, ship.id)
            else:
                # TODO plan staying still for a while
                return None

        return self._follow_next_point(ship)

    def is_done(self, ship: Ship):
        done = (not self.go_harvest and ship.position.norm == self.base_pos)
        if done:
            GameState.planner.remove_path(ship.id)
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
