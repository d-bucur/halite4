from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from kaggle_environments.envs.halite.helpers import Ship, ShipAction

from src.coordinates import P
from src.gamestate import GameState


@dataclass
class PreferredActions:
    ship: Ship
    actions: List[Optional[ShipAction]]


class Planner:
    def __init__(self):
        self._reserved: Dict[P, List[PreferredActions]] = defaultdict(lambda: list())

    # TODO get secondary points from gradient
    _secondary_actions = {
        ShipAction.NORTH: [ShipAction.EAST, None],
        ShipAction.EAST: [ShipAction.SOUTH, None],
        ShipAction.SOUTH: [ShipAction.WEST, None],
        ShipAction.WEST: [ShipAction.NORTH, None],
        None: [],
    }

    def reserve_action(self, ship: Ship, next_action: ShipAction):
        pos_delta = P.from_action(next_action)
        if not pos_delta:
            return
        pos = ship.position.norm + pos_delta
        pos = pos.resize(GameState.config.size)
        actions = self._secondary_actions[next_action]
        self._reserved[pos].append(PreferredActions(ship, [next_action] + actions))

    def resolve_collisions(self):
        while True:
            collision_found = False
            for pos, actions in self._reserved.items():
                if len(actions) > 1:
                    collision_found = True
                    self._replan_ships(actions)
                    break
            if not collision_found:
                break

    def at(self, point: P) -> List[PreferredActions]:
        return self._reserved[point]

    def _replan_ships(self, actions: List[PreferredActions]):
        replanned_action_idx = max(range(len(actions)), key=lambda i: len(actions[i].actions))
        replanned_action = actions.pop(replanned_action_idx)
        del replanned_action.actions[0]
        new_action = replanned_action.actions[0]
        replanned_action.ship.next_action = new_action
        new_pos = P.from_action(new_action) + replanned_action.ship.position.norm
        new_pos = new_pos.resize(GameState.config.size)
        self._reserved[new_pos].append(replanned_action)
