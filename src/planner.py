from collections import defaultdict
from typing import Dict, List

from kaggle_environments.envs.halite.helpers import Ship, ShipAction

from src.coordinates import P
from src.gamestate import GameState


class Planner:
    def __init__(self):
        self._reserved: Dict[P, List[Ship]] = defaultdict(lambda: list())

    def reserve_action(self, ship: Ship, next_action: ShipAction):
        next_action = P.from_action(next_action)
        if not next_action:
            return
        pos = ship.position.norm + next_action
        pos.resize(GameState.config.size)
        self._reserved[pos].append(ship)

    def resolve_collisions(self):
        while True:
            collision_found = False
            for pos, ships in self._reserved.items():
                if len(ships) > 1:
                    collision_found = True
                    replanned_ship = ships[0]
                    replanned_ship.next_action = None
                    del ships[0]
                    self._reserved[replanned_ship.position.norm].append(replanned_ship)
                    break
            if not collision_found:
                break
