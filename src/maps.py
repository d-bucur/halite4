import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Callable, Iterable

import numpy as np
from kaggle_environments.envs.halite.helpers import ShipAction

from src.coordinates import P
from src.gamestate import GameState

FieldType = np.ndarray


class AttractionMap:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.flow = make_field(self.array)
        self.priority = 1

    def flow_at(self, pos: P, escape_mins = False):
        """ returns a force which is the direction of the maximum ascent """
        MIN_CUTOFF = 0.15
        x, y = pos
        dx = self.flow[0][pos]
        dy = self.flow[1][pos]
        v = self.array[pos]
        # escape local minimums
        x_1 = (x + 1) % GameState.config.size
        if escape_mins and dx <= MIN_CUTOFF and v < self.array[(x - 1, y)] and v < self.array[(x_1, y)]:
            # TODO check index ranges
            if random.randint(0, 1) == 0:
                dx = self.flow[0][(x - 1, y)]
            else:
                dx = self.flow[0][(x_1, y)]

        y_1 = (y + 1) % GameState.config.size
        if escape_mins and dy <= MIN_CUTOFF and v < self.array[(x, y - 1)] and v < self.array[(x, y_1)]:
            # TODO check index ranges
            if random.randint(0, 1) == 0:
                dy = self.flow[1][(x, y - 1)]
            else:
                dy = self.flow[1][(x, y_1)]

        return P(dx, dy)

    def value_at(self, pos):
        return self.array[pos]

    def get_force(self, pos: P, name: str):
        return ContributingForce(
            name,
            self.flow_at(pos),
            self.priority
        )


def make_field(arr: np.ndarray):
    # x derivate
    dx = np.empty(arr.shape)
    for y in range(0, arr.shape[1]):
        curr = arr[(-2, y)]
        next = arr[(-1, y)]
        for x in range(-1, arr.shape[0] - 1):
            prev = curr
            curr = next
            next = arr[(x + 1, y)]
            if curr >= prev and curr >= next:
                dx[(x, y)] = 0.0
            else:
                dx[(x, y)] = next - prev

    # y derivate
    dy = np.empty(arr.shape)
    for x in range(0, arr.shape[0]):
        curr = arr[(x, -2)]
        next = arr[(x, -1)]
        for y in range(-1, arr.shape[1] - 1):
            prev = curr
            curr = next
            next = arr[(x, y + 1)]
            if curr >= prev and curr >= next:
                dy[(x, y)] = 0.0
            else:
                dy[(x, y)] = next - prev
    return [dx, dy]


def _neighbors(p: P, size):
    yield (p + P(0, 1)).resize(size)
    yield (p + P(1, 0)).resize(size)
    yield (p + P(-1, 0)).resize(size)
    yield (p + P(0, -1)).resize(size)


def visit_map(arr: np.ndarray, start_points: Iterable[P], start_values: float, step_func: Callable):
    frontier = deque()
    visited = set()
    for p in start_points:
        arr[p] = start_values
        frontier.append((p, start_values))
        visited.add(p)
    while frontier:
        p, last_val = frontier.popleft()
        for neigh in _neighbors(p, arr.shape[0]):
            if neigh not in visited:
                visited.add(neigh)
                next_val = step_func(last_val)
                arr[neigh] = next_val
                frontier.append((neigh, next_val))


def action_from_force(f: P, cutoff: float = 0) -> Optional[ShipAction]:
    x, y = f
    abs_x = abs(x)
    abs_y = abs(y)
    if abs_x <= cutoff and abs_y <= cutoff:
        return None
    elif abs_x > abs_y:
        if x > 0:
            return ShipAction.SOUTH
        else:
            return ShipAction.NORTH
    else:
        if y > 0:
            return ShipAction.EAST
        else:
            return ShipAction.WEST


@dataclass
class ContributingForce:
    name: str
    force: P
    weight: float

    def __str__(self):
        return f"{self.name:16} {self.force*self.weight} = {self.force} * {self.weight:0.1f}"


class ForceCombination:
    def __init__(self):
        self._forces: List[ContributingForce] = []

    def add(self, force: ContributingForce):
        self._forces.append(force)
        logging.info(force)

    def total(self) -> P:
        total_force = P(0, 0)
        total_weights = 0
        for f in self._forces:
            total_force += f.force * f.weight
            total_weights += f.weight
        return total_force / total_weights
