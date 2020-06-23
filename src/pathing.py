from collections import deque, defaultdict
import logging
from typing import List, Dict, Tuple

from src.coordinates import PointAlt

PlanningPoint = Tuple[PointAlt, int]


class PathPlanner:
    neighbor_tiles = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def __init__(self, size: int):
        self.size = size
        self.planning: Dict[PlanningPoint, str] = {}
        self.plan_x_id: Dict[str, Dict[int, PointAlt]] = defaultdict(dict)

    def reserve_path(self, start: PointAlt, target: PointAlt, start_time: int, path_id: str) -> List[PointAlt]:
        # TODO check if path_id already exists
        path = self.calc_path(start, target, start_time)
        time = start_time
        for point in path:
            plan_point = (point, time)
            self.planning[plan_point] = path_id
            self.plan_x_id[path_id][time] = point
            time += 1
        return path

    def calc_path(self, start: PointAlt, target: PointAlt, start_time: int) -> List[PointAlt]:
        frontier = deque([(start, start_time)])
        came_from = {
            start: None
        }

        while frontier:
            current, time = frontier.popleft()
            if current == target:
                break
            for neighbor in self._neighbors(current):
                plan_point = (neighbor, time + 1)
                if neighbor not in came_from and plan_point not in self.planning:
                    frontier.append(plan_point)
                    came_from[neighbor] = current

        path = [target]
        current = target
        while current != start:
            path.append(came_from[current])
            current = came_from[current]
        path.reverse()
        # TODO more efficient to just write times here
        return path

    def remove_path(self, path_id: str):
        if path_id not in self.plan_x_id:
            logging.warning(f"Tried to remove path for {path_id} but it was not present")
            return
        for time, point in self.plan_x_id[path_id].items():
            del self.planning[(point, time)]
        del self.plan_x_id[path_id]

    def _neighbors(self, p: PointAlt):
        for n in self.neighbor_tiles:
            tentative = p + n
            yield tentative.resize(self.size)
