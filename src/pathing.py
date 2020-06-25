from collections import deque
from typing import List

from src.coordinates import PointAlt


class PathPlanner:
    neighbor_tiles = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def calc_path(self, start: PointAlt, target: PointAlt, start_time: int) -> List[PointAlt]:
        # TODO handle case when no path is found
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
                if neighbor not in came_from:
                    frontier.append(plan_point)
                    came_from[neighbor] = current

        path = [target]
        current = target
        while current != start:
            if current not in came_from:
                print(f"Could not find path from {start} to {target}")
                return []  # TODO is this the case in which there is no path?
            path.append(came_from[current])
            current = came_from[current]
        path.reverse()
        return path

    def _neighbors(self, p: PointAlt):
        for n in self.neighbor_tiles:
            tentative = p + n
            yield tentative.resize(self.size)
