from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Set

from src.coordinates import PointAlt

PlanningPoint = Tuple[PointAlt, int]


class PathPlanner:
    neighbor_tiles = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    def __init__(self, size: int):
        self.size = size
        self.planning: Dict[PlanningPoint, str] = {}
        self.plan_x_id: Dict[str, Dict[int, PointAlt]] = defaultdict(dict)

    # TODO find efficient way to cleanup older times

    def reserve_path(self, start: PointAlt, target: PointAlt, start_time: int, path_id: str) -> List[PointAlt]:
        if path_id in self.plan_x_id:
            self.remove_path(path_id)

        path = self.calc_path(start, target, start_time)
        time = start_time
        for point in path:
            plan_point = (point, time)
            self.planning[plan_point] = path_id
            self.plan_x_id[path_id][time] = point
            time += 1
        return path

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
                if neighbor not in came_from and plan_point not in self.planning:
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
        # TODO more efficient to just write times here
        return path

    def reserve_standing(self, position: PointAlt, start_time: int, end_time: int, path_id: str):
        for time in range(start_time, end_time+1):
            plan_point = (position, time)
            self.planning[plan_point] = path_id
            self.plan_x_id[path_id][time] = position
            # TODO if collides with a path invalidate it

    def remove_path(self, path_id: str):
        if path_id not in self.plan_x_id:
            # logging.warning(f"Tried to remove path for {path_id} but it was not present")
            return
        for time, point in self.plan_x_id[path_id].items():
            k = (point, time)
            if k in self.planning:  # TODO should always happen. sometimes doesn't. probably a bug
                del self.planning[k]
        del self.plan_x_id[path_id]

    def point_at(self, path_id: str, time: int) -> Optional[PointAlt]:
        return self.plan_x_id[path_id].get(time, None)  # TODO  bug: sometimes time is not inside. should always be

    def _neighbors(self, p: PointAlt):
        for n in self.neighbor_tiles:
            tentative = p + n
            yield tentative.resize(self.size)

    def cleanup_missing(self, ids_to_keep: Set[str]):
        to_remove = [s for s in self.plan_x_id.keys() if s not in ids_to_keep]
        for path_id in to_remove:
            self.remove_path(path_id)
