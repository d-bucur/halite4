from src.coordinates import PointAlt
from src.pathing import PathPlanner


def test_paths_not_intersecting():
    planner = PathPlanner(30)
    planner.reserve_path(PointAlt(10, 11), PointAlt(12, 8), 0, 'a')
    assert len(planner.plan_x_id['a']) == 6  # optimal path
    planner.reserve_path(PointAlt(10, 10), PointAlt(15, 10), 0, 'b')
    assert len(planner.plan_x_id['b']) == 6  # no intersection, optimal path
    planner.reserve_path(PointAlt(14, 6), PointAlt(14, 12), 0, 'c')
    assert len(planner.plan_x_id['c']) > 7  # intersects, must be longer than optimal path
    planner.remove_path('b')
    assert 'b' not in planner.plan_x_id
    assert not any(v == 'b' for v in planner.planning.values())
