from kaggle_environments.envs.halite.helpers import Point

from src.coordinates import PointAlt


def test_pointalt():
    p1 = PointAlt(1, 2)
    p2 = PointAlt(1, 2)
    p3 = Point(1, 2)
    assert type(p1) == PointAlt
    assert type(p1 + p2) == PointAlt
    assert type(p1 + p3) == PointAlt
    assert type(p1.tuple) == tuple
