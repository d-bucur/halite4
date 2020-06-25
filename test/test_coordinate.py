from kaggle_environments.envs.halite.helpers import Point

from src.coordinates import P


def test_pointalt():
    p1 = P(1, 2)
    p2 = P(1, 2)
    p3 = Point(1, 2)
    assert type(p1) == P
    assert type(p1 + p2) == P
    assert type(p1 + p3) == P
    assert type(p1.tuple) == tuple
