from kaggle_environments.envs.halite.helpers import ShipAction

from src.coordinates import P

size = 9


def test_action_simple():
    start = P(0, 6)
    end = P(0, 5)
    assert end.action_from(start, size) == ShipAction.WEST
    assert start.action_from(end, size) == ShipAction.EAST

    start = P(6, 0)
    end = P(5, 0)
    assert end.action_from(start, size) == ShipAction.NORTH
    assert start.action_from(end, size) == ShipAction.SOUTH


def test_action_wraparound():
    start = P(0, 0)
    end = P(0, 8)
    assert end.action_from(start, size) == ShipAction.WEST
    assert start.action_from(end, size) == ShipAction.EAST

    start = P(0, 0)
    end = P(8, 0)
    assert end.action_from(start, size) == ShipAction.NORTH
    assert start.action_from(end, size) == ShipAction.SOUTH
