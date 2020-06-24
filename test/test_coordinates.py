from kaggle_environments.envs.halite.helpers import ShipAction

from src.coordinates import PointAlt

size = 9


def test_action_simple():
    start = PointAlt(0, 6)
    end = PointAlt(0, 5)
    assert end.action_from(start, size) == ShipAction.WEST
    assert start.action_from(end, size) == ShipAction.EAST

    start = PointAlt(6, 0)
    end = PointAlt(5, 0)
    assert end.action_from(start, size) == ShipAction.NORTH
    assert start.action_from(end, size) == ShipAction.SOUTH


def test_action_wraparound():
    start = PointAlt(0, 0)
    end = PointAlt(0, 8)
    assert end.action_from(start, size) == ShipAction.WEST
    assert start.action_from(end, size) == ShipAction.EAST

    start = PointAlt(0, 0)
    end = PointAlt(8, 0)
    assert end.action_from(start, size) == ShipAction.NORTH
    assert start.action_from(end, size) == ShipAction.SOUTH
