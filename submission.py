from kaggle_environments.envs.halite.helpers import Point, Board

from src.commander import Commander
from src.coordinates import from_point

_commander = Commander()


def commander():
    # helper method to expose data to notebook reliably with autoreload
    return _commander


def agent(obs, config):
    # add method to convert coordinate system back to topleft 0,0
    Point.norm = property(lambda self: from_point(self, config.size))

    board = Board(obs, config)
    _commander.update(board, obs)
    return _commander.get_next_actions()
