from kaggle_environments.envs.halite.helpers import Point, Board

from src.commander import Commander

_commander = Commander()


def commander():
    # helper method to expose data to notebook reliably with autoreload
    return _commander


def agent(obs, config):
    global _vars
    global saved_config
    # add method to convert coordinate system back to topleft 0,0
    Point.norm = property(lambda self: (config.size - 1 - self.y, self.x))

    board = Board(obs, config)
    _commander.update(board, obs)
    return _commander.get_next_actions()
