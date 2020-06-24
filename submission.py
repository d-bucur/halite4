from kaggle_environments.envs.halite.helpers import Point, Board

from src.commander import Commander
from src.coordinates import from_point
from src.gamestate import GameState

commander = None


def agent(obs, config):
    global commander
    board = Board(obs, config)
    if commander is None:
        commander = Commander(board.configuration)
    # add method to convert coordinate system back to topleft 0,0
    Point.norm = property(lambda self: from_point(self, config.size))

    GameState.board = board
    GameState.halite = obs.halite
    return commander.get_next_actions()
