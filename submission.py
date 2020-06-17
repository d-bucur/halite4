from kaggle_environments.envs.halite.helpers import *
from numpy import array


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


internals = AttributeDict()


def agent(obs, config):
    global internals

    board = Board(obs, config)
    obs = Observation(obs)
    config = Configuration(config)
    me = board.current_player
    internals.halite_map = array(obs.halite).reshape((config.size, config.size))

    #print(board)
    #print(board.cells)

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = ShipAction.NORTH

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
