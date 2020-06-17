from kaggle_environments.envs.halite.helpers import *
import numpy as np
from scipy.ndimage import gaussian_filter


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

    internals.halite_map = np.array(obs.halite).reshape((config.size, config.size))
    # internals.halite_heat = gaussian_filter(internals.halite_map, sigma=1)
    internals.threat_map = np.zeros((config.size, config.size))
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            internals.threat_map[ship.position] = -10000


    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = ShipAction.NORTH

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
