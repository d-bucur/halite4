from kaggle_environments.envs.halite.helpers import *
import numpy as np
from scipy.ndimage import gaussian_filter


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


_vars = AttributeDict()


def internals():
    # helper method to expose data to notebook reliably with autoreload
    return _vars


def agent(obs, config):
    global _vars
    # add method to convert coordinate system back to topleft 0,0
    Point.norm = property(lambda self: (config.size - 1 - self.y, self.x))

    board = Board(obs, config)
    obs = Observation(obs)
    config = Configuration(config)
    me = board.current_player

    _vars.halite_map = np.array(obs.halite).reshape((config.size, config.size))
    _vars.halite_map = gaussian_filter(_vars.halite_map, sigma=1, mode='wrap')

    _vars.threat_map = np.zeros((config.size, config.size))
    for ship in board.ships.values():
        if ship.player_id != board.current_player_id:
            _vars.threat_map[ship.position.norm] = -5000
    _vars.threat_map = gaussian_filter(_vars.threat_map, sigma=1.2, mode='wrap')

    _vars.reward_map = _vars.halite_map * 3 + _vars.threat_map

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = ShipAction.NORTH

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
