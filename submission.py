from kaggle_environments.envs.halite.helpers import *


def agent(obs, config):
    board = Board(obs, config)
    obs = Observation(obs)
    config = Configuration(config)
    #print(board)
    me = board.current_player
    #print(board.cells)

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = ShipAction.NORTH

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
