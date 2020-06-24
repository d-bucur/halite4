import typing

if typing.TYPE_CHECKING:
    from kaggle_environments.envs.halite.helpers import Board, Configuration
    from src.pathing import PathPlanner


class GameState:
    board: 'Board' = None
    config: 'Configuration' = None
    planner: 'PathPlanner' = None
    halite: typing.List[float] = None
