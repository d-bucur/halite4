import typing

if typing.TYPE_CHECKING:
    from kaggle_environments.envs.halite.helpers import Board, Configuration


class GameState:
    board: 'Board' = None
    config: 'Configuration' = None
    halite: typing.List[float] = None

    @classmethod
    def map_size(cls):
        return cls.config.size, cls.config.size
