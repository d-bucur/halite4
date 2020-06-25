from kaggle_environments.envs.halite.helpers import Point, Ship


def ship_repr(self):
    return f"Ship {self.id} at {self.position.norm}"


Ship.__repr__ = ship_repr
