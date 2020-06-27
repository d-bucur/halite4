from kaggle_environments.envs.halite.helpers import Point, Ship


def ship_repr(self):
    return f"Ship {self.id} at {self.position.norm}"


Ship.__repr__ = ship_repr
Point.__str__ = lambda self: f"({self.x:0.1f}, {self.y:0.1f})"
