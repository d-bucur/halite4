from kaggle_environments import make, evaluate
from kaggle_environments.envs.halite.halite import random_agent

from submission import agent


def test_agent_completes():
    # TODO does not check for bot internal errors
    env = make("halite", configuration={"episodeSteps": 100})
    env.run([agent, random_agent])
    assert env.done
