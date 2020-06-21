from kaggle_environments import make

from submission import agent


def test_agent_completes():
    env = make("halite", configuration={"episodeSteps": 400}, debug=True)
    env.run([agent, agent])
    assert not any(step[0].status == 'ERROR' for step in env.steps)
