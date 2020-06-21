import pytest
from kaggle_environments import evaluate

from submission import agent


@pytest.mark.skip
def test_wins_against_random():
    my_score, enemy_score = evaluate(
        "halite",
        [agent, "random"],
        num_episodes=1, configuration={"agentExec": "LOCAL"}
    )[0]
    assert my_score > enemy_score


@pytest.mark.skip
def test_wins_against_4_randoms():
    scores = evaluate(
        "halite",
        [agent, "random", "random", "random"],
        num_episodes=1, configuration={"agentExec": "LOCAL"}
    )
    assert scores[0] == max(scores)
