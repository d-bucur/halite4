from kaggle_environments import evaluate

from submission import agent


def test_wins_against_random():
    my_score, enemy_score = evaluate(
        "halite",
        [agent, "random"],
        num_episodes=1, configuration={"agentExec": "LOCAL"}
    )[0]
    assert my_score > enemy_score
