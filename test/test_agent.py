import pytest
from kaggle_environments import make


@pytest.mark.skip("does not check for bot internal errors")
def test_agent_completes():
    env = make("halite", configuration={"episodeSteps": 400}, debug=True)
    env.run(["submission.py", "submission.py", "submission.py", "submission.py"])
    assert env.done
