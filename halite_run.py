from kaggle_environments import make

env = make("halite", configuration={"episodeSteps": 400}, debug=True)

env.run(["submission.py", "random", "random", "random"])


def render_stdout():
    out = env.render(mode="ansi")
    print(out)


def render_html():
    out = env.render(mode="html")
    with open("result.html", "w") as file:
        file.write(out)


render_html()
