import open_spiel  # type: ignore
from shimmy import OpenSpielCompatibilityV0  # type: ignore

env = OpenSpielCompatibilityV0(game_name="kuhn_poker", render_mode="human")
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(info["action_mask"])
    print(env.last(), action)
    env.step(action)
    env.render()
env.close()
