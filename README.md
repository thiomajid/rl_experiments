# RL Experiments

This repository contains my experiments with RL on various problems for learning purposes.

## Gymnax Rendering Issues

Rendering issues in Gymnax can be solved by updating these two functions from this [gymnax file](.venv/lib/python3.12/site-packages/gymnax/visualize/vis_gym.py) to follow gymnasium's new API but also installing "gymnasium[classic_control]"

```python
def init_gym(ax, env, state, params):
    """Initialize gym environment."""
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    else:
        gym_env = gym.make(env.name, render_mode="rgb_array")
    gym_env.reset()
    set_gym_params(gym_env, env.name, params)
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    gym_env.reset()
    rgb_array = gym_env.render()
    ax.set_xticks([])
    ax.set_yticks([])
    gym_env.close()
    return ax.imshow(rgb_array)


def update_gym(im, env, state):
    """Update gym environment."""
    if env.name == "Pendulum-v1":
        gym_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    else:
        gym_env = gym.make(env.name, render_mode="rgb_array")
    gym_state = get_gym_state(state, env.name)
    if env.name == "Pendulum-v1":
        gym_env.env.last_u = gym_state[-1]
    gym_env.env.state = gym_state
    gym_env.reset()
    rgb_array = gym_env.render()
    im.set_data(rgb_array)
    gym_env.close()
    return im
```
