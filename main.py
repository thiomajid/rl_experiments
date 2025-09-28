import jax
import jax.tree as jtu
import jumanji
from jumanji.wrappers import AutoResetWrapper


def main():
    ENV_ID = "Snake-v1"
    VIDEO_FILENAME = ENV_ID.split("-")[0] + ".mp4"

    env = jumanji.make(ENV_ID)
    env = AutoResetWrapper(env)

    num_actions = env.action_spec.num_values
    rollout_length = 50

    key = jax.random.key(0)
    key_reset, key_rollout = jax.random.split(key)
    state, timestep = env.reset(key_reset)

    def run_n_steps(state, key, n):
        def step_fn(carry, key):
            state, _ = carry
            action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
            new_state, timestep = env.step(state, action)
            return (new_state, timestep), new_state

        initial_carry = (state, timestep)
        _, states = jax.lax.scan(step_fn, initial_carry, jax.random.split(key, n))
        return states

    print("Starting rollout")
    states = run_n_steps(state, key_rollout, rollout_length)
    print(f"Completed {rollout_length} steps")

    state_sequence = [
        jtu.map(lambda x: x[idx], states) for idx in range(rollout_length)
    ]

    print("Creating rollout animation")
    env.animate(state_sequence, save_path=f"./animations/{VIDEO_FILENAME}")
    print("Video generated successfully")


if __name__ == "__main__":
    main()
