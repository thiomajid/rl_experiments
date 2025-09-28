import jax
import jumanji
from jumanji.wrappers import AutoResetWrapper


def main():
    print("Hello from rl-experiments!")
    env = jumanji.make("Snake-v1")
    env = AutoResetWrapper(env)

    batch_size = 4
    rollout_length = 20
    num_actions = env.action_spec.num_values

    random_key = jax.random.key(0)
    key1, key2 = jax.random.split(random_key)

    def step_fn(state, key):
        action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
        new_state, timestep = env.step(state, action)
        return new_state, timestep

    def run_n_steps(state, key, n):
        random_keys = jax.random.split(key, n)
        state, rollout = jax.lax.scan(step_fn, state, random_keys)
        return rollout

    # Instantiate a batch of environment states
    keys = jax.random.split(key1, batch_size)
    state, timestep = jax.vmap(env.reset)(keys)

    # Collect a batch of rollouts
    keys = jax.random.split(key2, batch_size)
    rollout = jax.vmap(run_n_steps, in_axes=(0, 0, None))(state, keys, rollout_length)


if __name__ == "__main__":
    main()
