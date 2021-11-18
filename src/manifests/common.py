from agents.random import Random


def get_random_agent():
    random_spec = {
        "trial_name": "Random",
        "model_class": Random,
        "model_args": {},
        "bridge_args": {
            "n_envs": 1
        },
        "learn": {
            "total_timesteps": 0,
            "callback_fns": []
        }
    }

    return random_spec