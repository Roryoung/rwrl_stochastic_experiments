
import realworldrl_suite.environments as rwrl
from stable_baselines3 import PPO

from utils import merge_dict
from callbacks import LoggerCallback, SaveModelCallback
from manifests.common import get_random_agent, get_ppo_agent, merge_manifest_and_agent


def get_trial_manifest(noise):
    trial = {
        "trial_name": "action_noise=" + str(noise).replace(".", ","),
        "env_args": {
            "noise_spec" : dict(gaussian=dict(enable=bool(True*noise), observations=0, actions=noise))
        }
    }
    
    return trial


def get_base_manifest():
    base_manifest = {
        "exp_name": "gaussian_action_noise/cartpole_swingup",
        "training_mode": "skip_existing",
        "env_class": rwrl.load,
        "env_args": {
            "domain_name": "cartpole",
            "task_name": "realworld_swingup",
            "environment_kwargs": dict(log_safety_vars=False, flat_observation=True),
        },
        "bridge_args": {
            "n_envs": 8
        },
        "n_seeds": 5,
        "learn": {
            "total_timesteps": 1000 * 1500,
            "callback_fns": [
                {
                    "callback": LoggerCallback,
                    "args": {}
                },
                {
                    "callback": SaveModelCallback,
                    "args": {
                        "backup_frequency": 1000 * 500
                    }
                }
            ]
        },
        "eval_seed": {},
        "eval_trial": {}
    }

    return base_manifest


def get_agent_manifests():
    all_agents = []
    all_agents.append(get_ppo_agent())
    all_agents.append(get_random_agent())
    # all_agents += get_all_consistent_agents()
    return all_agents


def get_manifest():
    noise_levels = [0,0.2,0.4,0.6,0.8,1]
    # noise_levels = [0]
    # noise_levels = []

    # Get trail manifests
    manifest = [merge_dict(get_base_manifest(), get_trial_manifest(noise)) for noise in noise_levels]

    manifest = merge_manifest_and_agent(manifest, get_agent_manifests())
    
    return manifest

