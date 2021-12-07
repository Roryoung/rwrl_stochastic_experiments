
import realworldrl_suite.environments as rwrl

from utils import merge_dict
from callbacks import LoggerCallback, SaveModelCallback
from manifests.common import get_all_consistent_agents, get_random_agent, get_ppo_agent, merge_manifest_and_agent

def get_base_manifest():
    base_manifest = {
        "exp_name": "test",
        "trial_name": "test",
        "training_mode": "skip_existing",
        
        # Environment args
        "env_class": rwrl.load,
        "env_args": {
            "domain_name": "walker",
            "task_name": "realworld_stand",
            "environment_kwargs": dict(log_safety_vars=False, flat_observation=True),
        },
        "bridge_args": {
            "n_envs": 1
        },

        # Training args
        "n_seeds": 1,
        "learn": {
            "total_timesteps": 1000 * 1,
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

        # Evaluation args
        "eval_seed": {},
        "eval_trial": {}
    }

    return base_manifest


def get_agent_manifests():
    all_agents = []
    # all_agents.append(get_ppo_agent())
    all_agents.append(get_random_agent())
    # all_agents += get_all_consistent_agents()

    return all_agents


def get_manifest():
    # noise_levels = [i/5 for i in range(6)]
    noise_levels = [0]
    # noise_levels = []

    # Get trail manifests
    manifest = [get_base_manifest()]

    manifest = merge_manifest_and_agent(manifest, get_agent_manifests())
    
    return manifest

