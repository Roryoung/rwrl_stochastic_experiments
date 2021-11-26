from stable_baselines3 import PPO

from agents.random import Random
from agents.constant import Constant
from utils import merge_dict


def merge_manifest_and_agent(manifest_list, agent_list):
    all_manifests = []

    for manifest in manifest_list:
        for agent in agent_list:
            all_manifests.append(merge_dict(manifest, agent))

    return all_manifests

def get_ppo_agent():
    ppo_spec = {
        "model_name": "PPO",
        "model_class": PPO,
        "model_args": {
            "policy": "MlpPolicy"
        }
    }

    return ppo_spec


def get_random_agent():
    random_spec = {
        "model_name": "Random",
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


def get_consistent_agent(policy_type):
    consistent_spec = {
        "model_name": f"Constant_{policy_type}",
        "model_class": Constant,
        "model_args": {
            "policy_type": policy_type
        },
        "bridge_args": {
            "n_envs": 1
        },
        "learn": {
            "total_timesteps": 0,
            "callback_fns": []
        }
    }

    return consistent_spec


def get_all_consistent_agents():
    return [get_consistent_agent(policy_type) for policy_type in ["low", "high", "zero"]]
