import copy

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from agents.random import Random
from agents.constant import Constant
from utils import merge_dict, print_line, clear_line, make_vec_env


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



def get_plot_actions(noise_fn, n_runs=10):
    def plot_actions(trainer, *args, **kwargs):
        print_line(f"Action distribution")
        fig, axs = plt.subplots(3, 6, sharex=True, gridspec_kw={"right": 6, "bottom": -3})

        eval_bridge_args = copy.deepcopy(trainer.bridge_args)
        eval_bridge_args["n_envs"] = None
        eval_env = make_vec_env(env_class=trainer.env_class, env_args=trainer.env_args, exp_dir=trainer.exp_dir, **eval_bridge_args)

        all_actions = []
        noisy_actions = []

        for i in range(n_runs):
            obs = eval_env.reset()

            for j in range(1000):
                print_line(f"Action distribution | [{i}/{n_runs}] | [{j}/1000]")


                action, _ = trainer.model.predict(obs, deterministic=False)

                all_actions.append(action)
                noisy_actions.append(noise_fn(action))

                obs, _, _ ,_ = eval_env.step(action)


        all_actions = np.concatenate(all_actions)
        noisy_actions = np.concatenate(noisy_actions)
        clipped_actions = np.clip(noisy_actions, -1, 1)
        
        for i in range(6):
            n, *_ = axs[0, i].hist(all_actions[:, i], range=(-2, 2), density=True)
            max_y = 1.1*np.max(n)
            
            axs[1, i].hist(noisy_actions[:, i], range=(-2, 2), density=True)
            axs[2, i].hist(clipped_actions[:, i], range=(-2, 2), density=True)

            for j in range(3):
                axs[j,i].set_ylim([0,max_y])

        
        axs[0, 0].set_ylabel("Action")
        axs[1, 0].set_ylabel("Noisy Action")
        axs[2, 0].set_ylabel("Clipped Noisy Action")


        fig.savefig(f"{trainer.exp_dir}/evaluation/trial_{trainer.trial_no}/action_distribution.png", bbox_inches="tight")
        plt.close()
        clear_line()


    return plot_actions