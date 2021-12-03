import copy

import realworldrl_suite.environments as rwrl
import matplotlib.pyplot as plt
import numpy as np

from utils import clear_line, merge_dict, make_vec_env, print_line
from callbacks import LoggerCallback, SaveModelCallback
from manifests.common import get_all_consistent_agents, get_random_agent, get_ppo_agent, merge_manifest_and_agent


def get_trial_manifest(noise):
    trial = {
        "trial_name": "range=" + str(noise).replace(".", ","),
        "env_args": {
            "noise_spec" : dict(uniform=dict(enable=bool(True*noise), observations=(0,0), actions=noise))
        },

        "eval_seed": {
            "eval_functions": [
                get_plot_actions(lambda x: x + np.random.uniform(noise[0], noise[1]))
            ],
        }
    }
    
    return trial

def get_base_manifest():
    base_manifest = {
        "exp_name": "uniform_action_noise/walker_walk",
        "training_mode": "skip_existing",
        
        # Environment args
        "env_class": rwrl.load,
        "env_args": {
            "domain_name": "walker",
            "task_name": "realworld_walk",
            "environment_kwargs": dict(log_safety_vars=False, flat_observation=True),
        },
        "bridge_args": {
            "n_envs": 8
        },

        # Training args
        "n_seeds": 3,
        "learn": {
            "total_timesteps": 1000 * 15000,
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
    all_agents.append(get_ppo_agent())
    all_agents.append(get_random_agent())
    # all_agents += get_all_consistent_agents()

    return all_agents


def get_manifest():
    noise_levels = [(-i/5, i/5) for i in range(6)]
    # noise_levels = [(-1.0,1.0)]
    # noise_levels = []

    # Get trail manifests
    manifest = [merge_dict(get_base_manifest(), get_trial_manifest(noise)) for noise in noise_levels]

    manifest = merge_manifest_and_agent(manifest, get_agent_manifests())
    
    return manifest

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