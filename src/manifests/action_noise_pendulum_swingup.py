import copy

import numpy as np
import matplotlib.pyplot as plt

import realworldrl_suite.environments as rwrl
from stable_baselines3 import PPO

from utils import merge_dict, make_vec_env, clear_line
from callbacks import LoggerCallback, SaveModelCallback
from manifests.common import get_random_agent


def get_trial_manifest(noise):
    trial = {
        "trial_name": "action_noise=" + str(noise).replace(".", ","),
        "env_args": {
            "noise_spec" : dict(gaussian=dict(enable=bool(True*noise), observations=0, actions=noise))
        }
    }
    
    return trial


def get_manifest():
    base_manifest = {
        "exp_name": "action_noise_pedulum_swingup",
        "training_mode": "extend",
        "env_class": rwrl.load,
        "env_args": {
            "domain_name": "pendulum",
            "task_name": "realworld_swingup",
            "environment_kwargs": dict(log_safety_vars=False, flat_observation=True),
        },
        "bridge_args": {
            "n_envs": 8
        },
        "model_class": PPO,
        "model_args": {
            "policy": "MlpPolicy"
        },
        "n_seeds": 3,
        "learn": {
            "total_timesteps": 1000 * 10000,
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
        "eval_seed": {
            "eval_functions": [
                plot_phase_portrait
            ] 
        },
        "eval_trial": {}
    }

    noise_levels = [0,0.2,0.4,0.6,0.8,1]
    # noise_levels = [0.6]
    # noise_levels = []

    manifest = [
        merge_dict(base_manifest, get_trial_manifest(noise)) for noise in noise_levels
    ]
    
    manifest.append(merge_dict(base_manifest, get_random_agent()))

    return manifest

def plot_phase_portrait(trainer, *args, **kwargs):
    print(f"\Phase Portait", end="")
    x,y = np.meshgrid(np.linspace(-np.pi,np.pi,30),np.linspace(-10,10,40))
    u = y
    v = -0.1*y - 9.8*np.sin(x)
    c = np.sqrt(u**2 + v**2)

    # normalize
    (u, v) = (u, v)/c

    fig, ax = plt.subplots(1, 1)

    ax.quiver(x,y,u,v,c)
    ax.axvspan(-np.pi, -np.pi+(np.pi/6), alpha=0.5, color='green')
    ax.axvspan(np.pi, np.pi-(np.pi/6), alpha=0.5, color='green')
    ax.set_xlabel("Ө")
    ax.set_ylabel("$\dfrac{dӨ}{dt}$")

    eval_bridge_args = copy.deepcopy(trainer.bridge_args)
    eval_bridge_args["n_envs"] = None

    eval_env = make_vec_env(env_class=trainer.env_class, env_args=trainer.env_args, exp_dir=trainer.exp_dir, **eval_bridge_args)

    thetas = []
    theta_dots = []

    obs = eval_env.reset()

    for i in range(1000):
        print(f"\rPhase Portait [{i}/{1000}]", end="")
        theta = np.arctan(obs[:, 0]/obs[:, 1])

        if obs[:, 1] < 0:
            theta -= np.pi/2
        else:
            theta += np.pi/2

        thetas.append(theta)
        theta_dots.append(-obs[:, 2])

        action, _ = trainer.model.predict(obs, deterministic=False)
        obs, _, _ ,_ = eval_env.step(action)

    eval_env.close()
    clear_line()

    thetas = np.concatenate(thetas)
    theta_dots = np.concatenate(theta_dots)

    sample_trajectory = np.stack([thetas, theta_dots], axis=1)

    ax.scatter(sample_trajectory[0, 0], sample_trajectory[0, 1], color="red")
    ax.plot(sample_trajectory[:, 0], sample_trajectory[:, 1])
    fig.savefig(f"{trainer.exp_dir}/evaluation/trial_{trainer.trial_no}/phase_portrait.png", bbox_inches="tight")
    plt.close()



