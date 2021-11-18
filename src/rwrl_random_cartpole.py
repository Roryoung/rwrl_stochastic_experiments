import os
import json


import realworldrl_suite.environments as rwrl
import numpy as np
from PIL import Image
import subprocess

import imageio


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import  DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import  CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


import dmc_to_gym


def eval_render(model, exp_dir, reward_color=True, constraints_color=True, render_camera_ids=[0,1]):
    model.env.env_method("render_colors", reward=reward_color, constraints=constraints_color)
    model.env.env_method("render_angles", render_camera_ids=render_camera_ids)

    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')

    for i in range(1000):
        images.append(np.array(img))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode='rgb_array')

    imageio.mimsave(f"{exp_dir}/evaluation/sample_trajectory.gif",images, fps=50)


def eval_reward(model, exp_dir, n_runs=3):
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n_runs, deterministic=True)
    with open(f"{exp_dir}/evaluation/metrics.json", "w+") as f:
        try:
            saved_metrics = json.load(f)
        except:
            saved_metrics = {}

        saved_metrics["mean_reward"] = mean_reward
        saved_metrics["std_reward"] = std_reward

        json.dump(saved_metrics, f, indent=0)



exp_name = "PPO_noise_0_deterministic"
exp_dir = f"experiments/{exp_name}"

os.makedirs(exp_dir, exist_ok=True)
os.makedirs(f"{exp_dir}/logs", exist_ok=True)
os.makedirs(f"{exp_dir}/ckpt", exist_ok=True)
os.makedirs(f"{exp_dir}/ckpt/backup", exist_ok=True)
os.makedirs(f"{exp_dir}/evaluation", exist_ok=True)


dmc_env = rwrl.load(
    # domain_name='humanoid',
    # task_name='realworld_stand',
    domain_name='cartpole',
    task_name='realworld_balance',
    # combined_challenge='easy',
    log_output=f'{exp_dir}/results.npz',
    environment_kwargs=dict(log_safety_vars=True, flat_observation=True),
    # safety_spec=dict(enable=True, observations=True, safety_coeff=0),
    # perturb_spec=dict(enable=True, scheduler="constant", param="slider_damping")
    # noise_spec=dict(gaussian=dict(enable=True, actions=0.001))
    )

env = dmc_to_gym(dmc_env, width=640, height=480, from_pixels=False)
check_env(env)

env = Monitor(env, f"{exp_dir}/logs")
env = DummyVecEnv([lambda: env])

callbacks = [
    EvalCallback(env, best_model_save_path=f"{exp_dir}/ckpt",
                             log_path=f"{exp_dir}/logs", eval_freq=1000,
                             deterministic=True, verbose=0),
    CheckpointCallback(save_freq=1000*10, save_path=f"{exp_dir}/ckpt/backup", name_prefix="")
]

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"{exp_dir}/logs/tensorboard")
model.learn(total_timesteps=1000 * 100, callback=callbacks)
model.save(f"{exp_dir}/ckpt/final")

eval_render(model, exp_dir)
eval_reward(model, exp_dir)


