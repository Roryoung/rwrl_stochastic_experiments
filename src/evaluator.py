import os
import json
import copy
import shutil

import numpy as np
import imageio
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from utils import clear_line, make_vec_env, load_pkl_file


class Evaluator():
    def __init__(self, exp_dir, env_class, env_args, model_class, model_args, bridge_args={}, *args, **kwargs):
        self.exp_dir = exp_dir
        self.env_class = env_class
        self.env_args = env_args
        self.bridge_args = bridge_args

        trial_data = {
            "env_class": env_class,
            "env_args": env_args,
            "model": model_class,
            "model_args": model_args,
            "bridge_args": bridge_args,
        }
        

        if os.path.exists(self.exp_dir):
            existing_trial_data = load_pkl_file(f"{self.exp_dir}/trial_man.pkl")

            if existing_trial_data != trial_data:
                raise Exception("New trial manifest does not match existing one.")
            
        else: 
            raise Exception("No trials to evaulate.")
        
        self.env = make_vec_env(env_class=env_class, env_args=env_args, exp_dir=exp_dir, **bridge_args)
        self.model_class = model_class


    def load(self, model_loc=None):
        return self.model_class.load(model_loc, env=self.env)


    def close(self):
        self.env.close()
        
        
    def evaluate(self, render=True, rewards=True, tensorboard=True, *args, **kwargs):
        self._eval_render(*args, **kwargs)
        self._eval_reward(*args, **kwargs)
        self._collate_tensorboard(*args, **kwargs)

    
    def _eval_render(self, n_steps=1000, reward_color=True, constraints_color=True, render_camera_ids=[0,1], *args, **kwargs):
        print(f"\rEval Render", end="")
        eval_bridge_args = copy.deepcopy(self.bridge_args)
        eval_bridge_args["n_envs"] = None

        eval_env = make_vec_env(env_class=self.env_class, env_args=self.env_args, exp_dir=self.exp_dir, **eval_bridge_args)
        eval_env.env_method("render_colors", reward=reward_color, constraints=constraints_color)
        eval_env.env_method("render_angles", render_camera_ids=render_camera_ids)

        model = self.load(f"{self.exp_dir}/ckpt/best_model")

        images = []
        obs = eval_env.reset()
        img = eval_env.render()

        for i in range(n_steps):
            print(f"\rEval Render [{i}/{n_steps}]", end="")
            images.append(np.array(img))

            action, _ = model.predict(obs, deterministic=False)
            obs, _, _ ,_ = eval_env.step(action)
            img = eval_env.render()

        imageio.mimsave(f"{self.exp_dir}/evaluation/summary/best_policy.gif",images, fps=50)

        eval_env.close()
        clear_line()


    def _eval_reward(self, n_runs=3, *args, **kwargs):
        print(f"\rEval reward", end="")

        trial_dirs = next(os.walk(f"{self.exp_dir}/ckpt"))[1]
        all_rewards = []

        for trial in trial_dirs:
            model = self.load(f"{self.exp_dir}/ckpt/{trial}/final_model")
            rewards, _ = evaluate_policy(model.policy, self.env, n_eval_episodes=n_runs, deterministic=False, return_episode_rewards=True)
            all_rewards.append(rewards)
            
        all_rewards = np.concatenate(all_rewards)
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)


        with open(f"{self.exp_dir}/evaluation/summary/metrics.json", "w+") as f:
            try:
                saved_metrics = json.load(f)
            except:
                saved_metrics = {}

            saved_metrics["mean_reward"] = mean_reward
            saved_metrics["std_reward"] = std_reward
            saved_metrics["n_runs"] = n_runs
            saved_metrics["n_trials"] = len(trial_dirs)

            json.dump(saved_metrics, f, indent=4)

        clear_line()

    
    def _collate_tensorboard(self):
        if os.path.exists(f"{self.exp_dir}/logs/trial_0/_0"):
            all_per_key = get_all_per_key(f"{self.exp_dir}/logs")
            write_summary(all_per_key, f"{self.exp_dir}/logs")


def get_all_per_key(logdir): 
    scalar_accumulators = [EventAccumulator(f"{logdir}/{i}/_0").Reload().scalars for i in os.listdir(logdir) if i != "summary" and os.path.isdir(f"{logdir}/{i}")]

    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, values_per_key)))

    return all_per_key


def write_summary(all_per_key, logdir):
    exp_dir = f"{logdir}/summary"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)


    with SummaryWriter(log_dir=exp_dir) as writer:
        layout = {}

        for key, (steps, values) in all_per_key.items():
            label, run = key.split("/")[0], "/".join(key.split("/")[1:])

            mean_values = np.mean(values, axis=0)
            std_values = np.std(values, axis=0)

            for step, (mean, std) in zip(steps, zip(mean_values, std_values)): 
                writer.add_scalar(f"{key}_mean", mean, global_step=step)
                writer.add_scalar(f"{key}_lb", mean - std, global_step=step)
                writer.add_scalar(f"{key}_ub", mean + std, global_step=step)

            if label in layout:
                layout[label][run] = ["Margin", [f"{key}_mean", f"{key}_lb", f"{key}_ub"]]
            else:
                layout[label] = {run: ["Margin", [f"{key}_mean", f"{key}_lb", f"{key}_ub"]]}


        writer.add_custom_scalars(layout)

