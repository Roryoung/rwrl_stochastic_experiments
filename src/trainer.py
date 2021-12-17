import os
import json
import copy
from pickle import load

import numpy as np
import imageio
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from utils import clear_line, make_vec_env, make_exp_dirs, clear_exp_dirs, load_pkl_file, print_line, save_pkl_file
from callbacks import SaveBestModelStep

class Trainer():
    def __init__(self, exp_dir, env_class, env_args, model_name, model_class, model_args, bridge_args={}, trial_no=0, training_mode="collect", *args, **kwargs):
        assert training_mode in ("extend", "collect", "re-train")

        self.exp_dir = exp_dir
        self.env_class = env_class
        self.env_args = env_args
        self.bridge_args = bridge_args

        self.model_name = model_name
        self.model_class = model_class
        self.model_args = model_args

        self.trial_no = trial_no
        self.loaded = False

        trial_data = {
            "env_class": env_class,
            "env_args": env_args,
            "model": model_class,
            "model_args": model_args,
            "bridge_args": bridge_args,
        }
        
        make_exp_dirs(self.exp_dir, trial_no=self.trial_no)
        
        if not os.path.exists(f"{self.exp_dir}/trial_man.pkl"):
            save_pkl_file(trial_data, f"{self.exp_dir}/trial_man.pkl")
            training_mode = "collect"

        if training_mode == "re-train":
            # TODO: Need to restrucute as this will delete every trial each time it's called. 
            raise Exception("Not implemented yet.")
            clear_exp_dirs(self.exp_dir)
            make_exp_dirs(self.exp_dir)
            save_pkl_file(trial_data, f"{self.exp_dir}/trial_man.pkl")

        else:
            existing_trial_data = load_pkl_file(f"{self.exp_dir}/trial_man.pkl")

            if existing_trial_data != trial_data:
                raise Exception("New trial manifest does not match existing one.")
        

            if training_mode == "collect":
                try:
                    seed_info = load_pkl_file(f"{self.exp_dir}/seed_info.pkl")
                    self.trial_no = seed_info["total_trials"] + 1
                    seed_info["total_trials"] = self.trial_no
                    make_exp_dirs(self.exp_dir, trial_no=self.trial_no)

                
                except:
                    seed_info = {
                        "total_trials": 0
                    }
                
                save_pkl_file(seed_info, f"{self.exp_dir}/seed_info.pkl")
                    
                pass
            elif training_mode == "extend":
                self.loaded = True

        self.env = make_vec_env(env_class=env_class, env_args=env_args, exp_dir=exp_dir, **bridge_args)

        if self.loaded:
            self.model = self.model_class.load(path=f"{self.exp_dir}/ckpt/trial_{trial_no}/final_model.zip", env=self.env)
        else:
            self.model = self.model_class(env=self.env, tensorboard_log=f"{exp_dir}/logs/trial_{self.trial_no}", **model_args)


    def load(self, model_loc=None):
        try:
            self.model = self.model_class.load(path=f"{self.exp_dir}/ckpt/{model_loc}", env=self.env)
        except Exception as e:
            raise e
            

    def learn(self, total_timesteps=1000, callback_fns=[], *args, **kwargs):

        eval_env = make_vec_env(env_class=self.env_class, env_args=self.env_args, exp_dir=self.exp_dir, **self.bridge_args)

        best_step_callback = SaveBestModelStep(verbose=0)
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{self.exp_dir}/ckpt",
                                     eval_freq=1000, deterministic=False, render=False, verbose=0,
                                     callback_on_new_best=best_step_callback)
                             
        if os.path.exists(f"{self.exp_dir}/ckpt/best.json"):
            with open(f"{self.exp_dir}/ckpt/best.json") as f:
                best_score_info = json.load(f)
                eval_callback.best_mean_reward = float(best_score_info["score"])

        callbacks = [eval_callback]
        for callback in callback_fns:
            callback_fn = callback["callback"]
            callback_args = callback["args"]

            callbacks.append(callback_fn(trainer=self, **callback_args))
        
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="", reset_num_timesteps=False)
                
        with open(f"{self.exp_dir}/ckpt/best.json", "w+") as f:
            best_score_info = {
                "score": eval_callback.best_mean_reward,
                "trial_no": self.trial_no,
                "best_step": eval_callback.callback.best_step
            }

            json.dump(best_score_info, f, indent=4)


    def close(self):
        self.env.close()


    def evaluate(self, eval_functions=[], *args, **kwargs):
        self._eval_render(*args, **kwargs)
        self._eval_reward(*args, **kwargs)
        # self._eval_state_dist(*args, **kwargs)

        for function in eval_functions:
            function(self, *args, **kwargs)

    
    def _eval_render(self, n_steps=1000, reward_color=True, constraints_color=True, render_camera_ids=[0,1], *args, **kwargs):
        print(f"\rEval Render", end="")
        eval_bridge_args = copy.deepcopy(self.bridge_args)
        eval_bridge_args["n_envs"] = None

        eval_env = make_vec_env(env_class=self.env_class, env_args=self.env_args, exp_dir=self.exp_dir, **eval_bridge_args)
        eval_env.env_method("render_colors", reward=reward_color, constraints=constraints_color)
        eval_env.env_method("render_angles", render_camera_ids=render_camera_ids)

        images = []
        obs = eval_env.reset()
        img = eval_env.render()

        for i in range(n_steps):
            print(f"\rEval Render [{i}/{n_steps}]", end="")
            images.append(np.array(img))

            action, _ = self.model.predict(obs, deterministic=False)
            obs, _, _ ,_ = eval_env.step(action)
            img = eval_env.render()

        imageio.mimsave(f"{self.exp_dir}/evaluation/trial_{self.trial_no}/sample_trajectory.gif",images, fps=50)

        eval_env.close()
        clear_line()


    def _eval_reward(self, n_runs=10, *args, **kwargs):
        print(f"\rEval reward", end="")
        mean_reward, std_reward = evaluate_policy(self.model.policy, self.env, n_eval_episodes=n_runs, deterministic=False)
        with open(f"{self.exp_dir}/evaluation/trial_{self.trial_no}/metrics.json", "w+") as f:
            try:
                saved_metrics = json.load(f)
            except:
                saved_metrics = {}

            saved_metrics["mean_reward"] = mean_reward
            saved_metrics["std_reward"] = std_reward

            json.dump(saved_metrics, f, indent=4)

        clear_line()

    
    def _eval_state_dist(self, *args, **kwargs):
        print_line("Eval State Distribution")

        if os.path.exists(f"{self.exp_dir}/ckpt/trial_{self.trial_no}/visited_states.npy"):
            with open(f"{self.exp_dir}/ckpt/trial_{self.trial_no}/visited_states.npy", "rb") as f:
                visited_states = np.load(f)

            with open(f"{self.exp_dir}/evaluation/trial_{self.trial_no}/metrics.json", "w+") as f:
                try:
                    saved_metrics = json.load(f)
                except:
                    saved_metrics = {}

                saved_metrics["state_std"] = float(np.std(visited_states))

                json.dump(saved_metrics, f, indent=4)

        clear_line()

