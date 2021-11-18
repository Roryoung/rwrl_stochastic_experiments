import os
import copy
import pickle
import shutil

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import  DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import dmc_to_gym


def make_vec_env(env_class, env_args, exp_dir, seed=None, width=640, height=480, from_pixels=False, n_envs=None, *args, **kwargs):
    
    def make_env():
        env = dmc_to_gym(env_class(**env_args), seed=seed, width=width, height=height, from_pixels=from_pixels)
        check_env(env)
        env = Monitor(env, f"{exp_dir}/logs")
        return env

    if n_envs is None or n_envs <= 1:
        return DummyVecEnv([make_env])  
    
    return SubprocVecEnv([make_env for _ in range(n_envs)])


def make_exp_dirs(exp_dir, trial_no):
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs/trial_{trial_no}", exist_ok=True)
    os.makedirs(f"{exp_dir}/ckpt", exist_ok=True)
    os.makedirs(f"{exp_dir}/ckpt/trial_{trial_no}/", exist_ok=True)
    os.makedirs(f"{exp_dir}/ckpt/trial_{trial_no}/backup", exist_ok=True)
    os.makedirs(f"{exp_dir}/evaluation", exist_ok=True)
    os.makedirs(f"{exp_dir}/evaluation/trial_{trial_no}", exist_ok=True)
    os.makedirs(f"{exp_dir}/evaluation/summary", exist_ok=True)


def clear_exp_dirs(exp_dir):
    remove_dirs(f"{exp_dir}/logs")
    remove_dirs(f"{exp_dir}/ckpt")
    remove_dirs(f"{exp_dir}/ckpt/backup")
    remove_dirs(f"{exp_dir}/evaluation")

    make_exp_dirs(exp_dir)


def remove_dirs(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def merge_dict(a_s, b_s, path=[]):
    "merges b into a"
    a = copy.deepcopy(a_s)
    b = copy.deepcopy(b_s)

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                if b[key] == {}:
                    a[key] = b[key]
                else:
                    a[key] = merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def save_pkl_file(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_pkl_file(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def clear_line():
    print("\r" + " "*80, end="")
    print("\r", end="")