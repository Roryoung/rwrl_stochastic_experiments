import torch

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


class Random_policy(BasePolicy):
    def __init__(self, action_space, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, action_space=action_space, squash_output=squash_output, **kwargs)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:            
        random_sample = self.action_space.sample()[None, :]

        return torch.tensor(random_sample)


    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class Random(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        super(Random, self).__init__(policy=Random_policy, env=env, policy_base=Random_policy, learning_rate=0, *args, **kwargs)
        self.policy = Random_policy(action_space=env.action_space, observation_space=env.observation_space)
        

    def learn(self, *args, **kwargs):
        pass


    def save(self, *args, **kwargs):
        pass


    @classmethod
    def load(cls, save_loc, env, *args, **kwargs):
        return cls(env=env, *args, **kwargs)
        
    
    def _setup_model(self) -> None:
        return super()._setup_model()
