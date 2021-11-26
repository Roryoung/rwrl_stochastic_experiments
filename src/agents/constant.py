import torch

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


class Constant_policy(BasePolicy):
    def __init__(self, policy_type, action_space, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, action_space=action_space, squash_output=squash_output, **kwargs)
        if policy_type == "low":
            self.action = action_space.low[None, :]
        elif policy_type == "high":
            self.action = action_space.high[None, :]
        elif policy_type == "zero":
            self.action = action_space.high[None, :] * 0 


    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:            
        return torch.tensor(self.action)


    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class Constant(BaseAlgorithm):
    def __init__(self, env, policy_type="zero", *args, **kwargs):
        super(Constant, self).__init__(policy=Constant_policy, env=env, policy_base=Constant_policy, learning_rate=0, *args, **kwargs)
        self.policy = Constant_policy(policy_type, action_space=env.action_space, observation_space=env.observation_space)
        self.policy_type = policy_type
        

    def learn(self, *args, **kwargs):
        pass


    def save(self, *args, **kwargs):
        pass


    @classmethod
    def load(cls, path, env, *args, **kwargs):
        return cls(env=env, *args, **kwargs)
        
    
    def _setup_model(self) -> None:
        return super()._setup_model()
