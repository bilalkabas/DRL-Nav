from cgi import test
import os
import gym
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from scripts.network import NatureCNN


# Load train environment configs
with open('scripts/env_config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Model name
model_name = "best_model_" + config["test_mode"]

# Determine input image shape
image_shape = (50,50,1) if config["test_mode"]=="depth" else (50,50,3)

# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:test-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=image_shape,
        # Train and test envs shares same config for the test
        env_config=env_config["TrainEnv"],          
        input_mode=config["test_mode"],
        test_mode=config["test_type"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

policy_kwargs = dict(features_extractor_class=NatureCNN)

# Load an existing model
model = PPO.load(
    env=env,
    path=os.path.join("saved_policy", model_name),
    policy_kwargs=policy_kwargs
)

# Run the trained policy
obs = env.reset()
for i in range(2300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, info = env.step(action)
