# custom_lunar_lander.py
import gymnasium as gym
import numpy as np

class ReasoningLunarLander(gym.Wrapper):
    """
    A wrapper for LunarLander that adds a custom reward function to encourage
    reasoning-like behavior, inspired by DeepSeek's RL approach.
    
    The wrapper adds the following reward components:
    1. Smooth landing bonus: Rewards the agent for landing with low velocity
    2. Efficient trajectory bonus: Rewards the agent for using minimal fuel
    3. Stability bonus: Rewards the agent for maintaining a stable orientation
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_shaping = None
        self.cumulative_fuel_usage = 0
        
    def reset(self, **kwargs):
        self.prev_shaping = None
        self.cumulative_fuel_usage = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track fuel usage (actions 2 and 3 are the main and side thrusters)
        if action == 2 or action == 3:
            self.cumulative_fuel_usage += 1
            
        # Add custom rewards only when episode is ending
        if terminated:
            # Extract state information
            x_pos, y_pos, x_vel, y_vel, angle, angular_vel = obs[:6]
            
            # 1. Smooth landing bonus (if landing was successful)
            if reward > 0:  # Successful landing
                vel_magnitude = np.sqrt(x_vel**2 + y_vel**2)
                if vel_magnitude < 0.1:
                    smooth_landing_bonus = 50.0
                    reward += smooth_landing_bonus
                    if "custom_rewards" not in info:
                        info["custom_rewards"] = {}
                    info["custom_rewards"]["smooth_landing_bonus"] = smooth_landing_bonus
            
            # 2. Efficient trajectory bonus (reward for using less fuel)
            if reward > 0:  # Only for successful landings
                efficiency_bonus = max(0, 30.0 - 0.1 * self.cumulative_fuel_usage)
                reward += efficiency_bonus
                if "custom_rewards" not in info:
                    info["custom_rewards"] = {}
                info["custom_rewards"]["efficiency_bonus"] = efficiency_bonus
            
            # 3. Stability bonus (reward for landing with minimal angular velocity)
            if reward > 0 and abs(angular_vel) < 0.05:
                stability_bonus = 20.0
                reward += stability_bonus
                if "custom_rewards" not in info:
                    info["custom_rewards"] = {}
                info["custom_rewards"]["stability_bonus"] = stability_bonus
                
        return obs, reward, terminated, truncated, info

def make_reasoning_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"./videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        # Only apply the wrapper for LunarLander environments
        if "LunarLander" in env_id:
            env = ReasoningLunarLander(env)
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
