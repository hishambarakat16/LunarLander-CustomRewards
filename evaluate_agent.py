import argparse
import os
import time
import shutil

import gymnasium as gym
import numpy as np
import torch
from custom_lunar_lander import ReasoningLunarLander
from reasoning_ppo import Agent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--env-id", type=str, default="LunarLander-v2", help="The environment ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--record-best", action="store_true", help="Record video of the best episode")
    parser.add_argument("--custom-rewards", action="store_true", help="Use custom rewards wrapper")
    return parser.parse_args()

def make_env(env_id, seed, idx, capture_video, run_name, custom_rewards=False):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            video_folder = f"./videos/eval_{run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, video_folder)
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if custom_rewards and "LunarLander" in env_id:
            env = ReasoningLunarLander(env)
            
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def evaluate(args):
    run_name = f"{args.env_id}_eval_{int(time.time())}"
    
    # First pass: evaluate without recording to find the best episode
    env = make_env(args.env_id, args.seed, 0, False, run_name, args.custom_rewards)()
    
    # Create the agent
    dummy_env = gym.make(args.env_id)
    agent = Agent(gym.vector.SyncVectorEnv([lambda: dummy_env]))
    
    # Load the trained model
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    
    # Evaluate the agent
    episode_returns = []
    episode_lengths = []
    episode_seeds = []
    
    for episode in range(args.num_episodes):
        episode_seed = args.seed + episode
        episode_seeds.append(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs.reshape(1, -1)))
            obs, _, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            if done:
                if "episode" in info:
                    # Convert numpy arrays to float if needed
                    episode_return = float(info["episode"]["r"]) if isinstance(info["episode"]["r"], np.ndarray) else info["episode"]["r"]
                    episode_length = int(info["episode"]["l"]) if isinstance(info["episode"]["l"], np.ndarray) else info["episode"]["l"]
                    
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    print(f"Episode {episode+1}: Return = {episode_return:.2f}, Length = {episode_length}")
                    
                    # Print custom rewards if available
                    if "custom_rewards" in info:
                        print("Custom rewards:")
                        for reward_type, value in info["custom_rewards"].items():
                            value = float(value) if isinstance(value, np.ndarray) else value
                            print(f"  {reward_type} = {value:.2f}")
    
    # Find the best episode
    best_episode_idx = np.argmax(episode_returns)
    best_episode_return = episode_returns[best_episode_idx]
    best_episode_length = episode_lengths[best_episode_idx]
    best_episode_seed = episode_seeds[best_episode_idx]
    
    # Print evaluation results
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    
    print("\nEvaluation Results:")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")  # Changed to "Mean return" for consistent formatting
    print(f"Mean episode length: {mean_length:.2f}")
    print(f"Best episode: #{best_episode_idx+1}, Return = {best_episode_return:.2f}, Length = {best_episode_length}")
    
    # Record video of the best episode if requested
    video_path = None
    if args.record_best:
        print(f"\nRecording video of best episode (seed {best_episode_seed})...")
        timestamp = int(time.time())
        video_run_name = f"best_episode_{timestamp}"
        video_folder = f"./videos/eval_{video_run_name}"
        
        video_env = make_env(args.env_id, best_episode_seed, 0, True, video_run_name, args.custom_rewards)()
        
        obs, _ = video_env.reset(seed=best_episode_seed)
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs.reshape(1, -1)))
            obs, _, terminated, truncated, info = video_env.step(action.item())
            done = terminated or truncated
        
        video_env.close()
        
        # Find the video file (it should be the only .mp4 file in the directory)
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        if video_files:
            video_path = os.path.join(video_folder, video_files[0])
            print(f"Video saved to: {video_path}")  # Consistent format for easy extraction
        else:
            print("No video file found in the output directory")
    
    env.close()
    
    # Print final results in a consistent format for easy extraction
    print(f"\nFINAL RESULTS:")
    print(f"Mean return: {mean_return:.2f}")
    if video_path:
        print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
