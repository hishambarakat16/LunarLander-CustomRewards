import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

def load_policy(path="./tmp"):
    """
    Manually load the trained policy from the saved files.
    """
    # Create a new environment to get observation and action spaces
    env = gym.make('LunarLander-v2')
    
    # Create a new policy with the same architecture
    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.0  # Dummy learning rate as we're only doing inference
    )
    
    # Load the saved policy state
    policy.load_state_dict(torch.load(f"{path}/policy.pth"))
    policy.eval()  # Set to evaluation mode
    
    return policy

def collect_expert_demonstrations(n_episodes=100, quality_threshold=200):
    """
    Collect expert demonstrations from the pre-trained model.
    """
    print("Loading expert policy...")
    policy = load_policy()
    
    # Create environment
    env = gym.make('LunarLander-v2')
    
    # Storage for demonstrations
    demonstrations = []
    
    print(f"Collecting {n_episodes} expert demonstrations...")
    for episode in range(n_episodes):
        states, actions, rewards = [], [], []
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            
            # Get expert action
            with torch.no_grad():
                # Get the action distribution
                action_dist = policy.get_distribution(obs_tensor)
                # Get the most likely action (deterministic)
                action = action_dist.mode()
                action = action.squeeze().cpu().numpy()
            
            # Store state-action pair
            states.append(obs)
            actions.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
        
        # Calculate episode return
        episode_return = sum(rewards)
        
        # Only keep successful episodes (high return)
        if episode_return > quality_threshold:
            print(f"Episode {episode}: Return = {episode_return:.2f} - Adding to demonstrations")
            demonstrations.append({
                'states': np.array(states),
                'actions': np.array(actions),
                'return': episode_return
            })
        else:
            print(f"Episode {episode}: Return = {episode_return:.2f} - Discarding")
            
    print(f"Collected {len(demonstrations)} expert demonstrations")
    return demonstrations

if __name__ == "__main__":
    # Test the collection function
    demos = collect_expert_demonstrations(n_episodes=10)
    print(f"Collected {len(demos)} demonstrations")
    for i, demo in enumerate(demos):
        print(f"Demo {i}: {len(demo['states'])} steps, return: {demo['return']:.2f}")
