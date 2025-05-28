import os
import subprocess
import re
import time
import logging
from datetime import datetime

def run_training_and_visualization(
    total_timesteps=500000,
    learning_rate=0.0001,
    gamma=0.99,
    seed=42,
    custom_rewards=False,
    eval_episodes=30,
    record_best=True,
    gae_lambda=0.95,      # This parameter is recognized
    clip_coef=0.2,        # Changed from clip_range to clip_coef
    update_epochs=4,      # Changed from n_epochs to update_epochs
    ent_coef=0.01,        # This is recognized but default is 0.01
    vf_coef=0.5,          # This is recognized
    **kwargs
):
    """
    Run the full training, evaluation, and visualization pipeline.
    
    Args:
        total_timesteps: Total timesteps for training
        learning_rate: Learning rate for the PPO algorithm
        gamma: Discount factor
        seed: Random seed
        custom_rewards: Whether to use custom rewards
        eval_episodes: Number of episodes for evaluation
        record_best: Whether to record the best episode
        gae_lambda: GAE lambda parameter for PPO
        clip_coef: PPO clipping coefficient (renamed from clip_range)
        update_epochs: Number of epochs to update policy (renamed from n_epochs)
        ent_coef: Entropy coefficient for the loss calculation
        vf_coef: Value function coefficient for the loss calculation
        
    Returns:
        Dictionary with paths to results and metrics
    """
    # Set up logging
    os.makedirs("./logs/train_eval_visual", exist_ok=True)
    
    reward_type = "custom" if custom_rewards else "standard"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/train_eval_visual/run_{reward_type}_lr{learning_rate}_gamma{gamma}_seed{seed}_{timestamp}.log"
    
    logger = logging.getLogger(f"train_viz_{timestamp}")
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    logger.info(f"Starting training with learning_rate={learning_rate}, gamma={gamma}, seed={seed}, gae_lambda={gae_lambda}")
    logger.info(f"Log file: {log_file}")
    
    # Prepare training command with the correct parameter names
    train_cmd = [
        "python", "reasoning_ppo.py",
        "--env-id", "LunarLander-v2",
        "--total-timesteps", str(total_timesteps),
        "--learning-rate", str(learning_rate),
        "--gamma", str(gamma),
        "--seed", str(seed),
        "--gae-lambda", str(gae_lambda),
        "--clip-coef", str(clip_coef),
        "--update-epochs", str(update_epochs),
        "--ent-coef", str(ent_coef),
        "--vf-coef", str(vf_coef)
    ]
    
    if custom_rewards:
        train_cmd.append("--custom-rewards")
    
    logger.info(f"Executing: {' '.join(train_cmd)}")
    
    # Run training script and capture output
    try:
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_text = ""
        # Read and log output line by line
        for line in process.stdout:
            output_text += line
            logger.info(f"TRAINING: {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Training failed with return code {process.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None
    
    # Extract model path from output
    model_path_match = re.search(r"Model saved to (.+)", output_text)
    if not model_path_match:
        logger.error("Could not find model path in output")
        return None
        
    model_path = model_path_match.group(1).strip()
    logger.info(f"Found model path: {model_path}")
    
    # Run evaluation script
    eval_cmd = [
        "python", " .py",
        "--model-path", model_path,
        "--env-id", "LunarLander-v2",
        "--num-episodes", str(eval_episodes)
    ]
    
    if record_best:
        eval_cmd.append("--record-best")
        
    logger.info(f"Executing: {' '.join(eval_cmd)}")
    
    try:
        process = subprocess.Popen(
            eval_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        eval_output = ""
        # Read and log output line by line
        for line in process.stdout:
            eval_output += line
            logger.info(f"EVALUATION: {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Evaluation failed with return code {process.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return None
    
    # Extract mean return from evaluation output
    # Try different possible patterns for mean return
    mean_return = None
    mean_return_patterns = [
        r"Mean return: ([-+]?\d*\.\d+|\d+)",
        r"Average return: ([-+]?\d*\.\d+|\d+)",
        r"Mean episode reward: ([-+]?\d*\.\d+|\d+)",
        r"Average episode reward: ([-+]?\d*\.\d+|\d+)",
        r"Mean reward: ([-+]?\d*\.\d+|\d+)"
    ]

    for pattern in mean_return_patterns:
        mean_return_match = re.search(pattern, eval_output)
        if mean_return_match:
            mean_return = float(mean_return_match.group(1))
            logger.info(f"Mean return: {mean_return}")
            break

    if mean_return is None:
        # If we still couldn't find it, log the issue and search for any number after "return" or "reward"
        logger.warning("Could not find mean return using standard patterns")
        general_pattern = r"(?:return|reward|score).*?([-+]?\d*\.\d+|\d+)"
        general_match = re.search(general_pattern, eval_output, re.IGNORECASE)
        if general_match:
            mean_return = float(general_match.group(1))
            logger.info(f"Found potential mean return: {mean_return}")
        else:
            logger.error("Could not find any mean return value in evaluation output")

    # Extract video path from evaluation output
    # Try different possible patterns for video path
    video_path = None
    video_path_patterns = [
        r"Video saved to: (.+)",
        r"Video saved at: (.+)",
        r"Video recorded to: (.+)",
        r"Video recorded at: (.+)",
        r"Saved video to: (.+)",
        r"Saved video at: (.+)"
    ]

    for pattern in video_path_patterns:
        video_path_match = re.search(pattern, eval_output)
        if video_path_match:
            video_path = video_path_match.group(1).strip()
            logger.info(f"Video path: {video_path}")
            break

    if video_path is None:
        # If we still couldn't find it, look for lines containing "video" and a path-like string
        logger.warning("Could not find video path using standard patterns")
        for line in eval_output.splitlines():
            if "video" in line.lower() and ("/" in line or "\\" in line or "." in line):
                # This line might contain the video path
                path_pattern = r"(?:\/|\\|\w:)[\w\/\\.-]+"
                path_match = re.search(path_pattern, line)
                if path_match:
                    video_path = path_match.group(0).strip()
                    logger.info(f"Found potential video path: {video_path}")
                    break
        
        if video_path is None:
            logger.warning("Could not find any video path in evaluation output")
    
    # Extract run name from model path
    run_name = os.path.basename(model_path).replace(".pt", "")
    
    # Find tensorboard log directory
    tb_log_dir = f"runs/{run_name}"
    logger.info(f"Tensorboard log directory: {tb_log_dir}")
    
    # Create output directory for visualizations
    output_dir = f"./visualizations/{run_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created visualization directory: {output_dir}")
    
    # Call the visualization script
    viz_cmd = [
        "python", "visualize_results.py",
        "--logdir", tb_log_dir,
        "--output-dir", output_dir
    ]
    
    logger.info(f"Executing: {' '.join(viz_cmd)}")
    
    try:
        process = subprocess.Popen(
            viz_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read and log output line by line
        for line in process.stdout:
            logger.info(f"VISUALIZATION: {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Visualization failed with return code {process.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return None
    
    logger.info(f"Training, evaluation, and visualization completed successfully")
    
    # Return paths and metrics
    return {
        "model_path": model_path,
        "output_dir": output_dir,
        "mean_return": mean_return,
        "video_path": video_path,
        "log_file": log_file
    }



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training, evaluation, and visualization")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--custom-rewards", action="store_true", help="Use custom rewards")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Number of episodes for evaluation")
    parser.add_argument("--no-record", action="store_true", help="Don't record best episode")
    
    # Add PPO-specific arguments with correct parameter names
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of epochs for policy update")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    
    args = parser.parse_args()
    
    result = run_training_and_visualization(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        custom_rewards=args.custom_rewards,
        eval_episodes=args.eval_episodes,
        record_best=not args.no_record,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        update_epochs=args.update_epochs,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef
    )
    
    if result:
        print(f"Training and visualization completed successfully!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Visualizations saved to: {result['output_dir']}")
        print(f"Mean return: {result['mean_return']}")
        if result['video_path']:
            print(f"Video saved to: {result['video_path']}")
        print(f"Log file: {result['log_file']}")
    else:
        print("Training or visualization failed. Check logs for details.")