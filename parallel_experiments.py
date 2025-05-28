import os
import subprocess
import argparse
import json
import time
import logging
import concurrent.futures
from datetime import datetime
from train_and_visualize import run_training_and_visualization 

def run_parallel_training(
    hyperparameter_sets,
    max_parallel=10,
    results_dir="./experiment_results"
):
    """
    Run multiple training sessions in parallel with different hyperparameters.
    
    Args:
        hyperparameter_sets: List of dictionaries, each containing hyperparameters for a run
        max_parallel: Maximum number of parallel training sessions
        results_dir: Directory to store the experiment results
    """
    # Set up logging
    os.makedirs("./logs", exist_ok=True)
    log_filename = f"./logs/parallel_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger()
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a directory to store JSON results
    json_dir = "./experiment_metadata"
    os.makedirs(json_dir, exist_ok=True)
    
    # Create a unique experiment ID
    experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment {experiment_id} with {len(hyperparameter_sets)} hyperparameter sets")
    logger.info(f"Using up to {max_parallel} parallel processes")
    logger.info(f"Log file: {log_filename}")
    
    results = []
    
    # Run training sessions in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all training jobs
        future_to_params = {}
        
        for i, params in enumerate(hyperparameter_sets):
            logger.info(f"Submitting job {i+1}/{len(hyperparameter_sets)} with params: {params}")
            
            # Extract all parameters with defaults
            custom_rewards = params.get("custom_rewards", False)
            record_best = params.get("record_best", True)
            eval_episodes = params.get("eval_episodes", 30)
            
            # Pass all parameters to the training function
            # Add any additional PPO parameters that your function supports
            future = executor.submit(
                run_training_and_visualization,
                total_timesteps=params.get("total_timesteps", 500000),
                learning_rate=params.get("learning_rate", 0.00025),
                gamma=params.get("gamma", 0.99),
                seed=params.get("seed", 42),
                custom_rewards=custom_rewards,
                eval_episodes=eval_episodes,
                record_best=record_best,
                # Include additional PPO parameters if your function supports them
                **{k: v for k, v in params.items() if k not in [
                    "total_timesteps", "learning_rate", "gamma", "seed", 
                    "custom_rewards", "eval_episodes", "record_best", 
                    "experiment_focus"
                ]}
            )
            
            future_to_params[future] = params
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
            params = future_to_params[future]
            try:
                result = future.result()
                
                if result is None:
                    logger.warning(f"Run {i+1}/{len(hyperparameter_sets)} failed")
                    results.append({
                        "run_id": i,
                        "status": "failed",
                        **params
                    })
                else:
                    logger.info(f"Run {i+1}/{len(hyperparameter_sets)} completed successfully")
                    # Add the result data to our results list
                    results.append({
                        "run_id": i,
                        "status": "success",
                        "model_path": result["model_path"],
                        "visualization_dir": result["output_dir"],
                        "mean_return": result["mean_return"],
                        "video_path": result["video_path"],
                        "log_file": result["log_file"],
                        **params
                    })
            except Exception as e:
                logger.error(f"Run {i+1}/{len(hyperparameter_sets)} raised an exception: {e}", exc_info=True)
                results.append({
                    "run_id": i,
                    "status": "error",
                    "error": str(e),
                    **params
                })
    
    # Save results to JSON file
    results_file = os.path.join(json_dir, f"{experiment_id}_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "num_runs": len(hyperparameter_sets),
            "successful_runs": sum(1 for r in results if r['status'] == 'success'),
            "failed_runs": sum(1 for r in results if r['status'] != 'success'),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Experiment completed. Results saved to {results_file}")
    logger.info(f"Successful runs: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    
    return results_file, results

def create_hyperparameter_sets():
    # Default parameters from the best performing run (run_id 15)
    default_params = {
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "num_steps": 128,
        "update_epochs": 8,  # Using 8 instead of 4 based on best results
        "num_minibatches": 4,
        "ent_coef": 0.01,
        "experiment_focus": "default_best"
    }
    
    hyperparameter_sets = []
    
    # Add the default parameters first (our new baseline from best run)
    hyperparameter_sets.append(default_params.copy())
    
    # Learning rate variations with higher update_epochs 
    for lr in [0.0001, 0.00015, 0.0003]:
        params = default_params.copy()
        params["learning_rate"] = lr
        params["experiment_focus"] = "learning_rate"
        hyperparameter_sets.append(params)
    
    # Even higher update epochs to see if performance continues to improve 
    for update_epochs in [10, 12, 16]:
        params = default_params.copy()
        params["update_epochs"] = update_epochs
        params["experiment_focus"] = "update_epochs"
        hyperparameter_sets.append(params)
    
    # Clip coefficient variations with higher update_epochs 
    for clip_coef in [0.15, 0.25, 0.3]:
        params = default_params.copy()
        params["clip_coef"] = clip_coef
        params["experiment_focus"] = "clip_coef"
        hyperparameter_sets.append(params)
    
    # Number of minibatches variations with higher update_epochs 
    for num_minibatches in [2, 8]:
        params = default_params.copy()
        params["num_minibatches"] = num_minibatches
        params["experiment_focus"] = "num_minibatches"
        hyperparameter_sets.append(params)
    
    # Fine-tuning gamma with higher update_epochs 
    for gamma in [0.995, 0.999]:
        params = default_params.copy()
        params["gamma"] = gamma
        params["experiment_focus"] = "gamma"
        hyperparameter_sets.append(params)
    
    # Fine-tuning GAE lambda with higher update_epochs 
    for gae_lambda in [0.92, 0.97]:
        params = default_params.copy()
        params["gae_lambda"] = gae_lambda
        params["experiment_focus"] = "gae_lambda"
        hyperparameter_sets.append(params)
    
    # Fine-tuning entropy coefficient with higher update_epochs 
    for ent_coef in [0.005, 0.015, 0.02]:
        params = default_params.copy()
        params["ent_coef"] = ent_coef
        params["experiment_focus"] = "ent_coef"
        hyperparameter_sets.append(params)
    
    # Combinations of promising parameters 
    # Lower learning rate with even higher update epochs
    params = default_params.copy()
    params["learning_rate"] = 0.0001
    params["update_epochs"] = 12
    params["experiment_focus"] = "lr_epochs_combo"
    hyperparameter_sets.append(params)
    
    # Higher clip with higher update epochs
    params = default_params.copy()
    params["clip_coef"] = 0.25
    params["update_epochs"] = 12
    params["experiment_focus"] = "clip_epochs_combo"
    hyperparameter_sets.append(params)
    
    return hyperparameter_sets

def generate_hyperparameter_sets(num_experiments=10, include_custom_rewards=False):
    """Generate a diverse set of hyperparameters for experimentation."""
    
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001]
    gammas = [0.95, 0.97, 0.99]
    seeds = list(range(42, 52))  # 10 different seeds
    
    hyperparameter_sets = []
    
    # Create combinations to reach desired number of experiments
    for i in range(num_experiments):
        lr_idx = i % len(learning_rates)
        gamma_idx = (i // len(learning_rates)) % len(gammas)
        seed_idx = i % len(seeds)
        
        params = {
            "learning_rate": learning_rates[lr_idx],
            "gamma": gammas[gamma_idx],
            "total_timesteps": 500000,
            "seed": seeds[seed_idx],
            "eval_episodes": 30,
            "record_best": True
        }
        
        # If we want to include custom rewards, alternate between True and False
        if include_custom_rewards:
            params["custom_rewards"] = (i % 2 == 0)
        
        hyperparameter_sets.append(params)
    
    return hyperparameter_sets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple Lunar Lander training sessions in parallel")
    parser.add_argument("--config", type=str, help="Path to JSON config file with hyperparameter sets")
    parser.add_argument("--max-parallel", type=int, default=10, help="Maximum number of parallel training sessions")
    parser.add_argument("--num-experiments", type=int, default=10, help="Number of experiments to run if no config file")
    parser.add_argument("--results-dir", type=str, default="./experiment_results", help="Directory to store results")
    parser.add_argument("--custom-rewards", action="store_true", help="Include custom rewards in generated hyperparameters")
    parser.add_argument("--ppo-tuning", action="store_true", help="Use PPO-specific hyperparameter tuning sets")
    
    args = parser.parse_args()
    
    if args.config:
        # Load hyperparameter sets from config file
        with open(args.config, 'r') as f:
            hyperparameter_sets = json.load(f)
    elif args.ppo_tuning:
        # Use our PPO-specific hyperparameter tuning sets
        hyperparameter_sets = create_hyperparameter_sets()
    else:
        # Generate hyperparameter sets
        hyperparameter_sets = generate_hyperparameter_sets(
            args.num_experiments,
            include_custom_rewards=args.custom_rewards
        )
    
    # Save the generated hyperparameter sets
    os.makedirs("./experiment_metadata", exist_ok=True)
    with open(f"./experiment_metadata/hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(hyperparameter_sets, f, indent=2)
    
    results_file, results = run_parallel_training(
        hyperparameter_sets=hyperparameter_sets,
        max_parallel=args.max_parallel,
        results_dir=args.results_dir
    )
    
    print(f"All experiments completed. Metadata saved to {results_file}")
    print(f"Successful runs: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"See log file for details: ./logs/parallel_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
