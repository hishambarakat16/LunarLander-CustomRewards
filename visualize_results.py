# visualize_results.py
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="TensorBoard log directory")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save plots")
    return parser.parse_args()


def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    data = {}
    
    # Extract episodic returns
    if "charts/episodic_return" in event_acc.Tags()["scalars"]:
        steps, values = [], []
        for event in event_acc.Scalars("charts/episodic_return"):
            steps.append(event.step)
            values.append(event.value)
        data["episodic_return"] = {"steps": steps, "values": values}
    
    # Extract custom rewards if available
    for tag in event_acc.Tags()["scalars"]:
        if tag.startswith("custom_rewards/"):
            reward_type = tag.split("/")[1]
            steps, values = [], []
            for event in event_acc.Scalars(tag):
                steps.append(event.step)
                values.append(event.value)
            data[reward_type] = {"steps": steps, "values": values}
    
    return data


def plot_episodic_returns(data, output_dir):
    """Plot the episodic returns."""
    if "episodic_return" not in data:
        print("No episodic return data found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw data as scatter points
    plt.scatter(
        data["episodic_return"]["steps"],
        data["episodic_return"]["values"],
        alpha=0.3,
        label="Episode Returns",
        color="blue"
    )
    
    # Add a smoothed line using rolling average
    if len(data["episodic_return"]["values"]) > 10:
        df = pd.DataFrame({
            "steps": data["episodic_return"]["steps"],
            "values": data["episodic_return"]["values"]
        })
        df = df.sort_values("steps")
        df["smooth"] = df["values"].rolling(window=10).mean()
        plt.plot(df["steps"], df["smooth"], color="red", linewidth=2, label="10-episode moving average")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Return")
    plt.title("Training Progress: Episode Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "episodic_returns.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_custom_rewards(data, output_dir):
    """Plot the custom rewards."""
    custom_rewards = [key for key in data.keys() if key != "episodic_return"]
    
    if not custom_rewards:
        print("No custom reward data found.")
        return
    
    plt.figure(figsize=(12, 8))
    
    for reward_type in custom_rewards:
        plt.scatter(
            data[reward_type]["steps"],
            data[reward_type]["values"],
            alpha=0.5,
            label=f"{reward_type}"
        )
    
    plt.xlabel("Training Steps")
    plt.ylabel("Reward Value")
    plt.title("Custom Rewards During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "custom_rewards.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_rewards(data, output_dir):
    """Plot the episodic returns with custom rewards contribution."""
    if "episodic_return" not in data:
        print("No episodic return data found.")
        return
    
    custom_rewards = [key for key in data.keys() if key != "episodic_return"]
    
    if not custom_rewards:
        print("No custom reward data found for combined plot.")
        return
    
    # Convert to DataFrame for easier manipulation
    df_main = pd.DataFrame({
        "steps": data["episodic_return"]["steps"],
        "values": data["episodic_return"]["values"]
    })
    
    # Create DataFrames for each custom reward
    dfs = {}
    for reward_type in custom_rewards:
        dfs[reward_type] = pd.DataFrame({
            "steps": data[reward_type]["steps"],
            "values": data[reward_type]["values"]
        })
    
    # Create a separate subplot for custom rewards
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot main episodic returns on top subplot
    ax1.scatter(
        df_main["steps"],
        df_main["values"],
        alpha=0.3,
        label="Total Episode Returns",
        color="blue"
    )
    
    # Add smoothed line
    if len(df_main) > 10:
        df_main = df_main.sort_values("steps")
        df_main["smooth"] = df_main["values"].rolling(window=10).mean()
        ax1.plot(df_main["steps"], df_main["smooth"], color="blue", linewidth=2, label="10-episode moving average")
    
    ax1.set_ylabel("Episode Return")
    ax1.set_title("Episode Returns with Custom Reward Components")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot custom rewards on bottom subplot
    colors = ["red", "green", "purple", "orange", "brown", "cyan"]
    for i, reward_type in enumerate(custom_rewards):
        color = colors[i % len(colors)]
        ax2.scatter(
            dfs[reward_type]["steps"],
            dfs[reward_type]["values"],
            alpha=0.3,
            label=f"{reward_type}",
            color=color
        )
        
        # Add smoothed lines for custom rewards
        if len(dfs[reward_type]) > 10:
            dfs[reward_type] = dfs[reward_type].sort_values("steps")
            dfs[reward_type]["smooth"] = dfs[reward_type]["values"].rolling(window=10).mean()
            ax2.plot(dfs[reward_type]["steps"], dfs[reward_type]["smooth"], color=color, linewidth=2)
    
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Bonus Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "combined_rewards.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    
    # Find all event files in the log directory
    event_files = glob.glob(os.path.join(args.logdir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"No TensorBoard event files found in {args.logdir}")
        return
    
    # Use the most recent event file
    latest_event_file = max(event_files, key=os.path.getctime)
    print(f"Using TensorBoard log file: {latest_event_file}")
    
    # Extract data
    data = extract_tensorboard_data(args.logdir)
    
    # Create plots
    plot_episodic_returns(data, args.output_dir)
    plot_custom_rewards(data, args.output_dir)
    plot_combined_rewards(data, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

