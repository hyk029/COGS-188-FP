import os
import pickle
import numpy as np
from metrics_collection import main_with_metrics, compare_methods
import matplotlib.pyplot as plt

def run_all_methods(num_episodes=300, output_base_dir="model_results"):
    """
    Run all three reinforcement learning methods and save results in separate folders
    
    Args:
        num_episodes: Number of episodes to run for each method
        output_base_dir: Base directory for saving results
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    methods = ["q_learning", "sarsa", "mcts"]
    
    all_metrics = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running {method.upper()}")
        print(f"{'='*80}")
        
        method_dir = os.path.join(output_base_dir, f"{method}_results")
        os.makedirs(method_dir, exist_ok=True)
        
        actual_episodes = num_episodes if method != "mcts" else min(num_episodes // 10, 50)
        
        metrics, agent, env, fig = main_with_metrics(
            method=method, 
            num_episodes=actual_episodes
        )
        
        all_metrics[method.upper()] = metrics
        
        fig.savefig(os.path.join(method_dir, f"{method}_performance.png"), dpi=300)
        plt.close(fig)
        
        with open(os.path.join(method_dir, f"{method}_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
                

        plt.figure(figsize=(12, 6))
        rewards = metrics['rewards_history']
        window_size = min(10, len(rewards))
        if len(rewards) > window_size:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_rewards)
            plt.title(f'{method.upper()} Rewards Over Time (Smoothed)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(method_dir, f"{method}_rewards.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['memory_usage'])
        plt.title(f'{method.upper()} Memory Usage Growth')
        plt.xlabel('Episode')
        plt.ylabel('Entries (Q-table or Tree)')
        plt.savefig(os.path.join(method_dir, f"{method}_memory.png"), dpi=300)
        plt.close()
        
        print(f"Results for {method} saved in '{method_dir}'")
    
    comparison_dir = os.path.join(output_base_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    comp_fig = compare_methods(all_metrics)
    comp_fig.savefig(os.path.join(comparison_dir, 'methods_comparison.png'), dpi=300)
    plt.close(comp_fig)
    
    with open(os.path.join(comparison_dir, 'all_metrics.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\nComparison results saved in '{comparison_dir}'")

    from metrics_collection import visualize_baseline_results
    baseline_fig = visualize_baseline_results(all_metrics, output_dir=comparison_dir)
    if baseline_fig:
        plt.close(baseline_fig)

    print(f"\nBaseline comparison results saved in '{comparison_dir}'")
    
    return all_metrics, comp_fig

if __name__ == "__main__":
    all_metrics, comp_fig = run_all_methods(num_episodes=300)
    
    plt.figure(comp_fig.number)
    plt.show()