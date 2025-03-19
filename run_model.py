import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from metrics_collection import main_with_metrics, compare_methods, visualize_baseline_results

def run_all_methods(num_episodes=300, output_base_dir="model_results", debug_img_dir="debug_images"):
    """
    Run all three reinforcement learning methods and save results in separate folders
    Streamlined version with better MCTS handling and organized debug images
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    debug_img_path = os.path.join(output_base_dir, debug_img_dir)
    os.makedirs(debug_img_path, exist_ok=True)
    
    methods = ["q_learning", "sarsa", "mcts"]
    all_metrics = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running {method.upper()}")
        print(f"{'='*80}")
        
        method_dir = os.path.join(output_base_dir, f"{method}_results")
        os.makedirs(method_dir, exist_ok=True)
        
        method_debug_dir = os.path.join(debug_img_path, method)
        os.makedirs(method_debug_dir, exist_ok=True)
        
        actual_episodes = num_episodes
        if method == "mcts":
            actual_episodes = max(1, min(num_episodes // 20, 15))
            print(f"Using {actual_episodes} episodes for MCTS")
        
        if method == "mcts":
            metrics, agent, env, fig = main_with_metrics(
                method=method, 
                num_episodes=actual_episodes,
                n_simulations=40,
                c_puct=1.4,
                debug_dir=method_debug_dir
            )
        else:
            metrics, agent, env, fig = main_with_metrics(
                method=method, 
                num_episodes=actual_episodes,
                debug_dir=method_debug_dir
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
        print(f"Debug images for {method} saved in '{method_debug_dir}'")
    
    comparison_dir = os.path.join(output_base_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    valid_metrics = {}
    for method, metrics in all_metrics.items():
        if metrics and len(metrics.get('rewards_history', [])) > 0:
            valid_metrics[method] = metrics
    
    if len(valid_metrics) > 1:
        comp_fig = compare_methods(valid_metrics)
        comp_fig.savefig(os.path.join(comparison_dir, 'methods_comparison.png'), dpi=300)
        plt.close(comp_fig)
    else:
        print("Not enough valid metrics data to create comparison plots")
        comp_fig = None
    
    with open(os.path.join(comparison_dir, 'all_metrics.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\nComparison results saved in '{comparison_dir}'")

    baseline_fig = visualize_baseline_results(all_metrics, output_dir=comparison_dir)
    if baseline_fig:
        plt.close(baseline_fig)
        print(f"Baseline comparison results saved in '{comparison_dir}'")
    
    return all_metrics, comp_fig

if __name__ == "__main__":
    all_metrics, comp_fig = run_all_methods(num_episodes=500)
    
    if comp_fig:
        plt.figure(comp_fig.number)
        plt.show()