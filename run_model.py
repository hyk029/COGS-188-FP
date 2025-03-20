import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from metrics_collection import main_with_metrics, compare_methods, visualize_baseline_results

def run_all_methods(num_episodes=300, output_base_dir="model_results", use_dataset=False, dataset_path=None, balanced_only=True):
    os.makedirs(output_base_dir, exist_ok=True)
    
    methods = ["q_learning", "sarsa", "mcts"]
    all_metrics = {}
    
    fen_dataset = None
    if use_dataset and dataset_path:
        try:
            from __init__ import load_fen_positions
            print(f"Loading dataset from {dataset_path}...")
            fen_dataset = load_fen_positions(dataset_path, balanced_only=balanced_only)
            print(f"Loaded {len(fen_dataset)} FENs from dataset.")
            if not fen_dataset:
                print("No valid positions loaded, falling back to standard chess positions")
                use_dataset = False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to standard chess positions")
            use_dataset = False
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running {method.upper()}")
        print(f"{'='*80}")
        
        method_dir = os.path.join(output_base_dir, f"{method}_results")
        os.makedirs(method_dir, exist_ok=True)
        
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
                use_dataset=use_dataset,
                csv_file=dataset_path
            )
        else:
            metrics, agent, env, fig = main_with_metrics(
                method=method, 
                num_episodes=actual_episodes,
                use_dataset=use_dataset,
                csv_file=dataset_path
            )
        
        all_metrics[method.upper()] = metrics
        
        fig.savefig(os.path.join(method_dir, f"{method}_performance.png"), dpi=300)
        plt.close(fig)
        
        with open(os.path.join(method_dir, f"{method}_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
        
        if use_dataset:
            with open(os.path.join(method_dir, 'dataset_info.txt'), 'w') as f:
                f.write(f"Dataset: {dataset_path}\n")
                f.write(f"Number of positions: {len(fen_dataset) if fen_dataset else 0}\n")
                f.write(f"Balanced only: {balanced_only}\n")
        
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
    
    if use_dataset:
        with open(os.path.join(comparison_dir, 'dataset_info.txt'), 'w') as f:
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Number of positions: {len(fen_dataset) if fen_dataset else 0}\n")
            f.write(f"Balanced only: {balanced_only}\n")
    
    print(f"\nComparison results saved in '{comparison_dir}'")

    baseline_fig = visualize_baseline_results(all_metrics, output_dir=comparison_dir)
    if baseline_fig:
        plt.close(baseline_fig)
        print(f"Baseline comparison results saved in '{comparison_dir}'")
    
    return all_metrics, comp_fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run reinforcement learning methods for chess')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
    parser.add_argument('--output', type=str, default="model_results", help='Output directory')
    parser.add_argument('--dataset', type=str, default=None, help='Path to CSV file with FEN positions')
    parser.add_argument('--balanced', action='store_true', help='Only use balanced positions from dataset')
    
    args = parser.parse_args()
    
    use_dataset = args.dataset is not None
    all_metrics, comp_fig = run_all_methods(
        num_episodes=args.episodes,
        output_base_dir=args.output,
        use_dataset=use_dataset,
        dataset_path=args.dataset,
        balanced_only=args.balanced
    )
    
    if comp_fig:
        plt.figure(comp_fig.number)
        plt.show()