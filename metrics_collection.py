import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from __init__ import ChessEnv, FENDatasetChessEnv, QLearningAgent, SARSAAgent, MCTSAgent, load_fen_positions

def run_episodes_with_metrics(env, agent, num_episodes=1000, method="q_learning", episode_step_limit=100):
    metrics = {
        'rewards_history': [],
        'episode_lengths': [],
        'illegal_move_counts': [],
        'checkmate_counts': [],
        'q_value_evolution': [],
        'epsilon_history': [],
        'time_per_episode': [],
        'memory_usage': []
    }
    
    checkmate_count = 0
    
    for ep in range(num_episodes):
        start_time = time.time()
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        illegal_moves = 0
        
        if method == "sarsa":
            action = agent.choose_action(obs, env)
            
        if hasattr(agent, 'epsilon'):
            metrics['epsilon_history'].append(agent.epsilon)
            
        while not done and steps < episode_step_limit:
            steps += 1
            
            if method == "q_learning":
                action = agent.choose_action(obs, env)
                move = env._decode_action(action)
                if move not in env.board.legal_moves:
                    illegal_moves += 1
                
                next_obs, reward, done, _ = env.step(action)
                agent.update(obs, action, reward, next_obs, done, env)
                obs = next_obs
                total_reward += reward
                
            elif method == "sarsa":
                next_obs, reward, done, _ = env.step(action)
                
                move = env._decode_action(action)
                if move not in env.board.legal_moves:
                    illegal_moves += 1
                    
                if not done:
                    next_action = agent.choose_action(next_obs, env)
                else:
                    next_action = None
                agent.update(obs, action, reward, next_obs, next_action, done)
                obs = next_obs
                action = next_action
                total_reward += reward
                
            elif method == "mcts":
                action = agent.choose_action(env)
                
                move = env._decode_action(action)
                if move not in env.board.legal_moves:
                    illegal_moves += 1
                    
                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                total_reward += reward
            
        metrics['rewards_history'].append(total_reward)
        metrics['episode_lengths'].append(steps)
        metrics['illegal_move_counts'].append(illegal_moves)
        metrics['time_per_episode'].append(time.time() - start_time)
        
        if env.board.is_checkmate():
            checkmate_count += 1
        metrics['checkmate_counts'].append(checkmate_count)
        
        if method in ["q_learning", "sarsa"] and hasattr(agent, 'q_table') and agent.q_table:
            avg_q = sum(agent.q_table.values()) / max(len(agent.q_table), 1)
            metrics['q_value_evolution'].append(avg_q)
            metrics['memory_usage'].append(len(agent.q_table))
        
        if method == "mcts" and hasattr(agent, 'tree'):
            metrics['memory_usage'].append(len(agent.tree))
            
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes}, Reward={total_reward:.2f}, Steps={steps}, Time={metrics['time_per_episode'][-1]:.3f}s")
            if method in ["q_learning", "sarsa"]:
                print(f"  Epsilon: {agent.epsilon:.4f}, Q-table size: {len(agent.q_table)}")
            elif method == "mcts":
                print(f"  Tree size: {len(agent.tree)}")
    
    return metrics

def analyze_performance(metrics_dict, method_name):
    """
    Analyze and plot performance metrics
    
    Args:
        metrics_dict: Dictionary containing metrics from run_episodes_with_metrics
        method_name: String name of the method (q_learning, sarsa, mcts)
    """
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Performance Metrics for {method_name}', fontsize=16)
    
    rewards = metrics_dict['rewards_history']
    window_size = min(20, len(rewards))
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    axs[0, 0].plot(smoothed_rewards)
    axs[0, 0].set_title('Rewards (Smoothed)')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    
    axs[0, 1].plot(metrics_dict['episode_lengths'])
    axs[0, 1].set_title('Episode Lengths')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    
    axs[1, 0].plot(metrics_dict['checkmate_counts'])
    axs[1, 0].set_title('Cumulative Checkmates')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Count')
    
    axs[1, 1].plot(metrics_dict['illegal_move_counts'])
    axs[1, 1].set_title('Illegal Moves per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Count')
    
    axs[2, 0].plot(metrics_dict['memory_usage'])
    axs[2, 0].set_title('Memory Usage (Q-table or Tree Size)')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Entries')
    
    axs[2, 1].plot(metrics_dict['time_per_episode'])
    axs[2, 1].set_title('Computation Time per Episode')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Seconds')
    
    if 'epsilon_history' in metrics_dict and metrics_dict['epsilon_history']:
        fig_extra, axs_extra = plt.subplots(1, 2, figsize=(15, 5))
        fig_extra.suptitle(f'Additional Metrics for {method_name}', fontsize=16)
        
        axs_extra[0].plot(metrics_dict['epsilon_history'])
        axs_extra[0].set_title('Exploration Rate (Epsilon)')
        axs_extra[0].set_xlabel('Episode')
        axs_extra[0].set_ylabel('Epsilon')
        
        if 'q_value_evolution' in metrics_dict and metrics_dict['q_value_evolution']:
            axs_extra[1].plot(metrics_dict['q_value_evolution'])
            axs_extra[1].set_title('Average Q-Value Evolution')
            axs_extra[1].set_xlabel('Episode')
            axs_extra[1].set_ylabel('Average Q-Value')
    
    plt.tight_layout()
    plt.close(fig_extra)

    return fig

def main_with_metrics(method="q_learning", use_dataset=False, csv_file=None, 
                    num_episodes=500, episode_step_limit=50, 
                    alpha=0.1, gamma=0.99, epsilon_decay=0.995,
                    n_simulations=100, c_puct=1.4):
    
    if method not in ["q_learning", "sarsa", "mcts"]:
        print("Usage: python final_chess.py [q_learning|sarsa|mcts]")
        return None
    
    if use_dataset and csv_file:
        try:
            import csv
            print(f"Loading dataset from {csv_file}...")
            fen_dataset = load_fen_positions(csv_file, balanced_only=True)
            print(f"Loaded {len(fen_dataset)} FENs from dataset.")
            env = FENDatasetChessEnv(fen_dataset, max_steps=episode_step_limit)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            env = ChessEnv(max_steps=episode_step_limit)
    else:
        env = ChessEnv(max_steps=episode_step_limit)
    
    if method == "q_learning":
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.01, epsilon_decay=epsilon_decay)
        metrics = run_episodes_with_metrics(env, agent, num_episodes=num_episodes, method="q_learning", episode_step_limit=episode_step_limit)
    
    elif method == "sarsa":
        agent = SARSAAgent(alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_min=0.01, epsilon_decay=epsilon_decay)
        metrics = run_episodes_with_metrics(env, agent, num_episodes=num_episodes, method="sarsa", episode_step_limit=episode_step_limit)

    elif method == "mcts":
        agent = MCTSAgent(n_simulations=n_simulations, c_puct=c_puct)
        mcts_episodes = min(num_episodes // 10, 50)
        metrics = run_episodes_with_metrics(env, agent, num_episodes=mcts_episodes, method="mcts", episode_step_limit=episode_step_limit)
    
    fig = analyze_performance(metrics, method)
    
    return metrics, agent, env, fig

def compare_methods(metrics_dict):
    """
    Compare metrics from different methods side by side
    
    Args:
        metrics_dict: Dictionary with method names as keys and metrics dictionaries as values
    """
    methods = list(metrics_dict.keys())
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Method Comparison', fontsize=16)
    
    ax = axs[0, 0]
    for method in methods:
        rewards = metrics_dict[method]['rewards_history']
        window_size = min(20, len(rewards))
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax.plot(smoothed_rewards, label=method)
    ax.set_title('Rewards (Smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    
    ax = axs[0, 1]
    for method in methods:
        episode_lengths = metrics_dict[method]['episode_lengths']
        window_size = min(20, len(episode_lengths))
        smoothed_lengths = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax.plot(smoothed_lengths, label=method)
    ax.set_title('Episode Lengths (Smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    
    ax = axs[1, 0]
    for method in methods:
        times = metrics_dict[method]['time_per_episode']
        ax.plot(times, label=method)
    ax.set_title('Computation Time per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Seconds')
    ax.legend()
    
    ax = axs[1, 1]
    for method in methods:
        checkmates = metrics_dict[method]['checkmate_counts']
        episodes = list(range(1, len(checkmates) + 1))
        checkmate_rate = [c / e for c, e in zip(checkmates, episodes)]
        ax.plot(checkmate_rate, label=method)
    ax.set_title('Checkmate Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.legend()
    
    plt.tight_layout()

    return fig

def hyperparameter_experiment(method, param_name, param_values, num_episodes=200, episode_step_limit=50):
    """
    Run experiments with different hyperparameter values
    
    Args:
        method: Algorithm to test ("q_learning", "sarsa", "mcts")
        param_name: Name of the parameter to vary
        param_values: List of values to test for the parameter
        num_episodes: Number of episodes per experiment
        episode_step_limit: Maximum steps per episode
    
    Returns:
        Dictionary of metrics for each parameter value
    """
    results = {}
    
    for value in param_values:
        print(f"\nRunning {method} with {param_name}={value}")
        
        env = ChessEnv(max_steps=episode_step_limit)
        
        if method == "q_learning":
            if param_name == "alpha":
                agent = QLearningAgent(alpha=value, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
            elif param_name == "gamma":
                agent = QLearningAgent(alpha=0.1, gamma=value, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
            elif param_name == "epsilon_decay":
                agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=value)
            else:
                print(f"Unknown parameter: {param_name}")
                return None
                
            metrics = run_episodes_with_metrics(env, agent, num_episodes=num_episodes, method="q_learning", episode_step_limit=episode_step_limit)
        
        elif method == "sarsa":
            if param_name == "alpha":
                agent = SARSAAgent(alpha=value, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
            elif param_name == "gamma":
                agent = SARSAAgent(alpha=0.1, gamma=value, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
            elif param_name == "epsilon_decay":
                agent = SARSAAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=value)
            else:
                print(f"Unknown parameter: {param_name}")
                return None
                
            metrics = run_episodes_with_metrics(env, agent, num_episodes=num_episodes, method="sarsa", episode_step_limit=episode_step_limit)
        
        elif method == "mcts":
            if param_name == "n_simulations":
                agent = MCTSAgent(n_simulations=int(value), c_puct=1.4)
            elif param_name == "c_puct":
                agent = MCTSAgent(n_simulations=100, c_puct=value)
            else:
                print(f"Unknown parameter: {param_name}")
                return None
                
            mcts_episodes = min(num_episodes // 5, 40)
            metrics = run_episodes_with_metrics(env, agent, num_episodes=mcts_episodes, method="mcts", episode_step_limit=episode_step_limit)
        
        results[value] = metrics
    
    return results

def plot_hyperparameter_results(results, method, param_name):
    """
    Visualize results from hyperparameter experiments
    
    Args:
        results: Dictionary with parameter values as keys and metrics as values
        method: Algorithm name
        param_name: Name of the parameter that was varied
    """
    param_values = sorted(results.keys())
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Effect of {param_name} on {method} Performance', fontsize=16)
    
    ax = axs[0, 0]
    final_rewards = [np.mean(results[val]['rewards_history'][-10:]) for val in param_values]
    ax.plot(param_values, final_rewards, 'o-')
    ax.set_title('Final Average Reward (last 10 episodes)')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Reward')
    
    ax = axs[0, 1]
    convergence_speeds = []
    for val in param_values:
        rewards = results[val]['rewards_history']
        window_size = min(5, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        target = 0.9 * max(smoothed) if max(smoothed) < 0 else 0
        try:
            conv_ep = next(i for i, r in enumerate(smoothed) if r >= target)
            convergence_speeds.append(conv_ep)
        except StopIteration:
            convergence_speeds.append(len(smoothed))
            
    ax.plot(param_values, convergence_speeds, 'o-')
    ax.set_title('Convergence Speed (episodes)')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Episodes')
    
    ax = axs[1, 0]
    avg_times = [np.mean(results[val]['time_per_episode']) for val in param_values]
    ax.plot(param_values, avg_times, 'o-')
    ax.set_title('Average Computation Time per Episode')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Seconds')
    
    ax = axs[1, 1]
    memory_usage = [results[val]['memory_usage'][-1] for val in param_values]
    ax.plot(param_values, memory_usage, 'o-')
    ax.set_title('Final Memory Usage (Q-table or Tree Size)')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Entries')
    
    plt.tight_layout()
    
    return fig