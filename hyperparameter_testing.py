import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from metrics_collection import hyperparameter_experiment, plot_hyperparameter_results

def run_all_hyperparameter_tests(base_output_dir="hyperparameter_results"):
    """
    Run a comprehensive set of hyperparameter tests for all methods
    
    Args:
        base_output_dir: Directory to save results
    """
    os.makedirs(base_output_dir, exist_ok=True)
    
    experiments = [
        # Q-Learning experiments
        {
            'method': 'q_learning',
            'param_name': 'alpha',
            'param_values': [0.01, 0.05, 0.1, 0.2, 0.5],
            'episodes': 300
        },
        {
            'method': 'q_learning',
            'param_name': 'gamma',
            'param_values': [0.8, 0.9, 0.95, 0.99],
            'episodes': 300
        },
        {
            'method': 'q_learning',
            'param_name': 'epsilon_decay',
            'param_values': [0.99, 0.995, 0.998, 0.999],
            'episodes': 300
        },
        
        # SARSA experiments
        {
            'method': 'sarsa',
            'param_name': 'alpha',
            'param_values': [0.01, 0.05, 0.1, 0.2, 0.5],
            'episodes': 300
        },
        {
            'method': 'sarsa',
            'param_name': 'gamma',
            'param_values': [0.8, 0.9, 0.95, 0.99],
            'episodes': 300
        },
        {
            'method': 'sarsa',
            'param_name': 'epsilon_decay',
            'param_values': [0.99, 0.995, 0.998, 0.999],
            'episodes': 300
        },
        
        # MCTS experiments
        {
            'method': 'mcts',
            'param_name': 'n_simulations',
            'param_values': [50, 100, 200, 500],
            'episodes': 10
        },
        {
            'method': 'mcts',
            'param_name': 'c_puct',
            'param_values': [0.5, 1.0, 1.4, 2.0, 3.0],
            'episodes': 10
        }
    ]
    
    for exp in experiments:
        print(f"\n\n{'='*80}")
        print(f"Running {exp['method']} experiment for {exp['param_name']}")
        print(f"{'='*80}")
        
        exp_dir = os.path.join(base_output_dir, f"{exp['method']}_{exp['param_name']}")
        os.makedirs(exp_dir, exist_ok=True)
        
        results = hyperparameter_experiment(
            method=exp['method'],
            param_name=exp['param_name'],
            param_values=exp['param_values'],
            num_episodes=exp['episodes'],
            episode_step_limit=50
        )
        
        with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        fig = plot_hyperparameter_results(results, exp['method'], exp['param_name'])
        fig.savefig(os.path.join(exp_dir, 'summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        for value in exp['param_values']:
            plt.figure(figsize=(12, 6))
            rewards = results[value]['rewards_history']
            window_size = min(10, len(rewards))
            if len(rewards) > window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards)
                plt.title(f'{exp["method"]} with {exp["param_name"]}={value} - Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.savefig(os.path.join(exp_dir, f'rewards_{value}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Completed experiment for {exp['method']} - {exp['param_name']}")

def analyze_best_hyperparameters(base_results_dir="hyperparameter_results"):
    """
    Analyze the results of hyperparameter tests to find optimal values
    
    Args:
        base_results_dir: Directory containing hyperparameter test results
    """
    best_params = {
        'q_learning': {},
        'sarsa': {},
        'mcts': {}
    }
    
    for method in best_params.keys():
        for param_name in ['alpha', 'gamma', 'epsilon_decay', 'n_simulations', 'c_puct']:
            exp_dir = os.path.join(base_results_dir, f"{method}_{param_name}")
            if not os.path.exists(exp_dir):
                continue
                
            with open(os.path.join(exp_dir, 'results.pkl'), 'rb') as f:
                results = pickle.load(f)
            
            param_values = sorted(results.keys())
            final_rewards = [np.mean(results[val]['rewards_history'][-10:]) for val in param_values]
            
            best_idx = np.argmax(final_rewards) if min(final_rewards) >= 0 else np.argmin(final_rewards)
            best_value = param_values[best_idx]
            best_params[method][param_name] = best_value
            
            print(f"Best {param_name} for {method}: {best_value} (reward: {final_rewards[best_idx]:.4f})")
    
    return best_params

def run_final_comparison_with_best_params(best_params, num_episodes=500, output_dir="final_comparison"):
    """
    Run a final comparison of all methods using their best hyperparameters
    
    Args:
        best_params: Dictionary of best parameter values from analyze_best_hyperparameters
        num_episodes: Number of episodes to run for each method
        output_dir: Directory to save results
    """
    from metrics_collection import main_with_metrics, compare_methods
    
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    
    # Run Q-Learning with best parameters
    if 'q_learning' in best_params and best_params['q_learning']:
        print("\nRunning Q-Learning with best parameters")
        q_params = best_params['q_learning']
        alpha = q_params.get('alpha', 0.1)
        gamma = q_params.get('gamma', 0.99)
        epsilon_decay = q_params.get('epsilon_decay', 0.995)
        
        metrics, agent, env, fig = main_with_metrics(
            method="q_learning",
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay
        )
        all_metrics['Q-Learning'] = metrics
        fig.savefig(os.path.join(output_dir, 'q_learning_performance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Run SARSA with best parameters
    if 'sarsa' in best_params and best_params['sarsa']:
        print("\nRunning SARSA with best parameters")
        s_params = best_params['sarsa']
        alpha = s_params.get('alpha', 0.1)
        gamma = s_params.get('gamma', 0.99)
        epsilon_decay = s_params.get('epsilon_decay', 0.995)
        
        metrics, agent, env, fig = main_with_metrics(
            method="sarsa",
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay
        )
        all_metrics['SARSA'] = metrics
        fig.savefig(os.path.join(output_dir, 'sarsa_performance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Run MCTS with best parameters
    if 'mcts' in best_params and best_params['mcts']:
        print("\nRunning MCTS with best parameters")
        m_params = best_params['mcts']
        n_simulations = int(m_params.get('n_simulations', 100))
        c_puct = m_params.get('c_puct', 1.4)
        
        mcts_episodes = min(num_episodes // 10, 50)
        metrics, agent, env, fig = main_with_metrics(
            method="mcts",
            num_episodes=mcts_episodes,
            n_simulations=n_simulations,
            c_puct=c_puct
        )
        all_metrics['MCTS'] = metrics
        fig.savefig(os.path.join(output_dir, 'mcts_performance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    if len(all_metrics) > 1:
        comp_fig = compare_methods(all_metrics)
        comp_fig.savefig(os.path.join(output_dir, 'methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(comp_fig)
        
        with open(os.path.join(output_dir, 'all_metrics.pkl'), 'wb') as f:
            pickle.dump(all_metrics, f)
    
    return all_metrics

if __name__ == "__main__":
    run_all_hyperparameter_tests()
    best_params = analyze_best_hyperparameters()
    run_final_comparison_with_best_params(best_params)