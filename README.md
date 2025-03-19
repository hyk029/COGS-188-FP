# Comparative Study of MCTS, Q-Learning, and SARSA in Chess

This project investigates how hyperparameter tuning influences the performance of reinforcement learning algorithms for the game of chess, focusing on Monte Carlo Tree Search (MCTS), Q-Learning, and SARSA algorithms.

## Project Overview

In this study, we compare the performance of three different reinforcement learning approaches in a simplified chess environment. We evaluate how different hyperparameter settings affect metrics such as:
- Convergence speed
- Computational efficiency
- Performance against baseline heuristic agents
- Memory usage

Our implementation addresses the enormous state space challenge of chess through custom environments and efficient data structures.

## Repository Structure

- `__init__.py` - Core implementations of agents and environments
- `run_model.py` - Run all methods with default parameters
- `hyperparameter_testing.py` - Hyperparameter tuning and analysis
- `metrics_collection.py` - Metrics collection and visualization
- `tournament.py` - Tournament evaluation against baseline agents
- `heuristic_agents.py` - Implementation of baseline agents
- `create_dataset.py` - Tools for creating chess position datasets

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- python-chess
- gym
- tqdm

Install dependencies with:
```bash
pip install numpy matplotlib python-chess gym tqdm
```

## Usage Guide

### Creating a Position Dataset (Optional)

Generate a dataset of chess positions for training:

```bash
python create_dataset.py --output chess_positions.csv --positions 1000 --balanced
```

Options:
- `--output`: Output CSV file path
- `--positions`: Number of positions to generate
- `--balanced`: Only include positions with balanced material
- `--pgn`: Path to PGN file to extract positions from
- `--min-depth`: Minimum number of moves for random positions
- `--max-depth`: Maximum number of moves for random positions

### Running Default Models

Run all three algorithms with default parameters:

```bash
python run_model.py --episodes 500 --output model_results
```

To use a position dataset:
```bash
python run_model.py --episodes 500 --output model_results --dataset chess_positions.csv
```

Options:
- `--episodes`: Number of episodes to run
- `--output`: Output directory
- `--dataset`: Path to CSV file with FEN positions
- `--balanced`: Only use balanced positions from dataset

### Hyperparameter Tuning

Run comprehensive hyperparameter tests:

```bash
python hyperparameter_testing.py
```

This script will:
1. Run all three algorithms with various hyperparameter settings
2. Save results and visualizations to `hyperparameter_results/`
3. Analyze the best hyperparameter combinations
4. Run a final comparison with optimal parameters
5. Evaluate the best models against baseline agents

## File Descriptions

### Core Files

- `__init__.py`: Contains implementations of:
  - `ChessEnv`: Standard chess environment
  - `FENDatasetChessEnv`: Environment using FEN position datasets
  - `QLearningAgent`: Q-learning implementation
  - `SARSAAgent`: SARSA implementation
  - `MCTSAgent`: Monte Carlo Tree Search implementation

### Supporting Files

- `metrics_collection.py`: Implements functions for collecting performance metrics and visualizing results
- `tournament.py`: Provides tournament functionality for evaluating agents against baselines
- `heuristic_agents.py`: Implements baseline agents (Random, Material, Positional)

## Key Findings

The detailed results of our experiments can be found in the output directories after running the provided scripts. Some key aspects we evaluate:

- How learning rate affects Q-learning and SARSA convergence
- Impact of discount factor on long-term strategy
- Effects of exploration parameters on performance
- MCTS simulation count and exploration constant optimization
- Memory usage growth patterns
- Performance against different baseline agents

## Acknowledgements

This project was developed as part of the COGS 188 course at UC San Diego by:
- Han Young Kim
- Adrian Zhu Chou
- Veeva Gathani