import gym
import numpy as np
import random
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import csv
from gym import spaces
from collections import deque

def load_fen_positions(csv_file, balanced_only=True):
    fen_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)  
        for row in reader:
            fen = row['fen']
            board = chess.Board(fen)
            if not balanced_only or is_material_balanced(board):
                fen_list.append(fen)
    return fen_list

def is_material_balanced(board):
    piece_values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
    balance = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            sign = 1 if piece.color == chess.WHITE else -1
            balance += sign * piece_values.get(piece.piece_type, 0)
    return abs(balance) <= 1  

class ChessEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.board = chess.Board()
        self.max_steps = max_steps
        self.current_step_count = 0
        self.observation_space = spaces.Box(low=-6, high=6, shape=(64,), dtype=np.int8)
        self.action_space = spaces.Discrete(64 * 64)

    def reset(self):
        self.board.reset()
        self.current_step_count = 0
        return self._get_observation()

    def step(self, action):
        self.current_step_count += 1
        move = self._decode_action(action)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            legal = list(self.board.legal_moves)
            if legal:
                self.board.push(random.choice(legal))
        done = self.board.is_game_over() or (self.current_step_count >= self.max_steps)
        reward = self._get_reward(done)
        return self._get_observation(), reward, done, {}

    def _get_reward(self, done):
        if not done:
            return 0.0
        if self.board.is_checkmate():
            return -1.0
        return 0.0

    def _get_observation(self):
        obs = np.zeros(64, dtype=np.int8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                val = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
                obs[i] = val
        return obs

    def _decode_action(self, action_idx):
        from_sq = action_idx // 64
        to_sq   = action_idx % 64
        return chess.Move(from_sq, to_sq)

class FENDatasetChessEnv(gym.Env):
    def __init__(self, fen_list, max_steps=100):
        super().__init__()
        self.fen_list = fen_list
        self.max_steps = max_steps
        self.board = chess.Board()
        self.current_step_count = 0

        self.observation_space = spaces.Box(low=-6, high=6, shape=(64,), dtype=np.int8)
        self.action_space = spaces.Discrete(64 * 64)

    def reset(self):
        fen = random.choice(self.fen_list)
        self.board.set_fen(fen)
        self.current_step_count = 0
        return self._get_observation()

    def step(self, action):
        self.current_step_count += 1
        move = self._decode_action(action)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            legal = list(self.board.legal_moves)
            if legal:
                self.board.push(random.choice(legal))
        done = self.board.is_game_over() or (self.current_step_count >= self.max_steps)
        reward = self._get_reward(done)
        return self._get_observation(), reward, done, {}

    def _get_reward(self, done):
        if not done:
            return 0.0
        if self.board.is_checkmate():
            return -1.0
        return 0.0

    def _get_observation(self):
        obs = np.zeros(64, dtype=np.int8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                val = piece.piece_type if piece.color else -piece.piece_type
                obs[i] = val
        return obs

    def _decode_action(self, action_idx):
        from_sq = action_idx // 64
        to_sq   = action_idx % 64
        return chess.Move(from_sq, to_sq)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def set_q(self, state_key, action, value):
        self.q_table[(state_key, action)] = value

    def choose_action(self, state, env):
        state_key = self._state_to_key(state)
        
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                return env.action_space.sample()
                
            best_action = None
            best_q_val = -float('inf')
            
            for move in legal_moves:
                action_idx = self._encode_action(move)
                q_val = self.get_q(state_key, action_idx)
                if q_val > best_q_val:
                    best_q_val = q_val
                    best_action = action_idx
            
            return best_action if best_action is not None else env.action_space.sample()

    def update(self, old_state, action, reward, new_state, done, env):
        old_key = self._state_to_key(old_state)
        new_key = self._state_to_key(new_state)

        old_q = self.get_q(old_key, action)
        
        if done:
            td_target = reward
        else:
            best_next_q = -float('inf')
            for m in env.board.legal_moves:
                next_a = self._encode_action(m)
                best_next_q = max(best_next_q, self.get_q(new_key, next_a))
            td_target = reward + self.gamma * best_next_q

        updated_q = old_q + self.alpha * (td_target - old_q)
        self.set_q(old_key, action, updated_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _state_to_key(self, state):
        return tuple(state.tolist())

    def _encode_action(self, move):
        return move.from_square * 64 + move.to_square

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def set_q(self, state_key, action, value):
        self.q_table[(state_key, action)] = value

    def choose_action(self, state, env):
        state_key = self._state_to_key(state)
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                return env.action_space.sample()
            
            best_action = None
            best_q_val = -float('inf')
            for move in legal_moves:
                a_idx = self._encode_action(move)
                q_val = self.get_q(state_key, a_idx)
                if q_val > best_q_val:
                    best_q_val = q_val
                    best_action = a_idx
            return best_action if best_action is not None else env.action_space.sample()

    def update(self, old_state, action, reward, new_state, new_action, done):

        old_key = self._state_to_key(old_state)
        new_key = self._state_to_key(new_state)

        old_q = self.get_q(old_key, action)
        if done:
            td_target = reward
        else:
            new_q = self.get_q(new_key, new_action)
            td_target = reward + self.gamma * new_q

        updated_q = old_q + self.alpha * (td_target - old_q)
        self.set_q(old_key, action, updated_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _state_to_key(self, state):
        return tuple(state.tolist())

    def _encode_action(self, move):
        return move.from_square * 64 + move.to_square

class MCTSAgent:
    def __init__(self, n_simulations=100, c_puct=1.4):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.tree = {}  

    def choose_action(self, env):
        state_fen = env.board.fen()
        
        for _ in range(self.n_simulations):
            self._simulate(env.board.copy())
        
        root_node = self.tree.get(state_fen, None)
        if root_node is None or not root_node["children"]:
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                return env.action_space.sample()
            return self._encode_action(random.choice(legal_moves))
        
        best_move, best_stats = None, None
        for move, child_fen in root_node["children"].items():
            child_node = self.tree.get(child_fen, None)
            if child_node is None:
                continue
            if best_stats is None or child_node["N"] > best_stats["N"]:
                best_move = move
                best_stats = child_node
        
        if best_move is None:
            legal_moves = list(env.board.legal_moves)
            if not legal_moves:
                return env.action_space.sample()
            return self._encode_action(random.choice(legal_moves))
        
        return self._encode_action(best_move)

    def _simulate(self, board):
        visited = []
        fen_history = []
        
        while True:
            fen = board.fen()
            node = self.tree.setdefault(fen, {"N": 0, "W": 0.0, "children": {}})
            visited.append(fen)

            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            if not node["children"]:
                for move in legal_moves:
                    board.push(move)
                    child_fen = board.fen()
                    node["children"][move] = child_fen
                    board.pop()

            move = self._select_child(node, board)
            board.push(move)
            fen_history.append((fen, move))

        reward = self._get_terminal_reward(board)
        
        for fen in visited:
            node = self.tree[fen]
            node["N"] += 1
            node["W"] += reward
        return

    def _select_child(self, node, board):
        best_score = -float('inf')
        best_move = None
        
        N_parent = max(node["N"], 1e-8)
        for move, child_fen in node["children"].items():
            child_node = self.tree.setdefault(child_fen, {"N": 0, "W": 0.0, "children": {}})
            
            if child_node["N"] == 0:
                exploration = float('inf')
                q_value = 0.0
            else:
                q_value = child_node["W"] / child_node["N"]
                exploration_term = np.log(N_parent) / (child_node["N"] + 1e-8)
                exploration_term = max(exploration_term, 0)
                exploration = self.c_puct * np.sqrt(exploration_term)
                
            ucb = q_value + exploration
            if ucb > best_score:
                best_score = ucb
                best_move = move
        
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = random.choice(legal_moves)
            else:
                raise ValueError("No legal moves available in _select_child")
                
        return best_move

    def _get_terminal_reward(self, board):
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _encode_action(self, move):
        return move.from_square * 64 + move.to_square

def run_episodes(env, agent, num_episodes=1000, method="q_learning"):
    rewards_history = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        if method == "sarsa":
            action = agent.choose_action(obs, env)

        while not done:
            if method == "q_learning":
                action = agent.choose_action(obs, env)
                next_obs, reward, done, _ = env.step(action)
                agent.update(obs, action, reward, next_obs, done, env)
                obs = next_obs
                total_reward += reward
            elif method == "sarsa":
                next_obs, reward, done, _ = env.step(action)
                if not done:
                    next_action = agent.choose_action(next_obs, env)
                else:
                    next_action = None
                agent.update(obs, action, reward, next_obs, next_action, done)
                obs = next_obs
                action = next_action
                total_reward += reward
            else:
                raise ValueError("Unknown method for run_episodes")

        rewards_history.append(total_reward)
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes}, Reward={total_reward:.2f}")
    return rewards_history

def run_episodes_mcts(env, agent, num_episodes=100):
    rewards_history = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action_idx = agent.choose_action(env)
            next_obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
        rewards_history.append(total_reward)
        if (ep+1) % 10 == 0:
            print(f"MCTS Episode {ep+1}/{num_episodes}, Reward={total_reward:.2f}")
    return rewards_history

def main(method="q_learning", use_dataset=False, csv_file=None):

    if method not in ["q_learning", "sarsa", "mcts"]:
        print("Usage: python final_chess.py [q_learning|sarsa|mcts]")
        return

    if len(sys.argv) >= 3 and sys.argv[2].lower() == "dataset":
        use_dataset = True
        if len(sys.argv) < 4:
            print("Please provide a CSV file after 'dataset'!")
            sys.exit(1)
        csv_file = sys.argv[3]
        print(f"Loading dataset from {csv_file}...")
        fen_dataset = load_fen_positions(csv_file, balanced_only=True)
        print(f"Loaded {len(fen_dataset)} FENs from dataset.")

    if use_dataset:
        env = FENDatasetChessEnv(fen_dataset, max_steps=50)
    else:
        env = ChessEnv(max_steps=50)

    if method == "q_learning":
        agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
        run_episodes(env, agent, num_episodes=500, method="q_learning")
    
    elif method == "sarsa":
        agent = SARSAAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)
        run_episodes(env, agent, num_episodes=500, method="sarsa")
    
    elif method == "mcts":
        agent = MCTSAgent(n_simulations=100, c_puct=1.4)
        run_episodes_mcts(env, agent, num_episodes=50)
    
    else:
        print("Invalid argument. Choose from [q_learning|sarsa|mcts].")
        sys.exit(1)



