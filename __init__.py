import gym
import numpy as np
import random
import chess
import sys
import csv
import time
from gym import spaces

def load_fen_positions(csv_file, balanced_only=True, max_positions=5000):
    """
    Load FEN positions from a CSV file
    
    Args:
        csv_file: Path to CSV file containing FEN strings
        balanced_only: If True, only load positions where material is roughly balanced
        max_positions: Maximum number of positions to load
        
    Returns:
        List of FEN strings
    """
    fen_list = []
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)  
            for row in reader:
                if 'fen' not in row:
                    fen_key = next((key for key in row.keys() if 'fen' in key.lower()), None)
                    if not fen_key:
                        print(f"Warning: No FEN column found in the CSV. Available columns: {list(row.keys())}")
                        break
                else:
                    fen_key = 'fen'
                
                fen = row[fen_key]
                
                try:
                    board = chess.Board(fen)
                    if not balanced_only or is_material_balanced(board):
                        fen_list.append(fen)
                except Exception as e:
                    continue
                
                if len(fen_list) >= max_positions:
                    break
        
        print(f"Successfully loaded {len(fen_list)} FEN positions from {csv_file}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
    return fen_list

def is_material_balanced(board, threshold=2):
    """
    Check if the position has roughly balanced material
    
    Args:
        board: chess.Board object
        threshold: Maximum material imbalance allowed
        
    Returns:
        True if material is balanced within threshold
    """
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                    chess.ROOK: 5, chess.QUEEN: 9}
    
    white_material = 0
    black_material = 0
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    return abs(white_material - black_material) <= threshold

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
            material_advantage = self._calculate_material_advantage()
            return material_advantage * 0.01
        if self.board.is_checkmate():
            return -10.0
        return 1.0 

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
    
    def _calculate_material_advantage(self):
        """
        Calculate the material advantage for the current player
        """
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material

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
    """
    Optimized Monte Carlo Tree Search implementation with memory management
    """
    def __init__(self, n_simulations=50, c_puct=1.4, max_depth=15, timeout=2.0, max_tree_size=5000):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.timeout = timeout
        self.max_tree_size = max_tree_size
        self.tree = {}
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
    def choose_action(self, env):
        """
        Select the best action according to MCTS simulations
        """
        start_time = time.time()
        board = env.board.copy()
        state_key = self._board_to_key(board)
        
        if len(self.tree) > self.max_tree_size:
            self._prune_tree()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0
        
        simulation_count = 0
        while simulation_count < self.n_simulations:
            if time.time() - start_time > self.timeout:
                break
            
            self._simulate(board.copy(), state_key, 0)
            simulation_count += 1
            
            if simulation_count % 5 == 0 and time.time() - start_time > self.timeout * 0.8:
                break
        
        if state_key not in self.tree:
            return self._encode_action(random.choice(legal_moves))
        
        node = self.tree[state_key]
        if not node[2]:
            return self._encode_action(random.choice(legal_moves))
        
        best_move, best_visits = None, -1
        for move_str, child_key in node[2].items():
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                continue
                
            if child_key in self.tree:
                child_visits = self.tree[child_key][0]
                if child_visits > best_visits:
                    best_visits = child_visits
                    best_move = move
        
        if best_move is None:
            best_move = random.choice(legal_moves)
            
        return self._encode_action(best_move)
    
    def _simulate(self, board, state_key=None, depth=0):
        """
        Run a single MCTS simulation
        """
        if state_key is None:
            state_key = self._board_to_key(board)
            
        if depth >= self.max_depth:
            return self._evaluate_position(board)
            
        if board.is_game_over():
            return self._get_terminal_reward(board)
            
        if state_key not in self.tree:
            self.tree[state_key] = (0, 0.0, {})
            
            if depth > 0:
                return self._evaluate_position(board)
        
        node = self.tree[state_key]
        
        if not node[2]:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return self._evaluate_position(board)
                
            for move in legal_moves:
                node[2][move.uci()] = None
                
            selected_move = random.choice(legal_moves)
            board.push(selected_move)
            child_key = self._board_to_key(board)
            node[2][selected_move.uci()] = child_key
            
            value = self._simulate(board, child_key, depth + 1)
            
        else:
            selected_move = self._select_child(board, node)
            
            if selected_move is None:
                return self._evaluate_position(board)
                
            move_str = selected_move.uci()
            child_key = node[2].get(move_str)
            
            if child_key is None:
                board.push(selected_move)
                child_key = self._board_to_key(board)
                node[2][move_str] = child_key
            else:
                board.push(selected_move)
                

            value = self._simulate(board, child_key, depth + 1)
        
        visits, total_value, children = node
        self.tree[state_key] = (visits + 1, total_value + value, children)
        
        return value
    
    def _select_child(self, board, node):
        """
        Select child with highest UCT value
        """
        visits, _, children = node
        best_score = float('-inf')
        best_move = None
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        for move in legal_moves:
            move_str = move.uci()
            if move_str not in children:
                continue
                
            child_key = children[move_str]
            
            if child_key is None or child_key not in self.tree:
                return move
                
            child_visits, child_value, _ = self.tree[child_key]
            
            if child_visits == 0:
                return move
                
            exploration = 0.0
            if visits > 0 and child_visits > 0:
                try:
                    exploration = self.c_puct * np.sqrt(np.log(visits) / child_visits)
                except (ValueError, RuntimeWarning):
                    exploration = self.c_puct
            
            exploitation = child_value / max(child_visits, 1)
            uct_value = exploitation + exploration
            
            if uct_value > best_score:
                best_score = uct_value
                best_move = move
        
        if best_move is None and legal_moves:
            best_move = random.choice(legal_moves)
            
        return best_move
    
    def _get_terminal_reward(self, board):
        """
        Evaluate terminal game state
        """
        if board.is_checkmate():
            return -1.0
        return 0.0
    
    def _evaluate_position(self, board):
        """
        Simple material-based evaluation
        """
        if board.is_game_over():
            return self._get_terminal_reward(board)
            
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        normalized = np.tanh(score / 15.0)
        return -normalized
    
    def _board_to_key(self, board):
        """
        Convert board to a hashable key
        """
        return board.fen().split(' ')[0]
    
    def _encode_action(self, move):
        """
        Convert chess move to action index
        """
        return move.from_square * 64 + move.to_square
    
    def _prune_tree(self):
        """
        Prune search tree to reduce memory usage
        """
        if not self.tree:
            return
            
        sorted_nodes = sorted(self.tree.items(), key=lambda x: x[1][0], reverse=True)
        
        keep_count = min(50, len(sorted_nodes))
        nodes_to_keep = {k for k, _ in sorted_nodes[:keep_count]}
        
        new_tree = {k: v for k, v in self.tree.items() if k in nodes_to_keep}
        self.tree = new_tree

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



