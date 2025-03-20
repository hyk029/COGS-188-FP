import chess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from __init__ import ChessEnv, QLearningAgent, SARSAAgent, MCTSAgent
from heuristic_agents import RandomAgent, MaterialAgent, PositionalAgent

def decode_action(action_idx):
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    return chess.Move(from_sq, to_sq)

def tournament(trained_agent, opponent_agent, num_games=100, max_steps=100, agent_plays_white=True):
    wins = 0
    losses = 0
    draws = 0
    
    board = chess.Board()
    
    for game in tqdm(range(num_games)):
        board.reset()
        step = 0
        game_over = False
        
        white_agent = trained_agent if agent_plays_white else opponent_agent
        black_agent = opponent_agent if agent_plays_white else trained_agent
        
        while not game_over and step < max_steps:
            if board.turn == chess.WHITE:
                if isinstance(white_agent, (QLearningAgent, SARSAAgent, MCTSAgent)):
                    obs = np.zeros(64, dtype=np.int8)
                    for i in range(64):
                        piece = board.piece_at(i)
                        if piece:
                            val = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
                            obs[i] = val
                    
                    env = ChessEnv()
                    env.board = board.copy()
                    
                    if isinstance(white_agent, MCTSAgent):
                        action_idx = white_agent.choose_action(env)
                    else:
                        action_idx = white_agent.choose_action(obs, env)
                    move = decode_action(action_idx)
                else:
                    move = white_agent.choose_action(board)
                
                if move in board.legal_moves:
                    board.push(move)
                else:
                    legal = list(board.legal_moves)
                    if legal:
                        board.push(legal[0])
                    else:
                        game_over = True
            
            else:
                if isinstance(black_agent, (QLearningAgent, SARSAAgent, MCTSAgent)):
                    obs = np.zeros(64, dtype=np.int8)
                    for i in range(64):
                        piece = board.piece_at(i)
                        if piece:
                            val = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
                            obs[i] = val
                    
                    env = ChessEnv()
                    env.board = board.copy()
                    
                    if isinstance(black_agent, MCTSAgent):
                        action_idx = black_agent.choose_action(env)
                    else:
                        action_idx = black_agent.choose_action(obs, env)
                    move = decode_action(action_idx)
                else:
                    move = black_agent.choose_action(board)
                
                if move in board.legal_moves:
                    board.push(move)
                else:
                    legal = list(board.legal_moves)
                    if legal:
                        board.push(legal[0])
                    else:
                        game_over = True
            
            if board.is_game_over():
                game_over = True
            
            step += 1
        
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                if agent_plays_white:
                    losses += 1
                else:
                    wins += 1
            else:
                if agent_plays_white:
                    wins += 1
                else:
                    losses += 1
        else:
            draws += 1
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games
    }

def evaluate_against_baselines(agent, agent_type, num_games=100):
    baseline_agents = {
        "Random": RandomAgent(),
        "Material": MaterialAgent(),
        "Positional": PositionalAgent()
    }
    
    results = {}
    
    for baseline_name, baseline_agent in baseline_agents.items():
        print(f"Evaluating {agent_type} against {baseline_name} agent...")
        
        white_results = tournament(agent, baseline_agent, num_games=num_games//2, agent_plays_white=True)
        
        black_results = tournament(agent, baseline_agent, num_games=num_games//2, agent_plays_white=False)
        
        combined_results = {
            "wins": white_results["wins"] + black_results["wins"],
            "losses": white_results["losses"] + black_results["losses"],
            "draws": white_results["draws"] + black_results["draws"],
            "win_rate": (white_results["wins"] + black_results["wins"]) / num_games,
            "draw_rate": (white_results["draws"] + black_results["draws"]) / num_games,
            "loss_rate": (white_results["losses"] + black_results["losses"]) / num_games
        }
        
        results[baseline_name] = combined_results
    
    return results

def plot_baseline_results(results, agent_type):
    baselines = list(results.keys())
    win_rates = [results[baseline]["win_rate"] * 100 for baseline in baselines]
    draw_rates = [results[baseline]["draw_rate"] * 100 for baseline in baselines]
    loss_rates = [results[baseline]["loss_rate"] * 100 for baseline in baselines]
    
    x = np.arange(len(baselines))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, win_rates, width, label='Wins')
    rects2 = ax.bar(x, draw_rates, width, label='Draws')
    rects3 = ax.bar(x + width, loss_rates, width, label='Losses')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{agent_type} Performance Against Baseline Agents')
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.legend()
    
    plt.tight_layout()
    return fig