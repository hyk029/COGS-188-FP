import os
import csv
import random
import chess
import chess.pgn
import argparse
import time

def parse_pgn_file(pgn_file, max_games=1000):
    positions = []
    games_processed = 0
    
    try:
        with open(pgn_file, encoding='utf-8') as f:
            while games_processed < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                
                moves = list(game.mainline_moves())
                min_move = min(10, len(moves) // 3)
                
                for i, move in enumerate(moves):
                    board.push(move)
                    
                    if i >= min_move and i % 2 == 0:
                        if not board.is_check() and not board.is_game_over():
                            positions.append(board.fen())
                
                games_processed += 1
                if games_processed % 10 == 0:
                    print(f"Processed {games_processed} games, extracted {len(positions)} positions")
    
    except Exception as e:
        print(f"Error processing PGN file: {e}")
    
    return positions

def generate_random_positions(num_positions=1000, depth_range=(5, 30)):
    positions = []
    
    for i in range(num_positions):
        board = chess.Board()
        
        depth = random.randint(depth_range[0], depth_range[1])
        
        for _ in range(depth):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            
            move = random.choice(legal_moves)
            board.push(move)
        
        if not board.is_game_over():
            positions.append(board.fen())
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_positions} positions")
    
    return positions

def is_material_balanced(board, threshold=2):
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

def save_positions_to_csv(positions, output_file, include_balanced_only=False):
    filtered_positions = []
    
    if include_balanced_only:
        for fen in positions:
            try:
                board = chess.Board(fen)
                if is_material_balanced(board):
                    filtered_positions.append(fen)
            except:
                continue
    else:
        filtered_positions = positions
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['fen', 'balanced', 'halfmoves'])
        writer.writeheader()
        
        for fen in filtered_positions:
            try:
                board = chess.Board(fen)
                writer.writerow({
                    'fen': fen,
                    'balanced': is_material_balanced(board),
                    'halfmoves': board.halfmove_clock
                })
            except Exception as e:
                print(f"Error processing FEN {fen}: {e}")
    
    print(f"Saved {len(filtered_positions)} positions to {output_file}")
    return len(filtered_positions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dataset of chess positions')
    parser.add_argument('--output', type=str, default='chess_positions.csv', help='Output CSV file')
    parser.add_argument('--positions', type=int, default=1000, help='Number of positions to generate')
    parser.add_argument('--balanced', action='store_true', help='Only include balanced positions')
    parser.add_argument('--pgn', type=str, default=None, help='Path to PGN file to extract positions from')
    parser.add_argument('--min-depth', type=int, default=5, help='Minimum number of moves for random positions')
    parser.add_argument('--max-depth', type=int, default=30, help='Maximum number of moves for random positions')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.pgn:
        print(f"Extracting positions from PGN file: {args.pgn}")
        positions = parse_pgn_file(args.pgn, max_games=args.positions // 5)
    else:
        print(f"Generating {args.positions} random positions...")
        positions = generate_random_positions(
            num_positions=args.positions, 
            depth_range=(args.min_depth, args.max_depth)
        )
    
    print(f"Saving positions to {args.output}...")
    count = save_positions_to_csv(positions, args.output, include_balanced_only=args.balanced)
    
    elapsed_time = time.time() - start_time
    print(f"Done! Generated {count} positions in {elapsed_time:.2f} seconds.")