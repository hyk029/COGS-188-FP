import random
import chess
import numpy as np

class RandomAgent:
    """
    Simple agent that makes random legal moves.
    """
    def choose_action(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return legal_moves[random.randint(0, len(legal_moves)-1)]

class MaterialAgent:
    """
    Heuristic agent that prioritizes capturing highest value pieces.
    """
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
    def evaluate_move(self, board, move):
        if board.is_capture(move):
            to_square = move.to_square
            captured_piece = board.piece_at(to_square)
            if captured_piece:
                return self.piece_values.get(captured_piece.piece_type, 0)
        return 0

    def choose_action(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        move_values = [(move, self.evaluate_move(board, move)) for move in legal_moves]
        
        move_values.sort(key=lambda x: x[1], reverse=True)
        
        best_value = move_values[0][1]
        best_moves = [move for move, value in move_values if value == best_value]
        
        return random.choice(best_moves)

class PositionalAgent:
    """
    Agent that balances material with piece positioning.
    """
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
    def evaluate_board(self, board):
        if board.is_checkmate():
            return -10000 if board.turn else 10000
            
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        value = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if piece.piece_type == chess.PAWN:
                    position_value = self.pawn_table[square if piece.color else chess.square_mirror(square)]
                    piece_value += position_value
                    
                if piece.color:
                    value += piece_value
                else:
                    value -= piece_value
                    
        return value if board.turn else -value

    def choose_action(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        best_move = None
        best_value = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            value = -self.evaluate_board(board)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                
        return best_move