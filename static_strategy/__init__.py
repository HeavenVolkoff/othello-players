# Internal
import math
import typing as T

# External
from othello.enums import Color
from othello.models import Board

# Adapted from:
#   https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA
BOARD_STATIC_WEIGHTS = {
    (i + 1, j + 1): weight
    for i, line in enumerate(
        (
            (4, -3, 2, 2, 2, 2, -3, 4),
            (-3, -4, -1, -1, -1, -1, -4, -3),
            (2, -1, 1, 0, 0, 1, -1, 2),
            (2, -1, 0, 1, 1, 0, -1, 2),
            (2, -1, 0, 1, 1, 0, -1, 2),
            (2, -1, 1, 0, 0, 1, -1, 2),
            (-3, -4, -1, -1, -1, -1, -4, -3),
            (4, -3, 2, 2, 2, 2, -3, 4),
        )
    )
    for j, weight in enumerate(line)
}


class StaticStrategyPlayer:
    def __init__(self, color: Color) -> None:
        self.color = color

    def play(self, board: Board) -> T.Tuple[int, int]:
        color = self.color
        opp_color = color.opposite()
        best_move = (0, 0)
        max_evaluation = -math.inf

        for move in board.valid_moves(color):
            future_board = board.get_clone().play(move, color)
            player = sum(
                BOARD_STATIC_WEIGHTS[pos]
                for pos in future_board.POSITIONS
                if future_board[pos] == color
            )
            opp_player = sum(
                BOARD_STATIC_WEIGHTS[pos]
                for pos in future_board.POSITIONS
                if future_board[pos] == opp_color
            )
            evaluation = player - opp_player
            if evaluation > max_evaluation:
                best_move = move
                max_evaluation = evaluation

        return best_move


__all__ = ("StaticStrategyPlayer", "BOARD_STATIC_WEIGHTS")
