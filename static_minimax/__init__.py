# Internal
import typing as T

# External
from minimax import MinimaxPlayer
from minimax.node import Node
from othello.enums import Color
from static_strategy import BOARD_STATIC_WEIGHTS

# Type generics
Rational = T.Union[float, int]


class StaticNode(Node):
    def heuristic(self, color: Color) -> Rational:
        """Heuristic for how good of a move this node provides in a specific color perspective

        Adapted from:
            https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA

        Returns:
             Rational representing this node heuristic

        """
        opp_color = color.opposite()
        player = sum(
            BOARD_STATIC_WEIGHTS[pos] for pos in self.board.POSITIONS if self.board[pos] == color
        )
        opp_player = sum(
            BOARD_STATIC_WEIGHTS[pos]
            for pos in self.board.POSITIONS
            if self.board[pos] == opp_color
        )
        return player - opp_player


class StaticMiniMaxPlayer(MinimaxPlayer):
    Node = StaticNode


__all__ = ("StaticMiniMaxPlayer",)
