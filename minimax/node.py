# Internal
import math
import typing as T

# External
from othello.enums import Color
from othello.models import Board

# Project
from .heuristics import (
    Rational,
    score_heuristic,
    corners_heuristic,
    mobility_heuristic,
    stability_heuristic,
)

# Constants
SCORE_WEIGHT = 25
CORNERS_WEIGHT = 30
MOBILITY_WEIGHT = 5
STABILITY_WEIGHT = 15


class Node(T.Iterator["Node"]):
    __slots__ = (
        "move",
        "beta",
        "depth",
        "color",
        "board",
        "alpha",
        "parent",
        "next_move",
        "evaluation",
        "_iter",
        "_children",
        "_maximize",
    )

    def __init__(
        self,
        move: T.Tuple[int, int],
        color: Color,
        board: Board,
        *,
        beta: Rational = math.inf,
        depth: int = 0,
        alpha: Rational = -math.inf,
        parent: T.Optional["Node"] = None,
        maximize: bool = True,
    ):
        """Node representation for depth search of possible moves"""
        self.move = move
        self.beta = beta
        self.depth = depth
        self.color = color
        self.board = board
        self.alpha = alpha
        self.parent = parent
        self.next_move: T.Tuple[int, int] = (0, 0)
        self.evaluation: Rational = -math.inf if maximize else math.inf

        # Internal
        self._iter: T.Optional[T.Iterator["Node"]] = None
        self._maximize = maximize
        self._children: T.Optional[T.Sequence["Node"]] = None

    def __gt__(self, other: "Node") -> bool:
        return self.evaluation > other.evaluation

    def __lt__(self, other: "Node") -> bool:
        return self.evaluation < other.evaluation

    def __iter__(self) -> T.Iterator["Node"]:
        """Magic Iterator method

        Returns:
             Iterable for all children of this node

        """
        if self._iter is None:
            self._iter = iter(
                type(self)(
                    move,
                    self.color.opposite(),
                    self.board.get_clone().play(move, self.color),
                    beta=self.beta,
                    depth=self.depth + 1,
                    alpha=self.alpha,
                    parent=self,
                    maximize=not self._maximize,
                )
                for move in self.board.valid_moves(self.color)
            )

        return self._iter

    def __next__(self) -> "Node":
        """Magic Iterator method

        Returns:
             Next children in iterator

        """
        return next(iter(self))

    def minimax(self, evaluation: Rational, move: T.Tuple[int, int]) -> None:
        if self._maximize:
            if evaluation > self.evaluation:
                self.next_move = move
                self.evaluation = evaluation

            if self.evaluation >= self.beta:
                # Empty iterator
                _ = tuple(self)
            elif self.evaluation > self.alpha:
                self.alpha = self.evaluation
        else:
            if evaluation < self.evaluation:
                self.next_move = move
                self.evaluation = evaluation

            if self.evaluation <= self.alpha:
                # Empty iterator
                _ = tuple(self)
            elif self.evaluation < self.beta:
                self.beta = self.evaluation

    def heuristic(self, color: Color) -> Rational:
        """Heuristic for how good of a move this node provides in a specific color perspective

        Adapted from:
            https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA

        Returns:
             Rational representing this node heuristic

        """
        score_weight = SCORE_WEIGHT
        corners_weight = CORNERS_WEIGHT
        remaining_moves = self.board.MAX_TURNS - self.board.turns
        stability_weight = STABILITY_WEIGHT

        # Dynamic weights based on how many moves are left
        if remaining_moves <= 20:
            score_weight += 10
            corners_weight += 20
            stability_weight += 10
        if 20 < remaining_moves <= 40:
            corners_weight += 10
            stability_weight += 10

        return (
            (SCORE_WEIGHT * score_heuristic(self.board, color))
            + (CORNERS_WEIGHT * corners_heuristic(self.board, color))
            + (MOBILITY_WEIGHT * mobility_heuristic(self.board, color))
            + (STABILITY_WEIGHT * stability_heuristic(self.board, color))
        ) / (SCORE_WEIGHT + CORNERS_WEIGHT + MOBILITY_WEIGHT + STABILITY_WEIGHT)

    def get_info(self):
        return {
            "move": self.move,
            "color": self.color,
            "board": self.board,
            "beta": self.beta,
            "depth": self.depth,
            "alpha": self.alpha,
            "maximize": self._maximize,
            "next_move": self.next_move,
            "evaluation": self.evaluation,
        }


__all__ = ("Node",)
