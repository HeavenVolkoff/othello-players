# Internal
import math
import typing as T
from collections import Counter

# External
from othello.enums import Color
from othello.models import Board, Position

# Type generics
Rational = T.Union[float, int]

# Constants
CORNERS = (Position(1, 1), Position(1, 8), Position(8, 1), Position(8, 8))
BOARD_STABILITY_WEIGHTS = {
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
SCORE_WEIGHT = 25
STABILITY_WEIGHT = 15
MOBILITY_WEIGHT = 5
FRONT_MOBILITY_WEIGHT = 50
ACTUAL_MOBILITY_WEIGHT = 50
CORNERS_WEIGHT = 30
OCCUPIED_CORNERS_WEIGHT = 50
UNLIKELY_CORNERS_WEIGHT = 40
POTENTIAL_CORNERS_WEIGHT = 30
IRREVERSIBLE_CORNERS_WEIGHT = 15


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
        "maximizing",
        "_iter",
        "_children",
    )

    def __init__(
        self,
        move: T.Tuple[int, int],
        color: Color,
        board: "Board",
        *,
        beta: Rational = math.inf,
        depth: int = 0,
        alpha: Rational = -math.inf,
        parent: T.Optional["Node"] = None,
        maximizing: bool = True,
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
        self.evaluation: Rational = -math.inf if maximizing else math.inf
        self.maximizing = maximizing

        # Internal
        self._iter: T.Optional[T.Iterator["Node"]] = None
        self._children: T.Optional[T.Sequence["Node"]] = None

    def __iter__(self) -> T.Iterator["Node"]:
        """Magic Iterator method

        Returns:
             Iterable for all children of this node

        """
        if self._iter is None:
            self._iter = iter(
                Node(
                    move,
                    self.color.opposite(),
                    self.board.get_clone().play(move, self.color),
                    beta=self.beta,
                    depth=self.depth + 1,
                    alpha=self.alpha,
                    parent=self,
                    maximizing=not self.maximizing,
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

    def heuristic(self, color: Color) -> Rational:
        """Heuristic for how good of a move this node provides in a specific color perspective

        Adapted from:
            https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA

        Returns:
             Integer representing this node heuristic

        """
        opp_color = color.opposite()

        if color == Color.WHITE:
            score, opp_score = self.board.score()
        else:
            opp_score, score = self.board.score()
        score_h = (score - opp_score) / (score + opp_score) * 100

        occupied_corners = Counter(self.board[corner] for corner in CORNERS)
        occupied_corners_h = (
            (occupied_corners[color] - occupied_corners[opp_color])
            / (occupied_corners[color] + occupied_corners[opp_color])
            * 100
            if occupied_corners[color] + occupied_corners[opp_color]
            else 0
        )

        unlikely_corners: T.Counter[Color] = Counter()
        potential_corners: T.Counter[Color] = Counter()
        irreversible_corners: T.Counter[Color] = Counter()

        for corner in CORNERS:
            if self.board[corner] is not Color.EMPTY:
                continue

            for direction in self.board.DIRECTIONS:
                counter = irreversible_corners
                neighbour_pos = corner + direction
                neighbour = self.board[neighbour_pos]
                if neighbour in (Color.EMPTY, Color.OUTER):
                    continue

                for i in range(6):
                    neighbour_pos = neighbour_pos + direction
                    distant_neighbour = self.board[neighbour_pos]
                    if distant_neighbour != neighbour:
                        if distant_neighbour is Color.EMPTY:
                            counter = unlikely_corners
                        elif distant_neighbour is Color.OUTER:
                            raise RuntimeError(
                                "Potential corner calculation is incorrect"
                            )
                        else:
                            counter = potential_corners

                        break

                counter[neighbour] += 1

        unlikely_corners_h = (
            (unlikely_corners[color] - unlikely_corners[opp_color])
            / (unlikely_corners[color] + unlikely_corners[opp_color])
            * 100
            if unlikely_corners[color] + unlikely_corners[opp_color]
            else 0
        )

        potential_corners_h = (
            (potential_corners[color] - potential_corners[opp_color])
            / (potential_corners[color] + potential_corners[opp_color])
            * 100
            if potential_corners[color] + potential_corners[opp_color]
            else 0
        )

        irreversible_corners_h = (
            (irreversible_corners[color] - irreversible_corners[opp_color])
            / (irreversible_corners[color] + irreversible_corners[opp_color])
            * 100
            if irreversible_corners[color] + irreversible_corners[opp_color]
            else 0
        )

        moves = len(self.board.valid_moves(color))
        opp_moves = len(self.board.valid_moves(opp_color))
        mobility_h = (
            (moves - opp_moves) / (moves + opp_moves) * 100 if moves + opp_moves else 0
        )

        pieces_position = tuple(
            pos for pos in self.board.POSITIONS if self.board[pos] != Color.EMPTY
        )
        front_moves = Counter(
            self.board[pos]
            for direction in self.board.DIRECTIONS
            for pos in pieces_position
            if self.board[pos + direction] == Color.EMPTY
        )
        front_mobility_h = (
            (front_moves[color] - front_moves[opp_color])
            / (front_moves[color] + front_moves[opp_color])
            * 100
            if front_moves[color] + front_moves[opp_color]
            else 0
        )

        # TODO: Stability
        stability_h = 0

        return (
            (SCORE_WEIGHT * score_h)
            + (
                CORNERS_WEIGHT
                * (
                    OCCUPIED_CORNERS_WEIGHT * occupied_corners_h
                    + UNLIKELY_CORNERS_WEIGHT * unlikely_corners_h
                    + POTENTIAL_CORNERS_WEIGHT * potential_corners_h
                    + IRREVERSIBLE_CORNERS_WEIGHT * irreversible_corners_h
                )
                / (
                    4
                    * (
                        OCCUPIED_CORNERS_WEIGHT
                        + UNLIKELY_CORNERS_WEIGHT
                        + POTENTIAL_CORNERS_WEIGHT
                        + IRREVERSIBLE_CORNERS_WEIGHT
                    )
                )
            )
            + (
                MOBILITY_WEIGHT
                * (
                    ACTUAL_MOBILITY_WEIGHT * mobility_h
                    + FRONT_MOBILITY_WEIGHT * front_mobility_h
                )
                / (2 * (ACTUAL_MOBILITY_WEIGHT + FRONT_MOBILITY_WEIGHT))
            )
            + (STABILITY_WEIGHT * stability_h)
        ) / (SCORE_WEIGHT + CORNERS_WEIGHT + MOBILITY_WEIGHT + STABILITY_WEIGHT)


class MiniMaxPlayer:
    @staticmethod
    def minimax(
        board: "Board", color: Color, *, depth_limit: Rational = math.inf
    ) -> Node:
        current = Node((0, 0), color, board)
        while True:
            try:
                current = next(current)
                if current.depth >= depth_limit:
                    raise StopIteration
            except StopIteration:
                if abs(current.evaluation) == math.inf:
                    # We reached the bottom of the tree
                    # Assign node evaluation to it's heuristics
                    current.evaluation = current.heuristic(color)

                parent = current.parent
                if parent is None:
                    return current

                if parent.maximizing:
                    if current.evaluation > parent.evaluation:
                        parent.next_move = current.move
                        parent.evaluation = current.evaluation

                    if parent.evaluation >= parent.beta:
                        # Empty parent's children iterator
                        for _ in parent:
                            pass
                    elif parent.evaluation > parent.alpha:
                        parent.alpha = parent.evaluation
                else:
                    if current.evaluation <= parent.evaluation:
                        parent.next_move = current.move
                        parent.evaluation = current.evaluation

                    parent.beta = min(parent.beta, current.evaluation)
                    if parent.evaluation <= parent.alpha:
                        # Empty parent's children iterator
                        for _ in parent:
                            pass
                    elif parent.evaluation < parent.beta:
                        parent.beta = parent.evaluation

                current = parent

    def __init__(self, color: Color) -> None:
        self.depth = 5
        self.color = color

    def play(self, board: "Board") -> T.Tuple[int, int]:
        return self.minimax(board, self.color, depth_limit=self.depth).next_move
