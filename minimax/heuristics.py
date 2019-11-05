# Internal
import typing as T
from collections import Counter

# External
from othello.enums import Color
from othello.models import Board, Position

# Type generics
Rational = T.Union[float, int]

# Constants
FRONT_MOBILITY_WEIGHT = 50
ACTUAL_MOBILITY_WEIGHT = 50
OCCUPIED_CORNERS_WEIGHT = 50
UNLIKELY_CORNERS_WEIGHT = 40
POTENTIAL_CORNERS_WEIGHT = 30
IRREVERSIBLE_CORNERS_WEIGHT = 15


def score_heuristic(board: Board, color: Color) -> Rational:
    if color == Color.WHITE:
        score, opp_score = board.score()
    else:
        opp_score, score = board.score()

    return (score - opp_score) / (score + opp_score) * 100


def corners_heuristic(board: Board, color: Color) -> Rational:
    opp_color = color.opposite()
    occupied_corners = Counter(board[corner] for corner in board.CORNERS)
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

    for corner in board.CORNERS:
        if board[corner] is not Color.EMPTY:
            continue

        for direction in board.DIRECTIONS:
            counter = irreversible_corners
            neighbour_pos = corner + direction
            neighbour = board[neighbour_pos]
            if neighbour in (Color.EMPTY, Color.OUTER):
                continue

            for i in range(6):
                neighbour_pos = neighbour_pos + direction
                distant_neighbour = board[neighbour_pos]
                if distant_neighbour != neighbour:
                    if distant_neighbour is Color.EMPTY:
                        counter = unlikely_corners
                    elif distant_neighbour is Color.OUTER:
                        raise RuntimeError("Potential corner calculation is incorrect")
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

    return (
        OCCUPIED_CORNERS_WEIGHT * occupied_corners_h
        + UNLIKELY_CORNERS_WEIGHT * unlikely_corners_h
        + POTENTIAL_CORNERS_WEIGHT * potential_corners_h
        + IRREVERSIBLE_CORNERS_WEIGHT * irreversible_corners_h
    ) / (
        4
        * (
            OCCUPIED_CORNERS_WEIGHT
            + UNLIKELY_CORNERS_WEIGHT
            + POTENTIAL_CORNERS_WEIGHT
            + IRREVERSIBLE_CORNERS_WEIGHT
        )
    )


def mobility_heuristic(board: Board, color: Color) -> Rational:
    opp_color = color.opposite()
    moves = len(board.valid_moves(color))
    opp_moves = len(board.valid_moves(opp_color))
    mobility_h = (moves - opp_moves) / (moves + opp_moves) * 100 if moves + opp_moves else 0

    pieces_position = tuple(pos for pos in board.POSITIONS if board[pos] != Color.EMPTY)
    front_moves = Counter(
        board[pos]
        for direction in board.DIRECTIONS
        for pos in pieces_position
        if board[pos + direction] == Color.EMPTY
    )
    front_mobility_h = (
        (front_moves[color] - front_moves[opp_color])
        / (front_moves[color] + front_moves[opp_color])
        * 100
        if front_moves[color] + front_moves[opp_color]
        else 0
    )

    return (ACTUAL_MOBILITY_WEIGHT * mobility_h + FRONT_MOBILITY_WEIGHT * front_mobility_h) / (
        2 * (ACTUAL_MOBILITY_WEIGHT + FRONT_MOBILITY_WEIGHT)
    )


def stability_heuristic(board: Board, color: Color) -> Rational:
    opp_color = color.opposite()
    stability: T.Counter[Color] = Counter()
    directions = (Position(1, 0), Position(-1, 1), Position(0, 1), Position(1, 1))

    for pos in board.POSITIONS:
        occupied = board[pos]
        if occupied == Color.EMPTY:
            continue

        pos_stability = 2

        for direction in directions:
            neighbour_pos = pos + direction
            while neighbour_pos in board.POSITIONS:
                forward_neighbour = board[neighbour_pos]
                if forward_neighbour != occupied:
                    break
                neighbour_pos += direction
            else:
                continue

            neighbour_pos = pos - direction
            while neighbour_pos in board.POSITIONS:
                backward_neighbour = board[neighbour_pos]
                if backward_neighbour != occupied:
                    break
                neighbour_pos -= direction
            else:
                continue

            if forward_neighbour == backward_neighbour:
                pos_stability = 1
            else:
                pos_stability = 0
                break

        stability[occupied] += pos_stability

    return (
        (stability[color] - stability[opp_color]) / (stability[color] + stability[opp_color]) * 100
        if stability[color] + stability[opp_color]
        else 0
    )


__all__ = (
    "Rational",
    "score_heuristic",
    "corners_heuristic",
    "mobility_heuristic",
    "stability_heuristic",
)
