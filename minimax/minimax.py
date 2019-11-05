# Internal
import math
import typing as T
from datetime import datetime, timedelta

# External
from loky import get_reusable_executor
from othello.enums import Color
from othello.models import Board

# Project
from .node import Node
from .heuristics import Rational


class MinimaxPlayer:
    NODE_CLS = Node
    TIME_LIMIT = timedelta(seconds=2.9)

    def __init__(self, color: Color) -> None:
        self.color = color

    def play(self, board: Board) -> T.Tuple[int, int]:
        return self.minimax_multiprocess(board, self.color).next_move

    def minimax(self, head: T.Optional[Node] = None, board: T.Optional[Board] = None) -> Node:
        if head:
            current = head
        else:
            assert board
            current = self.NODE_CLS((0, 0), self.color, board)

        past = initial = datetime.now()
        depth_limit = 3
        while True:
            try:
                current = next(current)
                if current.depth >= depth_limit:
                    raise StopIteration
            except StopIteration:
                if abs(current.evaluation) == math.inf:
                    # We reached the bottom of the tree
                    # Assign node evaluation to it's heuristics
                    current.evaluation = current.heuristic(self.color)

                parent = current.parent
                if parent is None:
                    return current

                parent.minimax(current.evaluation, current.move)

                if parent.parent is None:
                    now = datetime.now()
                    local_diff = now - past
                    global_diff = now - initial

                    depth_limit = int(
                        math.log((self.TIME_LIMIT - global_diff) / local_diff) / math.log(7) + 0.1
                    )

                    if global_diff > self.TIME_LIMIT or depth_limit == 0:
                        return parent

                    past = now

                current = parent

    def intermediate(self, node_info: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        del node_info["next_move"]
        del node_info["evaluation"]

        return self.minimax(self.NODE_CLS(**node_info)).get_info()

    def minimax_multiprocess(
        self, board: Board, color: Color, *, depth_limit: Rational = math.inf
    ) -> Node:
        head = self.NODE_CLS((0, 0), color, board)
        children = tuple(head)

        with get_reusable_executor() as executor:
            for child, info in zip(
                children,
                executor.map(self.intermediate, (child.get_info() for child in children)),
            ):
                child.next_move = info["next_move"]
                child.evaluation = info["evaluation"]

        best_children = max(children)
        head.next_move = best_children.move
        head.evaluation = best_children.evaluation
        return head


__all__ = ("MinimaxPlayer",)
