from typing import TypeVar, Callable, Tuple

import numpy as np
from chess import Board
from chess.pgn import Game, Headers

ArrayTypeAndShape = TypeVar("ArrayTypeAndShape", bound=Tuple(np.dtype, np._Shape))
GameFilter = TypeVar("GameFilter", bound=Callable[[Headers], bool])
GameProcessor = TypeVar("GameProcessor", bound=Callable[[Game], np.ndarray])
PositionFilter = TypeVar("PositionFilter", bound=Callable[[Board], bool])
PositionProcessor = TypeVar("PositionProcessor", bound=Callable[[Board, int], np.ndarray])
PositionAggregator = TypeVar("PositionAggregator", bound=Callable[[np.ndarray], np.ndarray])
