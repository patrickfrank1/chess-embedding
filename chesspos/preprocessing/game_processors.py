import logging
import chesspos.custom_types as ct

import chess
import chess.pgn
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.ERROR,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	filename="pgn_extract.log"
)

class GameProcessor(BaseModel):
	is_process_position: ct.PositionFilter
	position_processor: ct.PositionProcessor
	position_aggregator: ct.PositionAggregator = lambda x: x
	_board: chess.Board = chess.Board()
	_encodings: list[np.ndarray] = []
	
	def __call__(self, game: chess.pgn.Game) -> np.ndarray:
		return self.game_processor(game)

	def _push_move(self, move: chess.Move, move_nr: int = -1) -> None:
		"""Push a move to the board and append the position to the list of positions"""
		try:
			self._board.push(move)
		except Exception as e:
			logger.error(f"Exception occurred in position number {move_nr}")
			return self._encodings

	def game_processor(self, game: chess.pgn.Game) -> np.ndarray:
		"""Process a game and return a numpy array of the processed positions"""
		for i, move in enumerate(game.mainline()):
			self._push_move(move, i)
			if self.is_process_position(self._board):
				encoding = self.position_processor(self._board)
				self._encodings.append(encoding)
		aggregated_encodings = self.position_aggregator(np.array(self._encodings))
		return aggregated_encodings

