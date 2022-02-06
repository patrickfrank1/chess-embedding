import logging
from typing import Callable, Union

import chess
import chess.pgn
import numpy as np

from chesspos.preprocessing.board_converter import board_to_bitboard, board_to_tensor

def sparse_positions_to_tensor(game: chess.pgn.Game):
	return single_positions(game, board_to_tensor, heavy_subsampling)

def positions_to_tensor(game: chess.pgn.Game):
	return single_positions(game, board_to_tensor, subsample_opening_positions)

def positions_to_bitboard(game: chess.pgn.Game):
	return single_positions(game, board_to_bitboard, subsample_opening_positions)

def positions_to_tensor_triplets(game: chess.pgn.Game) -> np.ndarray:
	position_encodings = single_positions(game, board_to_tensor, subsample_opening_positions)
	number_positions = position_encodings.shape[0]
	anchor_index = np.random.randint(number_positions-1)
	positive_index = anchor_index + 1
	negative_index = (anchor_index + number_positions // 2) % number_positions
	triplets = position_encodings[[anchor_index, positive_index, negative_index],...]
	return triplets.reshape(1, 3, *position_encodings.shape[1:])

def single_positions(game: chess.pgn.Game, position_encoder: Callable[[chess.Board], np.ndarray],
	sample_move: Callable[[Union[int,chess.Move, None]], bool]) -> np.ndarray:
	logger = logging.getLogger(__name__)
	output_shape = position_encoder(chess.Board()).shape
	output_type = position_encoder(chess.Board()).dtype
	output = np.empty((0, *output_shape), dtype=output_type)

	board = chess.Board()
	for i, move in enumerate(game.mainline_moves()):
		try:
			board.push(move)
		except Exception as e:
			logger.error(f"Exception occurred in position number {i}", exec_info=True)
			return output

		if sample_move(i):
			output = np.append(output, position_encoder(board).reshape(1,*output_shape), axis=0)
	return output

def subsample_opening_positions(ply: int) -> bool:
	selection_probability = min(ply / 40, 1.0)
	return np.random.random() < selection_probability

def heavy_subsampling(ply: int):
	selection_probability = subsample_opening_positions(ply) / 50.0
	return np.random.random() < selection_probability