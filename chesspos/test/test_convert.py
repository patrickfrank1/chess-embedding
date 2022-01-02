import numpy as np
import chess

from chesspos.preprocessing.board_converter import board_to_bitboard, bitboard_to_board

start_board = chess.Board(chess.STARTING_FEN)
start_bb = np.load("./chesspos/test/startpos.npy")

def test_board_to_bitboard():
	board_result = board_to_bitboard(start_board)
	assert board_result.shape == (773,)
	assert board_result.dtype == 'bool'
	assert np.all(board_result == start_bb)

def test_full_conversion():
	start_bitboard = board_to_bitboard(start_board)
	reconstructed_board = bitboard_to_board(start_bitboard)
	print(reconstructed_board.__str__())
	assert start_board.board_fen() == reconstructed_board.board_fen()

if __name__ == "__main__":
	test_full_conversion()