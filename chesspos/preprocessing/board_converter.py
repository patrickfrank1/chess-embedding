import chess
import numpy as np

def board_to_bitboard(board):
	embedding = np.array([], dtype=bool)
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			bmp = np.zeros(shape=(64,)).astype(bool)
			for j in list(board.pieces(i, color)):
				bmp[j] = True
			embedding = np.concatenate((embedding, bmp))
	additional = np.array([
		bool(board.turn),
		board.has_kingside_castling_rights(chess.WHITE),
		board.has_queenside_castling_rights(chess.WHITE),
		board.has_kingside_castling_rights(chess.BLACK),
		board.has_queenside_castling_rights(chess.BLACK)
	])
	embedding = np.concatenate((embedding, additional))
	return embedding

def bitboard_to_board(bb):
	# set up empty board
	reconstructed_board = chess.Board()
	reconstructed_board.clear()
	# loop over all pieces and squares
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6 + i - 1
			piece = chess.Piece(i,color)

			bitmask = bb[idx*64:(idx+1)*64]
			squares = np.argwhere(bitmask)
			squares = [square for sublist in squares for square in sublist] # flatten list of lists

			for square in squares:
				reconstructed_board.set_piece_at(square,piece)
	# set global board information
	reconstructed_board.turn = bb[768]

	castling_rights = ''
	if bb[770]: # castling_h1
		castling_rights += 'K'
	if bb[769]: # castling_a1
		castling_rights += 'Q'
	if bb[772]: # castling_h8
		castling_rights += 'k'
	if bb[771]: # castling_a8
		castling_rights += 'q'
	reconstructed_board.set_castling_fen(castling_rights)

	return reconstructed_board

def board_to_tensor(board):
	embedding = np.empty((8,8,0), dtype=bool)
	# one plane per piece
	for color in [1, 0]:
		for i in range(1, 7): # P N B R Q K / white
			bmp = np.zeros(shape=(64,)).astype(bool)
			for j in list(board.pieces(i, color)):
				bmp[j] = True
			bmp = bmp.reshape((8,8,1))
			embedding = np.concatenate((embedding, bmp), axis=2)

	# castling rights at plane embedding(:,:,12)
	castling_rights = np.zeros((8,8,1), dtype=bool)
	castling_rights[0,0,0] = board.has_queenside_castling_rights(chess.WHITE)
	castling_rights[0,7,0] = board.has_kingside_castling_rights(chess.WHITE)
	castling_rights[7,0,0] = board.has_queenside_castling_rights(chess.BLACK)
	castling_rights[7,7,0] = board.has_kingside_castling_rights(chess.BLACK)
	embedding = np.concatenate((embedding, castling_rights), axis=2)

	# en passant squares at plane embedding(:,:,13)
	en_passant = np.zeros((64,), dtype=bool)
	if board.has_legal_en_passant():
		en_passant[board.ep_square] = True
	en_passant = en_passant.reshape((8,8,1))
	embedding = np.concatenate((embedding, en_passant), axis=2)

	# turn at plane embedding(:,:,14)
	turn = np.zeros((8,8,1), dtype=bool)
	turn[0,0,0] = board.turn
	embedding = np.concatenate((embedding, turn), axis=2)

	return embedding

def tensor_to_board(tensor):
	assert tensor.shape == (8,8,15)

	# set up empty board
	reconstructed_board = chess.Board()
	reconstructed_board.clear()

	# loop over all pieces and squares
	for color in [1, 0]: # white, black
		for i in range(1, 7): # P N B R Q K
			idx = (1-color)*6 + i - 1
			piece = chess.Piece(i,color)
			square_bitmask = tensor[:,:,idx].reshape((64,))
			squares = np.flatnonzero(square_bitmask)
			for square in squares:
				reconstructed_board.set_piece_at(square,piece)
	
	# set castling rights
	castling_rights = ''
	if tensor[0,0,12]: castling_rights += 'Q'
	if tensor[0,7,12]: castling_rights += 'K'
	if tensor[7,0,12]: castling_rights += 'q'
	if tensor[7,7,12]: castling_rights += 'k'
	reconstructed_board.set_castling_fen(castling_rights)

	# set en passant square
	en_passant = tensor[:,:,13].reshape((64,))
	if np.any(en_passant):
		reconstructed_board.ep_square = np.flatnonzero(en_passant)[0]

	# set turn
	reconstructed_board.turn = tensor[0,0,14]

	return reconstructed_board			
