from typing import Callable
from attrs import define, field

import numpy as np
import h5py
import chess
import chess.pgn

from chesspos.utils.utils import correct_file_ending

@define
class PgnExtractor():
	pgn_path: str
	save_path: str
	game_processor: Callable[[chess.pgn.Game], np.ndarray]
	game_filter: Callable[[chess.pgn.Headers], bool] = lambda header: False
	chunk_size: int = 100000
	_game_counter: int = field(init=False, default=0)
	_chunk_counter: int = field(init=False, default=0)
	_encoding_counter: int = field(init=False, default=0)
	_encoding_shape: np.ndarray = field(init=False)
	_encoding_type: np.dtype = field(init=False)
	@_encoding_shape.default
	def _get_encoding_shape(self):
		_, shape = self._get_encoding_type_and_shape()
		return shape

	@_encoding_type.default
	def _get_encoding_type(self):
		dtype, _ = self._get_encoding_type_and_shape()
		return dtype 

	def _get_encoding_type_and_shape(self):
		with open(correct_file_ending(self.pgn_path, "pgn"), 'r') as f:
			game = chess.pgn.read_game(f)
			encoding = self.game_processor(game)
			print(f"Type of encoding: {encoding.dtype}, shape of encoding: {encoding.shape[1:]}")
			return encoding.dtype, encoding.shape[1:]

	def _write_chunk_to_file(self, chunk: np.ndarray, metadata: np.ndarray):
		print(f"Encoding counter: {self._encoding_counter}")
		if self._encoding_counter == self.chunk_size:
			print(f"Saving chunk {self._chunk_counter}")
			fname = correct_file_ending(self.save_path, "h5")

			with h5py.File(fname, "a") as save_file:
				data1 = save_file.create_dataset(f"encoding_{self._chunk_counter}", data=chunk, compression="gzip", compression_opts=9)
				data2 = save_file.create_dataset(f"game_id_{self._chunk_counter}", data=metadata, compression="gzip", compression_opts=9)
				print(f"Saved encodings with shape {chunk.shape}")

			self._chunk_counter += 1
			self._encoding_counter = 0

	def _games(self, number_games):
		with open(correct_file_ending(self.pgn_path, "pgn"), 'r') as pgn_file:
			while True:
				header = chess.pgn.read_headers(pgn_file)
				self._game_counter += 1

				# Get next suitable game or finish extraction
				if header is None:
					print(f"Processed games {self._game_counter}")
					yield None
				elif self.game_filter(header):
					print(f"Discarded game {self._game_counter}")
					continue
				elif self._game_counter >= number_games:
					print(f"Processed games {self._game_counter}")
					yield None
				else:
					yield chess.pgn.read_game(pgn_file)

	def extract(self, number_games: int = int(1e18)):
		encoding_chunk = np.empty((self.chunk_size, *self._encoding_shape), dtype=self._encoding_type)
		game_id = np.empty((self.chunk_size,), dtype=np.int32)

		for game in self._games(number_games):
			if game is None:
				chunk = encoding_chunk[:self._encoding_counter]
				metadata = game_id[:self._encoding_counter]
				self._write_chunk_to_file(encoding_chunk, game_id)
				break

			# Extract information from that game
			encodings = self.game_processor(game)
			print(f"Encodings shape: {encodings.shape}")
			# Edge case: chunksize reached
			number_encodings = min(encodings.shape[0], self.chunk_size - self._encoding_counter)
			print(f"Extracted {number_encodings} encodings from game {self._game_counter}")
			new_encoding_counter = self._encoding_counter + number_encodings
			encoding_chunk[self._encoding_counter:new_encoding_counter, ...] = encodings[:number_encodings, ...]
			game_id[self._encoding_counter:new_encoding_counter] = self._game_counter*np.ones(number_encodings, dtype=np.int32)
			self._encoding_counter = new_encoding_counter

			# Save chunk if it is full
			self._write_chunk_to_file(encoding_chunk, game_id)