import logging
from typing import Tuple

import numpy as np
import h5py
import chess
import chess.pgn
from pydantic import BaseModel

import chesspos.custom_types as ct
from chesspos.utils.utils import correct_file_ending

logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.ERROR,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	filename="pgn_extract.log"
)

class PgnExtractor(BaseModel):
	pgn_path: str
	save_path: str
	is_process_game: ct.GameFilter
	game_processor: ct.GameProcessor
	chunk_size: int = 100000
	log_level: int = logging.ERROR
	_game_counter: int = 0
	_chunk_counter: int = 0
	_encoding_counter: int = 0
	_encoding_shape: Tuple[int, ...]
	_encoding_type: np.dtype
	_discarded_games: int = 0
	_processed_games: int = 0

	# pydantic shortcoming: private underscore variables can not be validated using the @validator decorator
	# https://github.com/pydantic/pydantic/issues/655
	# so the default value needs to be set overwriting the __init__ method
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		object.__setattr__(self, '_encoding_shape', self._get_encoding_shape())
		object.__setattr__(self, '_encoding_type', self._get_encoding_type())

	def _get_encoding_shape(self):
		_, shape = self._get_encoding_type_and_shape()
		return shape

	def _get_encoding_type(self):
		dtype, _ = self._get_encoding_type_and_shape()
		return dtype 

	def _get_encoding_type_and_shape(self):
		with open(correct_file_ending(self.pgn_path, "pgn"), 'r') as f:
			game = chess.pgn.read_game(f)
			encoding = self.game_processor(game)
			logger.info(f"Type of encoding: {encoding.dtype}, shape of encoding: {encoding.shape[1:]}")
			return encoding.dtype, encoding.shape[1:]

	def _write_chunk_to_file(self, chunk: np.ndarray, metadata: np.ndarray):
		logger.info(f"Saving chunk {self._chunk_counter}")
		fname = correct_file_ending(self.save_path, "h5")

		try:
			with h5py.File(fname, "a") as save_file:
				data1 = save_file.create_dataset(f"encoding_{self._chunk_counter}", data=chunk, compression="gzip", compression_opts=9)
				data2 = save_file.create_dataset(f"game_id_{self._chunk_counter}", data=metadata, compression="gzip", compression_opts=9)
				logger.info(f"Saved encodings with shape {chunk.shape}")
		except Exception as e:
			logger.error(f"Could not save chunk {self._chunk_counter}", exc_info=True)
			raise e

		self._chunk_counter += 1
		self._encoding_counter = 0

	def _games(self, number_games):
		with open(correct_file_ending(self.pgn_path, "pgn"), 'r') as pgn_file:
			while True:
				header = chess.pgn.read_headers(pgn_file)
				self._game_counter += 1

				# Get next suitable game or finish extraction
				if header is None:
					logger.info(f"Processed {self._game_counter} games in total")
					yield None
				elif self.is_process_game(header):
					self._discarded_games += 1
					continue
				elif self._game_counter >= number_games:
					logger.info(f"Processed {self._game_counter} games in total")
					yield None
				else:
					self._processed_games += 1
					yield chess.pgn.read_game(pgn_file)

	def extract(self, number_games: int = int(1e18)):
		encoding_chunk = np.empty((self.chunk_size, *self._encoding_shape), dtype=self._encoding_type)
		game_id = np.empty((self.chunk_size,), dtype=np.int32)

		for game in self._games(number_games):
			if game is None:
				chunk = encoding_chunk[:self._encoding_counter]
				metadata = game_id[:self._encoding_counter]
				self._write_chunk_to_file(chunk, metadata)
				break

			# Extract information from that game
			encodings = self.game_processor(game)
			# Edge case: chunksize reached
			number_encodings = min(encodings.shape[0], self.chunk_size - self._encoding_counter)
			#logger.info(f"Extracted {number_encodings} encodings from game {self._game_counter}")
			new_encoding_counter = self._encoding_counter + number_encodings
			encoding_chunk[self._encoding_counter:new_encoding_counter, ...] = encodings[:number_encodings, ...]
			game_id[self._encoding_counter:new_encoding_counter] = self._game_counter*np.ones(number_encodings, dtype=np.int32)
			self._encoding_counter = new_encoding_counter

			# Save chunk if it is full
			if self._encoding_counter == self.chunk_size:
				logger.info(f"Processed {self._processed_games} games in this chunk")
				logger.info(f"Filtered {self._discarded_games} games in this chunk")
				self._processed_games = 0
				self._discarded_games = 0
				self._write_chunk_to_file(encoding_chunk, game_id)
		
		# Log file headers
		fname = correct_file_ending(self.save_path, "h5")
		with h5py.File(fname, 'r') as hf:
			logger.info(f"Keys: {hf.keys()}")
			logger.info(f"Shape of encoding: {hf['encoding_0'].shape}")
			logger.info(f"Shape of game_id: {hf['game_id_0'].shape}")
