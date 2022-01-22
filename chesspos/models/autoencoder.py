from abc import abstractmethod
from functools import cmp_to_key, wraps
from typing import Callable, List, overload
from colorama import Fore, Style
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

import numpy as np
import chess

from chesspos.models.trainable_model import TrainableModel

class AutoencoderModel(TrainableModel):
	@wraps(TrainableModel.__init__)
	def __init__(
		self,
		output_to_board: Callable[[np.ndarray], chess.Board] = None,
		**kwargs
	) -> None:
		super(AutoencoderModel, self).__init__(**kwargs)

		self.encoder = self._define_encoder()
		self.decoder = self._define_decoder()
		self.output_to_board = output_to_board

		self._compile()

	def _compile(self) -> None:
		super()._compile()
		self.encoder.compile(optimizer=self.optimizer, loss=None, metrics=None)
		keras.utils.plot_model(self.encoder, to_file=f"{self.save_dir}/encoder.png", show_shapes=True)

		self.decoder.compile(optimizer=self.optimizer, loss=None, metrics=None)
		keras.utils.plot_model(self.decoder, to_file=f"{self.save_dir}/decoder.png", show_shapes=True)

	@abstractmethod
	def _define_encoder(self) -> Model:
		pass

	@abstractmethod
	def _define_decoder(self) -> Model:
		pass

	def _define_model(self) -> Model:
		encoder = self._define_encoder()
		decoder = self._define_decoder()
		autoencoder = decoder(encoder)
		return keras.Model(inputs=encoder.get_layer(index=0).input, outputs=autoencoder, name="autoencoder")

	def get_encoder(self):
		return self.encoder


	def get_decoder(self):
		return self.decoder

	@staticmethod
	def binarize_array(array, threshold=0.5):
		return np.where(array > threshold, True, False)

	def _check_output_converter(self) -> None:
		if self.output_to_board is None:
			raise ValueError("No model output to board converter set.")

	def _predict_from_embedding(self, embedding: np.ndarray) -> np.ndarray:
		return self.decoder.predict(embedding)
	
	def predict_board_from_embedding(self, embedding: np.ndarray) -> chess.Board:
		return self.output_to_board(self._predict_from_embedding(embedding))

	def evaluate(self, samples: np.ndarray) -> np.float32:
		return super().evaluate(samples, samples)

	def evaluate_from_board(self, boards: List[chess.Board]) -> np.float32:
		self._check_input_converter()
		inputs = np.empty((len(boards), *self.train_generator.sample_shape))
		for i, board in enumerate(boards):
			inputs[i] = self.board_to_input(board)
		return self.evaluate(inputs)

	def _get_sorted_losses(self, sort_fn: Callable, number_samples: int = 1000) -> List[dict]:
		test_generator = self.test_generator.get_generator()
		batch_size = self.test_generator.batch_size
		samples = []

		for i in range(number_samples // batch_size):
			x, y = next(test_generator)
			print(x.shape)
			for j in range(batch_size):
				input = x[j].reshape(1, *x[j].shape)
				loss = self.model.evaluate(input)
				samples.append({'input':input, 'loss':loss})

			sort_wrapper = lambda a, b: sort_fn(a['loss'], b['loss'])
			samples.sort(key=cmp_to_key(sort_wrapper))

		return samples

	def get_best_samples(self, number_samples: int)-> List[dict]:
		max_loss = lambda x, y : x - y
		return self._get_sorted_losses(max_loss, number_samples=10*number_samples)[:number_samples]

	def get_worst_samples(self, number_samples: int)-> List[dict]:
		min_loss = lambda x, y : y - x
		return self._get_sorted_losses(min_loss, number_samples=10*number_samples)[:number_samples]

	def compare_input_to_prediction(self, input: np.ndarray) -> None:
		prediction = self.model.predict(input)[0,:,:,:,0]
		print(prediction.shape)
		prediction = self.binarize_array(prediction)
		prediction_board = self.output_to_board(prediction)
		prediction_board = prediction_board.__str__().split("\n")

		input_board = self.output_to_board(input[0,:,:,:,0])
		input_board = input_board.__str__().split("\n")

		for i in range(len(prediction_board)):
			prediction_board[i] = ''.join([f'{Fore.RED}{prediction_board[i][j]}{Style.RESET_ALL}' if input_board[i][j] != prediction_board[i][j] else prediction_board[i][j] for j in range(len(prediction_board[i]))])

		output_str = "Original:          Reconstructed:\n"
		output_str += ''.join([f"{input_board[i]}    {prediction_board[i]}\n" for i in range(len(input_board))]) +'\n'
		return output_str

	def interpolate(self, sample1, sample2):
		pass

	def interpolate2d(self, sample1, sample2, sample3, sample4):
		pass