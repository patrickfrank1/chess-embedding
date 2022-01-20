from abc import abstractmethod
from configparser import Interpolation
from typing import Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

import numpy as np
import chess

from chesspos.preprocessing.sample_generator import SampleGenerator
from chesspos.models.trainable_model import TrainableModel

class Autoencoder(TrainableModel):
	def __init__(
		self,
		*args,
		output_to_board: Callable[[np.ndarray], chess.Board] = None,
		**kwargs
	) -> None:
		super().__init__(*args, **kwargs)

		self.encoder = self._define_encoder()
		self.decoder = self._define_decoder()
		self.output_to_board = kwargs.pop('output_to_board')

		self._compile()

	def _compile(self) -> None:
		super()._compile()
		self.encoder.compile(optimizer=self.optimizer, loss=None, metrics=None)
		self.decoder.compile(optimizer=self.optimizer, loss=None, metrics=None)

	@abstractmethod
	def _define_encoder(self) -> Model:
		pass

	@abstractmethod
	def _define_decoder(self) -> Model:
		pass

	def _define_model(self) -> Model:
		encoder = self._define_encoder()
		decoder = self._define_decoder()
		autoencoder = decoder(encoder(encoder.input_layer))
		return keras.Model(inputs=encoder.input_layer, outputs=autoencoder, name="autoencoder")

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

	def _get_sorted_samples(self, sort_fn, number_samples, steps=None):
		pass

	def get_best_samples(self, number_samples, steps=None):
		pass

	def get_worst_samples(self, number_samples, steps=None):
		pass

	def compare_sample_to_prediction(self, bitboard):
		pass

	def interpolate(self, sample1, sample2):
		pass

	def interpolate2d(self, sample1, sample2, sample3, sample4):
		pass