import os
import h5py
import numpy as np
import tensorflow as tf

from chesspos.utils.utils import correct_file_ending, files_from_directory
from chesspos.preprocessing.board_preprocessor import easy_triplets, semihard_triplets, hard_triplets, singlets



class SampleGenerator():
	def __init__(
		self,
		sample_dir,
		format,
		batch_size=16
	):
		self.H5_COL_KEY = 'tuples'
		self.sample_dir = sample_dir
		self.format = format
		self.batch_size = batch_size

		self.subsampling_functions = None
		self.generator = None


	def set_subsampling_functions(self, sampling_functions):
		self.subsampling_functions = []
		for element in sampling_functions:
			if element == 'easy_triplets':
				self.subsampling_functions.append(easy_triplets)
			elif element == 'semihard_triplets':
				self.subsampling_functions.append(semihard_triplets)
			elif element == 'hard_triplets':
				self.subsampling_functions.append(hard_triplets)
			elif element == 'singlets':
				self.subsampling_functions.append(singlets)
			else:
				if callable(element):
					self.subsampling_functions.append(element)
				else:
					raise ValueError(f'{element} is not callable.')


	def construct_generator(self):
		def generator():
			sample_files = files_from_directory(os.path.abspath(self.sample_dir), file_type="h5")
			tuples = None

			if self.format == "bitboard":
				tuples = np.empty(shape=(0, 15, 773), dtype=bool)
			elif self.format == "tensor":
				tuples = np.empty(shape=(0, 15, 8, 8, 15, 1), dtype=bool)
			else:
				raise ValueError(f'{self.format} is not a valid format.')

			for file in sample_files:
				fname = correct_file_ending(file, 'h5')

				with h5py.File(fname, 'r') as hf:

					for key in hf.keys():
						if self.H5_COL_KEY in key:
							new_tuples = np.asarray(hf[key][:], dtype=np.float16)
							new_tuples = new_tuples.reshape((*new_tuples.shape, 1))
							tuples = np.concatenate((tuples, new_tuples))

							while len(tuples) >= self.batch_size:
								if isinstance(self.subsampling_functions, (list, tuple, np.ndarray)):
									for fn in self.subsampling_functions:
										triplets = fn(tuples[:self.batch_size])
										yield tf.constant(triplets, dtype=tf.float16),tf.constant(triplets, dtype=tf.float16)
								else:
									triplets = self.subsampling_functions(tuples[:self.batch_size])
									yield triplets
								tuples = tuples[self.batch_size:]
		self.generator = generator

	def get_generator(self):
		return tf.data.Dataset.from_generator(
			self.generator,
			output_signature=(
				tf.TensorSpec(shape=(self.batch_size, 8, 8, 15, 1), dtype=tf.float16),
				tf.TensorSpec(shape=(self.batch_size, 8, 8, 15, 1), dtype=tf.float16)
			)
		)

		
	def number_samples(self):
		samples = 0
		sample_files = files_from_directory(os.path.abspath(self.sample_dir), file_type="h5")
		for file in sample_files:
			fname = correct_file_ending(file, 'h5')
			with h5py.File(fname, 'r') as hf:
				for key in hf.keys():
					if self.H5_COL_KEY in key:
						samples += len(hf[key])
		return samples
