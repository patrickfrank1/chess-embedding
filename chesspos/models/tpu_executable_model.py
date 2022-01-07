import os
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from chesspos.models.trainable_model import TrainableModel
from chesspos.models.chessposition_inspectable_autoencoder import ChesspositionInspectableAutoencoderMixin

def use_tpu(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		try:
			device_name = os.environ['COLAB_TPU_ADDR']
			TPU_ADDRESS = 'grpc://' + device_name
			print('Found TPU at: {}'.format(TPU_ADDRESS))

			resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
			tf.config.experimental_connect_to_cluster(resolver)
			tf.tpu.experimental.initialize_tpu_system(resolver)
			strategy = tf.distribute.TPUStrategy(resolver)

			with strategy.scope():
				func(*args, **kwargs)

		except KeyError:
			print('TPU not found')
	return wrapper

class TPUCnnResnetAutoencoder(TrainableModel): #, ChesspositionInspectableAutoencoderMixin):
	def __init__(
		self,
		input_size,
		embedding_size,
		train_generator,
		test_generator,
		train_steps_per_epoch,
		test_steps_per_epoch,
		save_dir,
		hidden_layers=[],
		optimizer='rmsprop',
		loss=None,
		metrics=None,
		tf_callbacks = None
	):
		super().__init__(
			save_dir, train_generator, test_generator, train_steps_per_epoch,
			test_steps_per_epoch, optimizer, loss, metrics, tf_callbacks = tf_callbacks
		)

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_layers = hidden_layers
		self.filters = 128

		self.encoder = None
		self.decoder = None

	def build_model(self):
		# build encoder
		input_layer = layers.Input(shape=(8,8,15,1), dtype=tf.float16)

		conv_1 = layers.Conv3D(8, (2,2,15), activation="relu", padding="same")(input_layer)
		conv_1 = layers.BatchNormalization()(conv_1)
		print(conv_1.shape)
		conv_2 = layers.Conv3D(16, (2,2,15), activation="relu", padding="same")(conv_1)
		conv_2 = layers.BatchNormalization()(conv_2)
		print(conv_2.shape)
		conv_3 = layers.Conv3D(32, (2,2,15), activation="relu", padding="same")(conv_2)
		conv_3 = layers.BatchNormalization()(conv_3)
		print(conv_3.shape)
		conv_4 = layers.Conv3D(64, (2,2,15), activation="relu", padding="same")(conv_3)
		conv_4 = layers.BatchNormalization()(conv_4)
		conv_5 = layers.Conv3D(128, (2,2,15), activation="relu", padding="same")(conv_4)
		conv_5 = layers.BatchNormalization()(conv_5)
		conv_6 = layers.Conv3D(256, (2,2,15), activation="relu", padding="same")(conv_5)
		conv_6 = layers.BatchNormalization()(conv_6)
		conv_7 = layers.Conv3D(512, (2,2,15), activation="relu", padding="same")(conv_6)
		conv_7 = layers.BatchNormalization()(conv_7)
		
		final_conv_shape = conv_7.shape[1:]
		print("final_conv_shape:", final_conv_shape)
		encoder = layers.Flatten()(conv_7)
		encoder = layers.Dense(self.embedding_size, activation="relu")(encoder)

		encoder_model = keras.Model(inputs=input_layer, outputs=encoder, name='encoder')
		encoder_model.summary()
		self.encoder = encoder_model

		# build decoder
		decoder_input = layers.Input(shape=(self.embedding_size,))
		decoder_dense = layers.Dense(np.prod(final_conv_shape), activation="relu")(decoder_input)

		deconv_1 = layers.Reshape(final_conv_shape)(decoder_dense)
		deconv_1 = layers.Conv3DTranspose(256, (2,2,15), activation="relu", padding="same")(deconv_1)
		deconv_1 = layers.BatchNormalization()(deconv_1)
		deconv_2 = layers.Conv3DTranspose(128, (2,2,15), activation="relu", padding="same")(deconv_1)
		deconv_2 = layers.BatchNormalization()(deconv_2)
		deconv_3 = layers.Conv3DTranspose(64, (2,2,15), activation="relu", padding="same")(deconv_2)
		deconv_3 = layers.BatchNormalization()(deconv_3)
		deconv_4 = layers.Conv3DTranspose(32, (2,2,15), activation="relu", padding="same")(deconv_3)
		deconv_4 = layers.BatchNormalization()(deconv_4)
		deconv_5 = layers.Conv3DTranspose(16, (2,2,15), activation="relu", padding="same")(deconv_4)
		deconv_5 = layers.BatchNormalization()(deconv_5)
		deconv_6 = layers.Conv3DTranspose(8, (2,2,15), activation="relu", padding="same")(deconv_5)
		deconv_6 = layers.BatchNormalization()(deconv_6)
		deconv_7 = layers.Conv3DTranspose(1, (2,2,15), activation="sigmoid", padding="same")(deconv_6)

		decoder_model = keras.Model(inputs=decoder_input, outputs=deconv_7, name='decoder')
		decoder_model.summary()
		self.decoder = decoder_model

		autoencoder = encoder_model(input_layer)
		autoencoder = decoder_model(autoencoder)

		autoencoder_model = keras.Model(inputs=input_layer, outputs=autoencoder, name='autocoder')
		autoencoder_model.summary()
		self.model = autoencoder_model

	def compile(self):

		try:
			device_name = os.environ['COLAB_TPU_ADDR']
			TPU_ADDRESS = 'grpc://' + device_name
			print('Found TPU at: {}'.format(TPU_ADDRESS))

			resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
			tf.config.experimental_connect_to_cluster(resolver)
			tf.tpu.experimental.initialize_tpu_system(resolver)
			strategy = tf.distribute.TPUStrategy(resolver)

			with strategy.scope():
				self.build_model()
				super().compile()
				self.encoder.compile(optimizer='rmsprop', loss=None, metrics=None)
				self.decoder.compile(optimizer='rmsprop', loss=None, metrics=None)

		except KeyError:
			print('TPU not found')   


	def get_encoder(self):
		if self.encoder is None:
			raise Exception("No encoder model defined.")
		else:
			return self.encoder


	def get_decoder(self):
		if self.decoder is None:
			raise Exception("No decoder model defined.")
		else:
			return self.decoder
