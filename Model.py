import keras
import tensorflow
from keras.metrics import *


def getModel(modelInput):
	from keras.layers import Dropout, Dense, Activation, CuDNNLSTM
	from keras.optimizers import Adam
	x = CuDNNLSTM(4096, return_sequences=True)(modelInput)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)
	x = CuDNNLSTM(2048, return_sequences=True)(x)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)
	x = CuDNNLSTM(1024, return_sequences = True)(x)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)
	x = CuDNNLSTM(512, return_sequences = True)(x)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)
	x = CuDNNLSTM(256)(x)
	x = Activation('relu')(x)
	x = Dense(256)(x)
	modelOutput = Dense(128, activation = 'sigmoid')(x)

	model = keras.Model(modelInput, modelOutput)
	model.compile(loss = lossFunction(), optimizer = Adam(), metrics = ['acc'])
	return model


def lossFunction():
	def f_(target, output):
		"""
		   Weighted binary crossentropy between an output tensor
		   and a target tensor. POS_WEIGHT is used as a multiplier
		   for the positive targets.

		   Combination of the following functions:
		   * keras.losses.binary_crossentropy
		   * keras.backend.tensorflow_backend.binary_crossentropy
		   * tf.nn.weighted_cross_entropy_with_logits
		   """
		# transform back to logits
		import keras.backend.tensorflow_backend as tfb
		POSITIVE_WEIGHT = 30
		_epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
		output = tensorflow.clip_by_value(output, _epsilon, 1 - _epsilon)
		output = tensorflow.log(output / (1 - output))
		# compute weighted loss
		loss = tensorflow.nn.weighted_cross_entropy_with_logits(targets = target,
		                                                logits = output,
		                                                pos_weight = POSITIVE_WEIGHT)
		return tensorflow.reduce_mean(loss, axis = -1)
	return f_


def trainModel(model, num_epochs, filename, x_train, y_train, batch_size, sample_weight = None, class_weight = None,
               validation_data = None, validation_split = 0.2):
	filepath = 'weights/' + filename
	checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss', verbose = 0, save_weights_only = True,
	                             save_best_only = True,
	                             mode = 'auto', period = 1)
	tensor_board = keras.callbacks.TensorBoard(log_dir = 'logs/', histogram_freq = 0, batch_size = batch_size)
	history = model.fit(x_train, y_train, verbose = 1, batch_size = batch_size, epochs = num_epochs,
	                    validation_split = validation_split,
	                    class_weight = class_weight,
	                    callbacks = [checkpoint, tensor_board])

	return history


def loadModel(model, fileName):
	path = 'weights/' + fileName
	model.load_weights(path)

