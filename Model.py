import keras
import tensorflow as tf
from keras.metrics import *


def getModel(modelInput):
	from keras.layers import Dropout, Dense, Activation, CuDNNLSTM, TimeDistributed, Conv1D, Reshape
	from keras.optimizers import Adam, RMSprop

	x = CuDNNLSTM(512, return_sequences=True, bias_initializer = 'random_uniform')(modelInput)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)

	x = CuDNNLSTM(128, return_sequences=False, bias_initializer = 'random_uniform')(x)
	modelOutput = Activation('sigmoid')(x)
	#
	# x = CuDNNLSTM(1024, return_sequences = True)(x)
	# x = Activation('relu')(x)
	# x = Dropout(0.3)(x)

	# x = CuDNNLSTM(512, return_sequences = True)(x)
	# x = Activation('relu')(x)
	# x = Dropout(0.3)(x)

	# x = CuDNNLSTM(256)(x)
	# x = Activation('relu')(x)
	# x = Dropout(0.3)(x)

	# x = Dense(256, activation = 'relu')(x)
	# x = Dropout(0.3)(x)
	#
	# modelOutput = Dense(128, activation = 'softmax')(x)

	model = keras.Model(modelInput, modelOutput)
	model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(), metrics = ['acc'])
	return model


def getModel2(modelInput):
	x = modelInput
	inputLayer = x
	x = keras.layers.TimeDistributed(keras.layers.Reshape((257, 1)))(x)
	x = keras.layers.TimeDistributed(keras.layers.Conv1D(16, 3, padding = 'same', activation = 'relu'))(x)
	x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.TimeDistributed(keras.layers.Conv1D(32, 3, padding = 'same', activation = 'relu'))(x)
	x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.TimeDistributed(keras.layers.Conv1D(64, 3, padding = 'same', activation = 'relu'))(x)
	x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
	x = keras.layers.CuDNNLSTM(256, return_sequences = False, bias_initializer = 'random_uniform')(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.Dense(257, activation = 'tanh')(x)
	modelOutput = x

	model = keras.Model(inputLayer, modelOutput)
	model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Nadam(), metrics = ['acc'])
	return model


def getDeltaTimeModel(modelInput):
	x = modelInput
	inputLayer = x
	x = keras.layers.Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Conv1D(64, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Conv1D(128, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.CuDNNLSTM(256, return_sequences = True, bias_initializer = 'random_uniform')(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.CuDNNLSTM(128, return_sequences = False, bias_initializer = 'random_uniform')(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Dense(64, activation = 'relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Dense(32, activation = 'relu')(x)
	x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Dense(1, activation = 'softmax')(x)
	modelOutput = x

	model = keras.Model(inputLayer, modelOutput)
	model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.RMSprop(), metrics = ['acc'])
	return model


def getAutoencoderModel(inputLayer):
	x = inputLayer

	# x = keras.layers.Conv1D(16, 3, padding = 'same', activation = 'linear')(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.Conv1D(32, 3, padding = 'same', activation = 'linear')(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.Conv1D(64, 3, padding = 'same', activation = 'linear')(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.CuDNNLSTM(128, return_sequences = True)(x)
	# x = keras.layers.Flatten()(x)
	# x = keras.layers.Dense(128)(x)
	# encoded = x
	# x = keras.layers.Dense(128 * sequenceLength)(x)
	# x = keras.layers.Reshape((sequenceLength, 128))(x)
	# x = keras.layers.CuDNNLSTM(64, return_sequences = True)(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.Conv1D(32, 3, padding = 'same', activation = 'linear')(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.Conv1D(16, 3, padding = 'same', activation = 'linear')(x)
	# x = keras.layers.Dropout(0.2)(x)
	# x = keras.layers.Conv1D(257, 3, padding = 'same', activation = 'softmax')(x)

	x = keras.layers.Reshape((256, 1))(x)
	x = keras.layers.Conv1D(16, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling1D(2)(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling1D(2)(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.Conv1D(64, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling1D(2)(x)
	x = keras.layers.Dropout(0.2)(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(128, activation = 'relu')(x)
	x = keras.layers.Dense(32, activation = 'relu')(x)
	x = keras.layers.Dense(16, activation = 'relu')(x)
	encoded = x
	x = keras.layers.Dense(32, activation = 'relu')(x)
	x = keras.layers.Dense(64, activation = 'relu')(x)
	x = keras.layers.Dense(2048, activation = 'relu')(x)
	x = keras.layers.Reshape((32, 64))(x)
	x = keras.layers.Dropout(0.1)(x)
	x = keras.layers.UpSampling1D(2)(x)
	x = keras.layers.Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Dropout(0.1)(x)
	x = keras.layers.UpSampling1D(2)(x)
	x = keras.layers.Conv1D(16, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Dropout(0.1)(x)
	x = keras.layers.UpSampling1D(2)(x)
	x = keras.layers.Conv1D(1, 3, padding = 'same', activation = 'relu')(x)
	x = keras.layers.Reshape((256,))(x)
	x = keras.layers.Dense(256, activation = 'tanh')(x)

	decoded = x
	outputLayer = x

	encoderModel = keras.Model(inputLayer, encoded)
	trainerModel = keras.Model(inputLayer, outputLayer)

	encoderModel.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.RMSprop(), metrics = ['acc'])
	trainerModel.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(lr = 0.001), metrics = ['acc'])

	return trainerModel, encoderModel


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
		POSITIVE_WEIGHT = 10
		_epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
		output = tf.log(output / (1 - output))
		# compute weighted loss
		loss = tf.nn.weighted_cross_entropy_with_logits(targets = target,
		                                                logits = output,
		                                                pos_weight = POSITIVE_WEIGHT)
		return tf.reduce_mean(loss, axis = -1)
	return f_


def lossFunction2(weight):
	def loss(yTrue, yPred):
		error = yPred - yTrue
		positive = tf.clip_by_value(error, clip_value_min = 0, clip_value_max = float('Inf'))
		negative = keras.backend.abs(tf.clip_by_value(error, clip_value_min = float('-Inf'), clip_value_max = 0))
		scaledNegative = negative * weight
		# summedError = (positive + scaledNegative)
		return keras.backend.mean(positive) + keras.backend.mean(scaledNegative)

	return loss


def lossFunction3(weight):
	def loss(yTrue, yPred):
		falsePositives = tf.ceil(tf.clip_by_value(yPred - yTrue, clip_value_min = 0, clip_value_max = float('Inf')))
		falseNegatives = tf.ceil(tf.clip_by_value(yTrue - yPred, clip_value_min = 0, clip_value_max = float('Inf')))
		return ((keras.backend.mean(falseNegatives) * weight) + keras.backend.mean(falsePositives)) / 2

	return loss


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

