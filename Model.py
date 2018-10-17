import keras


def getModel(modelInput):
	from keras.layers import LSTM, Dropout, Dense, Activation
	from keras.optimizers import Adam
	x = LSTM(512, activation = 'relu', return_sequences=True, input_shape = (modelInput.shape[1], modelInput.shape[2]))(modelInput)
	x = Dropout(0.25)(x)
	x = LSTM(1024, activation = 'relu', return_sequences=True)(x)
	x = Dropout(0.3)(x)
	x = LSTM(512, activation = 'relu')(x)
	x = Dense(512)(x)
	modelOutput = Dense(128, activation = 'softmax')(x)

	model = keras.Model(modelInput, modelOutput)
	model.compile(loss = 'binary_crossentropy', optimizer = Adam(), metrics = ['acc'])
	return model


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