import glob
import pickle
import music21
from music21 import converter, instrument, note, chord
import numpy as np
import Preprocessing as pp
import Normaliser as norm
import Model as m
import keras


def preprocessData(fileName):
	count = 1
	folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
	encodedData = []
	for file in folder:
		print('Processing file %d of %d: %s' % (count, len(folder), repr(file)))
		score = pp.readFile(file)
		pp.transposeScore(score)
		slices = pp.timeSliceScore(score, step = 0.125)
		numericSlices = pp.numerifySlices(slices)
		encodedData.extend(pp.encodeSlices(numericSlices))
		count += 1

	with open(fileName, 'wb') as f:
		pickle.dump(encodedData, f)
	return encodedData


def makeData(encodedFileName, fileName):
	with open(encodedFileName, 'rb') as f:
		encodedData = pickle.load(f)

	print('Creating model IO ...')
	modelX, modelY = pp.createModelIO(encodedData, 64, 4)

	with open(fileName, 'wb') as f:
		pickle.dump([modelX, modelY], f)
	return modelX, modelY


def loadData(fileName):
	x = np.load(fileName + '_x.npy')
	y = np.load(fileName + '_y.npy')
	return x, y


def train():
	print('Trainer')
	modelX, modelY = loadData('model.dat')
	modelInput = keras.layers.Input((64, 128))
	model = m.getModel(modelInput)
	m.trainModel(model, num_epochs = 1000, filename = 'model.weights', x_train = modelX, y_train = modelY,
	             batch_size = 32)


def generate():
	print('Generator')
	modelX, modelY = loadData('model.dat')
	modelInput = keras.layers.Input(modelX.shape[1:])
	model = m.getModel(modelInput)
	m.loadModel(model, 'model.weights')

	start = np.random.randint(0, len(modelX) - 1)
	pattern = modelX[start]
	prediction_output = []

	for i in range(256):
		prediction_input = np.reshape(pattern, (1, len(pattern), 128))
		prediction = model.predict(prediction_input, verbose = 0)
		result = divMax(prediction)
		prediction_output.append(result)
		pattern = np.concatenate((pattern, result))
		pattern = pattern[1:]

	print('Generator finished')
	return prediction_output


def divMax(inp):
	min = np.min(inp)
	max = np.max(inp)
	mid = (max - min)/2
	test = inp[0]
	le = len(inp)
	for i in range(len(inp[0])):
		if inp[0][i] < mid:
			inp[0][i] = 0
		elif inp[0][i] >= mid:
			inp[0][i] = 1
	return inp

# -----------------------------


def makeData2(encodedFilename, ioFilename, sequenceLength, stride = 1, outputLength = 1):
	import Preprocessing2 as pp
	print("Making data...")
	# with open(encodedFilename, 'rb') as f:
	# 	encodedData = pickle.load(f)
	encodedData = np.load(encodedFilename + '.npy')

	print('Creating model IO ...')
	modelX, modelY = pp.createModelIO(encodedData, sequenceLength, stride, outputLength)

	# with open(ioFilename, 'wb') as f:
	# 	pickle.dump([modelX, modelY], f)
	np.save(ioFilename+'_x', np.asarray(modelX))
	np.save(ioFilename+'_y', np.asarray(modelY))
	return modelX, modelY


def preprocessData2(fileName):
	import Preprocessing2 as pp
	count = 1
	folder = glob.glob("C:/Users/Main/Documents/Data/Piano/*.mid")
	encodedData = []
	print('Pre-processing data ...')
	for file in folder:
		# print('Processing file %d of %d: %s' % (count, len(folder), repr(file)))
		printProgressBar(count, len(folder), length = 40)
		noteEvents = pp.readFileAsNoteEventList(file)
		enc = pp.encodeNoteEvents(noteEvents)
		encodedData.extend(enc)
		count += 1


	# np.savetxt('test.csv', np.asarray(encodedData), delimiter = ',')
	# with open(fileName, 'wb') as f:
	# 	pickle.dump(encodedData, f)
	np.save(fileName, encodedData)
	return encodedData


def train2(sequenceLength, ioFilename, modelFilename, loadPrevious = False):
	print('Trainer')
	modelX, modelY = loadData(ioFilename)
	modelInput = keras.layers.Input((sequenceLength, 257))
	model = m.getModel2(modelInput)
	model.summary()
	if loadPrevious:
		m.loadModel(model, modelFilename)
	checkpoint = keras.callbacks.ModelCheckpoint(modelFilename, monitor = 'val_acc', verbose = 1, save_best_only = True,
	                                             mode = 'max')
	model.fit(x = modelX, y = modelY, epochs = 1000, batch_size = 1024, validation_split = 0.1, callbacks = [checkpoint], verbose = 0)


def trainDeltaTime(sequenceLength, ioFilename, modelFilename, loadPrevious = False):
	print('Trainer')
	modelX, modelY = loadData(ioFilename)
	modelY = modelY[:, -1:]
	modelInput = keras.layers.Input((sequenceLength, 257))
	model = m.getDeltaTimeModel(modelInput)
	model.summary()
	if loadPrevious:
		m.loadModel(model, modelFilename)
	checkpoint = keras.callbacks.ModelCheckpoint(modelFilename, monitor = 'val_acc', verbose = 1, save_best_only = True,
	                                             mode = 'max')
	model.fit(x = modelX, y = modelY, epochs = 1000, batch_size = 1024, validation_split = 0.1, callbacks = [checkpoint], verbose = 1)


def generate2(length, stride = None):
	print('Generator')
	modelX, modelY = loadData('model2')
	modelInput = keras.layers.Input(modelX.shape[1:])
	model = m.getModel2(modelInput)
	model.load_weights('m2.h5')
	# m.loadModel(model, 'm2.h5')
	# model = keras.models.load_model('m2.h5')

	start = np.random.randint(0, len(modelX) - 1)
	pattern = modelX[start]
	prediction_output = []

	for i in range(length):
		prediction_input = np.reshape(pattern, (1, len(pattern), 257))
		prediction = model.predict(prediction_input, verbose = 0)
		# result = divMax(prediction)
		result = prediction
		if stride is not None and i % stride == 0:
			newRandom = np.random.randint(0, len(modelX) - 1)
			prediction_output.append(modelX[newRandom, 0:1, :])
		else:
			prediction_output.append(result)
		pattern = np.concatenate((pattern, result))
		pattern = pattern[1:]

	print('Generator finished')
	output = np.array(prediction_output)
	output = output.reshape((length, 257))

	np.savetxt('test.csv', np.asarray(output), delimiter = ',')
	return output


def generateDeltaTime(length, stride = None):
	print('Generator')
	modelX, modelY = loadData('model2')
	modelInput = keras.layers.Input(modelX.shape[1:])
	model = m.getDeltaTimeModel(modelInput)
	model.load_weights('m2dt.h5')
	# m.loadModel(model, 'm2.h5')
	# model = keras.models.load_model('m2.h5')

	pred = model.predict(modelX)
	start = np.random.randint(0, len(modelX) - 1)
	pattern = modelX[start]
	prediction_output = []

	for i in range(length):
		prediction_input = np.reshape(pattern, (1, len(pattern), 257))
		prediction = model.predict(prediction_input, verbose = 0)
		# result = divMax(prediction)
		result = prediction
		if stride is not None and i % stride == 0:
			newRandom = np.random.randint(0, len(modelX) - 1)
			prediction_output.append(modelX[newRandom, 0:1, :])
		else:
			prediction_output.append(result)
		pattern = np.concatenate((pattern, result))
		pattern = pattern[1:]

	print('Generator finished')
	output = np.array(prediction_output)
	output = output.reshape((length, 1))

	# np.savetxt('test.csv', np.asarray(output), delimiter = ',')
	return output


def save2(data):
	import Postprocessing as pp
	pp.encodedToMidi(data)


def main():
	sequenceLength = 16
	outputLength = 1
	stride = 4

	# norm.normaliseKeys('C:/Users/Main/Documents/Data/Piano')

	# preprocessData2('intermediate2')
	makeData2('intermediate2', 'model2', sequenceLength, stride, outputLength)
	# train2(sequenceLength, 'model2', 'm2.h5', loadPrevious = False)
	trainDeltaTime(sequenceLength, 'model2', 'm2dt.h5', loadPrevious = False)
	deltaTime = generateDeltaTime(length = 128)
	output = generate2(length = 1024, stride = 16)
	save2(output)


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>'):
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '')
	# Print New Line on Complete
	if iteration == total:
		print()


if __name__ == "__main__":
	main()
