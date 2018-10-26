import glob
import pickle
import music21
from music21 import converter, instrument, note, chord
import numpy as np
import Preprocessing as pp
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
	with open(fileName, 'rb') as f:
		modelX, modelY = pickle.load(f)
	return modelX, modelY


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


def makeData2(encodedFilename, ioFilename, sequenceLength):
	import Preprocessing2 as pp
	print("Making data...")
	with open(encodedFilename, 'rb') as f:
		encodedData = pickle.load(f)

	print('Creating model IO ...')
	modelX, modelY = pp.createModelIO(encodedData, sequenceLength, 1)

	with open(ioFilename, 'wb') as f:
		pickle.dump([modelX, modelY], f)
	return modelX, modelY


def preprocessData2(fileName):
	import Preprocessing2 as pp
	count = 1
	folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
	encodedData = []
	for file in folder:
		print('Processing file %d of %d: %s' % (count, len(folder), repr(file)))
		noteEvents = pp.readFileAsNoteEventList(file)
		encodedData.extend(pp.encodeNoteEvents(noteEvents))
		count += 1

	with open(fileName, 'wb') as f:
		pickle.dump(encodedData, f)
	return encodedData


def train2(sequenceLength):
	print('Trainer')
	modelX, modelY = loadData('model2.dat')
	modelInput = keras.layers.Input((sequenceLength, 257))
	model = m.getModel2(modelInput)
	m.trainModel(model, num_epochs = 1000, filename = 'model2.weights', x_train = modelX, y_train = modelY,
	             batch_size = 32)


def generate2():
	print('Generator')
	modelX, modelY = loadData('model2.dat')
	modelInput = keras.layers.Input(modelX.shape[1:])
	model = m.getModel2(modelInput)
	m.loadModel(model, 'model2.weights')

	start = np.random.randint(0, len(modelX) - 1)
	pattern = modelX[start]
	prediction_output = []

	for i in range(256):
		prediction_input = np.reshape(pattern, (1, len(pattern), 257))
		prediction = model.predict(prediction_input, verbose = 0)
		result = divMax(prediction)
		prediction_output.append(result)
		pattern = np.concatenate((pattern, result))
		pattern = pattern[1:]

	print('Generator finished')
	return prediction_output


def main():
	# preprocessData('intermediate.dat')

	# makeData('intermediate.dat', 'model.dat')
	# train()
	#
	# generated = generate()
	# with open('generated.dat', 'wb') as f:
	# 	pickle.dump(generated, f)
	#
	# with open('generated.dat', 'rb') as f:
	# 	generated = pickle.load(f)
	#
	# generated = np.reshape(generated, (128, 128))
	# print(np.shape(generated))
	#
	#
	# print('Main function done')

	sequenceLength = 16

	#preprocessData2('intermediate2.dat')
	#makeData2('intermediate2.dat', 'model2.dat', sequenceLength)
	train2(sequenceLength)
	generate2()


main()
print('End')
