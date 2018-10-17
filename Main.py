import glob
import music21
from music21 import converter, instrument, note, chord
import numpy as np
import Preprocessing as pp
import Model as m


def main():
	count = 1
	folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
	for file in folder:
		print('Reading file %d of %d: %s' % (count, len(folder), repr(file)))
		score = pp.readFile(file)
		pp.transposeScore(score)
		slices = pp.timeSliceScore(score, step = 0.125)
		numericSlices = pp.numerifySlices(slices)
		oneHotSlices = pp.encodeSlices(numericSlices)
		modelInput, modelOutput = pp.createModelIO(oneHotSlices, 64)
		model = m.getModel(modelInput)
		m.trainModel(model, num_epochs = 200, filename = 'model.weights', x_train = modelInput, y_train = modelOutput, batch_size = 64)


main()
print('End')
