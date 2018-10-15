import glob
import music21
from music21 import converter, instrument, note, chord
import numpy as np


def getAllNotesPlayingAt(offsetStart, offsetEnd, stream):
	result = []
	notes = stream.getElementsByOffset(offsetStart = offsetStart, offsetEnd = offsetEnd, mustBeginInSpan = False,
	                                   mustFinishInSpan = False)
	for element in notes:
		if isinstance(element, note.Note):
			result.append(element.pitch)
		elif isinstance(element, chord.Chord):
			result.append(element.pitches)
	return result


def readFile(path):
	score: music21.stream.Score = converter.parse(path)
	return score


def transposeScore(score):
	key: music21.key.Key = score.analyze('key')
	intervalToC: music21.interval.Interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
	score.transpose(intervalToC, inPlace = True)
	return score


def timeSliceScore(score, step = 0.25):
	notes = []
	parts = instrument.partitionByInstrument(score)
	if parts:  # file has instrument parts
		notes_to_parse = parts.parts[0].recurse()
	else:  # file has notes in a flat structure
		notes_to_parse = score.flat.notes
	for element in notes_to_parse:
		if isinstance(element, note.Note):
			notes.append(element)
		elif isinstance(element, chord.Chord):
			notes.append(element)

	totalLength = score.quarterLength
	noteSlices = []
	for i in np.arange(0, totalLength - step, step):
		noteSlices.append(getNotesPlayingBetween(notes, i, i + step))

	return noteSlices


def getNotesPlayingBetween(notes, start, end):
	result = []
	for item in notes:
		noteStart = item.offset
		noteEnd = noteStart + item.quarterLength
		if noteStart <= end and start <= noteEnd:
			result.append(item)
	return result


def numerifySlices(slices):
	numericalNotes = []
	for slice in slices:
		concurrentPitches = []
		concurrentNumbers = []
		for item in slice:
			if isinstance(item, note.Note):
				concurrentPitches.append(item.pitch)
			elif isinstance(item, chord.Chord):
				concurrentPitches.extend(item.pitches)
		for pitch in concurrentPitches:
			concurrentNumbers.append(pitch.midi)
		numericalNotes.append(concurrentNumbers)

	return numericalNotes


def encodeSlices(slices):
	oneHotEncoded = []
	for slice in slices:
		oneHotVector = np.zeros(128)
		for value in slice:
			oneHotVector[value] = 1
		oneHotEncoded.append(oneHotVector)
	return oneHotEncoded


def main():
	count = 1
	folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
	for file in folder:
		print('Reading file %d of %d: %s' % (count, len(folder), repr(file)))
		score = readFile(file)
		transposeScore(score)
		slices = timeSliceScore(score, step = 0.125)
		numericSlices = numerifySlices(slices)
		oneHotSlices = encodeSlices(numericSlices)


main()
print('End')
