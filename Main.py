import glob
import music21
from music21 import converter, instrument, note, chord

def readAndParseFile(path):
	notesAndChords = []
	score: music21.stream.Score = converter.parse(path)

	# Transpose to common key
	key: music21.key.Key = score.analyze('key')
	intervalToC: music21.interval.Interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
	score.transpose(intervalToC, inPlace = True)

	parts = music21.instrument.partitionByInstrument(score)
	if parts:  # file has instrument parts
		notes_to_parse = parts.parts[0].recurse()
	else:  # file has notes in a flat structure
		notes_to_parse = score.flat.notes

	for element in notes_to_parse:
		if isinstance(element, note.Note):
			notesAndChords.append(element)
		elif isinstance(element, chord.Chord):
			notesAndChords.append(element)

	return notesAndChords

def main():
	count = 1
	notesAndChords = []
	folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
	for file in folder:
		print('Reading file %d of %d: %s' % (count, len(folder), repr(file)))
		notesAndChords.append(readAndParseFile(file))
		count += 1

main()
print('End')
