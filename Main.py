import glob
import music21
from music21 import converter, instrument, note, chord

notesAndChords = []
folder = glob.glob("C:/Users/Main/Documents/Data/Chopin/*.mid")
count = 1
for file in folder:
	print('Reading file %d of %d: %s' % (count, len(folder), repr(file)))
	score: music21.stream.Score = converter.parse(file)

	# Transpose to common key
	key: music21.key.Key = score.analyze('key')
	intervalToC: music21.interval.Interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
	score.transpose(intervalToC, inPlace = True)
	count += 1

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

print('End')
