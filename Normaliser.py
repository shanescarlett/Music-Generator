import music21


def normaliseKeys(path):
	import glob
	count = 1
	folder = glob.glob(path + "/*.mid")
	for file in folder:
		print('Processing file %d of %d: %s' % (count, len(folder), repr(file)))
		count += 1
		score: music21.stream.Score = music21.converter.parse(file)
		transposeScore(score)
		score.write('midi', fp = file)


def transposeScore(score):
	key: music21.key.Key = score.analyze('key')
	intervalToC: music21.interval.Interval = music21.interval.Interval(key.tonic, music21.pitch.Pitch('C'))
	score.transpose(intervalToC, inPlace = True)