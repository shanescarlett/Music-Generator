import music21
import numpy as np


def scaleUpValues(data):
	for line in data:
		for i in range(len(line) - 1):
			line[i] = np.round(line[i] * 127)
		line[256] *= 1000


def encodedToMidi(encoded):
	track = music21.midi.MidiTrack(1)
	scaleUpValues(encoded)
	for slice in encoded:
		# setTempo = music21.midi.MidiEvent(track, type = 'SET_TEMPO')
		# setTempo.data = b'\rD\xbd'
		# track.events.append(setTempo)
		deltaTime = music21.midi.DeltaTime(track, time = int(np.round(slice[256])), channel = 1)
		track.events.append(deltaTime)
		for i in range(len(slice) - 1 - 128):
			if slice[i] > 0:
				event = music21.midi.MidiEvent(track, type = 'NOTE_ON')
				event.pitch = i
				event.velocity = slice[i]
				# event.velocity = 64
				event.channel = 1
				track.events.append(event)
		for i in range(128, len(slice) - 1):
			if slice[i] > 10:
				event = music21.midi.MidiEvent(track, type = 'NOTE_OFF')
				event.pitch = i - 128
				event.velocity = slice[i]
				event.channel = 1
				track.events.append(event)

	track.updateEvents()

	file = music21.midi.MidiFile()
	file.ticksPerQuarterNote = 480
	file.tracks.append(track)

	# Sanitise output for writing
	stream = music21.midi.translate.midiFileToStream(file)
	stream.write('midi', fp='test.mid')