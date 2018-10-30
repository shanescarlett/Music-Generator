import music21
from music21 import converter, instrument, note, chord
import numpy as np


def parseTempo(bytes):
	# Returns milliseconds per beat
	return ((bytes[3] << 16) + (bytes[4] << 8) + bytes[5])/1000


def ticksToMilliseconds(msPerBeat, ticksPerBeat):
	return msPerBeat/ticksPerBeat


def max(first, second):
	if first < second:
		return second
	else:
		return first


def readFileAsNoteEventList(path):
	midiFile = music21.midi.MidiFile()
	midiFile.open(path, attrib = 'rb')
	midiFile.read()
	ticksPerQuarterNote = midiFile.ticksPerQuarterNote
	tracks = midiFile.tracks
	allEvents = []
	for track in tracks:
		event: music21.midi.MidiEvent
		timeCounter = 0
		for event in track.events:
			if event.isDeltaTime():
				timeCounter += event.time
			if event.type is 'SET_TEMPO':
				allEvents.append(['TEMPO', parseTempo(event.getBytes()), 0, timeCounter])
			if event.isNoteOn():
				allEvents.append(['NOTE_ON', event.pitch, event.velocity, timeCounter])
			if event.isNoteOff():
				allEvents.append(['NOTE_OFF', event.pitch, event.velocity, timeCounter])

	# Sort event ledger
	allEvents.sort(key=lambda x: x[3])

	# Write real times to events
	currentTempo = 500
	for event in allEvents:
		if event[0] is 'TEMPO':
			currentTempo = event[1]
		if event[0] is 'NOTE_ON' or event[0] is 'NOTE_OFF':
			event[3] = np.round(event[3] * ticksToMilliseconds(currentTempo, ticksPerQuarterNote))

	return [s for s in allEvents if s[0] != 'TEMPO']


def encodeNoteEvents(noteEvents):
	encoded = []
	lastNoteTime = noteEvents[0][3]
	currentLine = np.zeros(128+128+1)
	noteEvents.sort(key = lambda x: x[3])
	for event in noteEvents:
		if event[3] != lastNoteTime:
			encoded.append(currentLine)
			currentLine = np.zeros(128 + 128 + 1)
			deltaTime = event[3] - lastNoteTime
			currentLine[-1] = np.tanh(max(0, deltaTime / 1000))
			lastNoteTime = event[3]


		if event[0] is 'NOTE_ON':
			currentLine[event[1]] = event[2] / 128
		elif event[0] is 'NOTE_OFF':
			currentLine[event[1] + 128] = 1
	return encoded


def createModelIO(slices, sequenceSize, stride):
	networkInput = []
	networkOutput = []
	encodedLength = len(slices[0])
	# create input sequences and the corresponding outputs
	for i in range(0, len(slices) - sequenceSize, stride):
		networkInput.append(slices[i:i + sequenceSize])
		networkOutput.append(slices[i + sequenceSize])

	n_patterns = len(networkInput)
	networkInput = np.reshape(networkInput, (n_patterns, sequenceSize, encodedLength))
	return networkInput, np.asarray(networkOutput)