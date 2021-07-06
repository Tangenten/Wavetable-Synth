import collections
import math
import multiprocessing

import numpy
from scipy import fftpack
from scipy import signal
import pygame
import pyaudio
import audiolazy

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = signal.butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = signal.lfilter(b, a, data)
	return y

def linearInterpolation(y1, y2, frac):
	return (y1 * (1.0 - frac) + y2 * frac)

def ampToDB(amp):
	if amp < 0.00001:
		return -200
	return 20.0 * math.log10(amp)

def DBToAmp(db):
	if db == 0:
		return 1
	return math.pow(10.0, db / 20)

class ADSR():
	def __init__(self, sampleRate, channels):
		self.sampleRate = sampleRate
		self.channels = channels
		
		self.attackTime = int(0.1 * self.sampleRate)
		self.decayTime = int(0.1 * self.sampleRate)
		self.sustainValue = 0.7
		self.releaseTime = int(0.5 * self.sampleRate)
		self.releaseValue = 1
		
		self.currentTime = 0
		self.currentValue = 0
		
		self.noteState = False
		self.activeState = False
		
	def render(self, samplesToRender):
		envelope = numpy.zeros(samplesToRender)
		if self.activeState:
			for i in range(0, samplesToRender, self.channels):
				if self.noteState:
					if self.currentTime <= self.attackTime:
						self.currentValue = numpy.interp(self.currentTime, [0, self.attackTime], [0, 1])
						for j in range(self.channels):
							envelope[i+j] = self.currentValue
						self.currentTime += 1
					elif self.currentTime <= (self.attackTime + self.decayTime):
						self.currentValue = numpy.interp(self.currentTime, [self.attackTime, self.attackTime + self.decayTime], [1, self.sustainValue])
						for j in range(self.channels):
							envelope[i+j] = self.currentValue
						self.currentTime += 1
					else:
						if self.sustainValue != 0:
							self.currentValue = self.sustainValue
							for j in range(self.channels):
								envelope[i+j] = self.currentValue
						else:
							self.activeState = False
							break
				elif self.currentTime <= self.attackTime + self.decayTime + self.releaseTime:
					self.currentValue = numpy.interp(self.currentTime, [self.attackTime + self.decayTime, self.attackTime + self.decayTime + self.releaseTime], [self.releaseValue, 0])
					for j in range(self.channels):
						envelope[i+j] = self.currentValue
					self.currentTime += 1
				else:
					self.activeState = False
					break
					
		return envelope
		
	def noteOn(self):
		self.noteState = True
		self.activeState = True
		self.currentTime = 0
		self.currentValue = 0
	
	def noteOff(self):
		self.noteState = False
		self.currentTime = self.attackTime + self.decayTime
		self.releaseValue = self.currentValue

class synthesizer():
	def __init__(self, sampleRate, channels):
		self.sampleRate = sampleRate
		self.channels = channels
		
		self.frequency = 440.0
		self.amplitude = 0.9
		self.phase = 0.0
		
		self.time = 0.0
		self.deltaTime = 1.0 / self.sampleRate
		
		self.waveTableLength = 1024
		self.waveTable = numpy.zeros(self.waveTableLength+1)
		self.waveTable[: self.waveTableLength] = 0.0
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
		self.readIndex = 0.0
		
		self.ADSR = ADSR(self.sampleRate, self.channels)

	def render(self, samplesToRender):
		waveform = numpy.empty(samplesToRender)
		
		for i in range(0, samplesToRender, self.channels):
			readIndexRoundDown = int(self.readIndex) % self.waveTableLength
			readIndexRoundUp = int(self.readIndex + 1) % self.waveTableLength
			readIndexFractional = self.readIndex - int(self.readIndex)
			
			value = linearInterpolation(self.waveTable[readIndexRoundDown], self.waveTable[readIndexRoundUp], readIndexFractional)
			for j in range(self.channels):
				waveform[i+j] = value * self.amplitude
			
			self.readIndex += self.readSpeed

		envelope = self.ADSR.render(samplesToRender)
		waveform *= envelope
		return waveform
		
	def setWavetable(self, waveTable):
		sig_fft = fftpack.fft(waveTable)
		freq = fftpack.fftfreq(sig_fft.size, d = 1 / 44100)
		sig_fft = sig_fft
		for i in range(len(freq)):
			if abs(freq[i]) > 20000:
				sig_fft[i] = 0
		
		waveTable = fftpack.ifft(sig_fft)
		waveTable = waveTable.real
		
		maxValue = max(abs(waveTable))
		if maxValue != 0:
			scale = (1.0 / maxValue)
			waveTable = scale * waveTable
			
		self.waveTable[ : self.waveTableLength] = waveTable
		self.baseFrequency = self.sampleRate / self.waveTableLength
		self.readSpeed = self.frequency / self.baseFrequency
		
	def noteIn(self, frequency):
		if self.frequency == frequency and self.ADSR.activeState:
			self.noteOff(frequency)
		else:
			self.noteOn(frequency)
			
	def noteOn(self, frequency):
		self.frequency = frequency
		self.readSpeed = self.frequency / self.baseFrequency
		self.ADSR.noteOn()
		
	def noteOff(self, frequency):
		self.ADSR.noteOff()
		
class audioHandler(multiprocessing.Process):
	def __init__(self):
		super(multiprocessing.Process, self).__init__()
		self.daemon = True
		
		self.callbackPipeChild, self.callbackPipeParent = multiprocessing.Pipe(duplex = False)
		self.samplesQueue = multiprocessing.Queue()
		self.messageQueue = multiprocessing.Queue()
		
		self.bitDepth = 32
		self.sampleRate = 44100
		self.channels = 2
		self.frameCount = self.sampleRate // 32
		self.bufferSize = self.frameCount * self.channels
		
		self.running = False
		
	def audioCallback(self, in_data, frame_count, time_info, status):
		if status == pyaudio.paOutputOverflow or status == pyaudio.paOutputUnderflow:
			print("Underflow / Overflow")
		
		samples = self.callbackPipeChild.recv()
		self.samplesQueue.put_nowait(samples)
		
		return (numpy.array(samples, dtype = numpy.float32), pyaudio.paContinue)
		
	def input(self):
		while not self.messageQueue.empty():
			message = self.messageQueue.get_nowait()
			messageKey = message[0]
			messageValue = message[1]
			
			if messageKey == "STOP":
				self.running = False
			elif messageKey == "NOTE":
				self.synth.noteIn(messageValue)
			elif messageKey == "WAVETABLE":
				self.synth.setWavetable(messageValue)
	
	def run(self):
		pyAudio = pyaudio.PyAudio()
		audioCallback = pyAudio.open(format = pyaudio.paFloat32, channels = self.channels, rate = self.sampleRate, frames_per_buffer = self.frameCount, stream_callback = self.audioCallback, output = True, start = False)
		audioCallback.start_stream()
		
		self.synth = synthesizer(self.sampleRate, self.channels)
		
		self.running = True
		while self.running:
			samples = self.synth.render(self.bufferSize)
			self.callbackPipeParent.send(samples)  # Blocking if size == 1
			self.input()
		
		audioCallback.stop_stream()
		audioCallback.close()
		pyAudio.terminate()
		self.terminate()
	
	def stop(self):
		self.messageQueue.put_nowait(["STOP", None])
		
	def sendWavetable(self, waveTable):
		self.messageQueue.put_nowait(["WAVETABLE", waveTable])
	
	def sendNote(self, frequency):
		self.messageQueue.put_nowait(["NOTE", frequency])


class graphicHandler:
	def __init__(self, audioHandler):
		pygame.init()
		pygame.display.set_caption("Wavetable Synth")
		
		self.clock = pygame.time.Clock()
		self.width = 720
		self.height = 720
		self.screen = pygame.display.set_mode((self.width, self.height))
		self.deltaTime = 0
		self.running = False

		self.audioHandler = audioHandler
		self.samples = collections.deque(maxlen = int((self.audioHandler.bufferSize)))
		self.samples.extend([0.0] * int((self.audioHandler.bufferSize)))
		
		self.wavetablePainter = wavetablePainter(self.width, self.height)
	
	def start(self):
		self.audioHandler.start()
		self.render()
	
	def stop(self):
		self.running = False
	
	def getAudioSamplesForFrame(self):
		while not self.audioHandler.samplesQueue.empty():
			self.samples.extend(self.audioHandler.samplesQueue.get_nowait())
		
		samplesToRender = int(self.audioHandler.sampleRate * self.deltaTime)
		samples = []
		for i in range(samplesToRender):
			if self.samples:
				samples.append(self.samples.popleft())
			else:
				samples.append(0.0)
		
		return samples
	
	def render(self):
		self.running = True
		while self.running:
			self.screen.fill((0, 0, 0))
			self.deltaTime = 0.016 if self.deltaTime == 0.0 else self.clock.get_time() / 1000
			
			samples = self.getAudioSamplesForFrame()
			
			points = self.wavetablePainter.render()
			for i in range(self.width - 1):
				pygame.draw.line(self.screen, [255,255,255,255], [i, points[i]], [i+1, points[i+1]])
			
			self.audioHandler.sendWavetable(self.wavetablePainter.waveTable)
			pygame.display.flip()
			self.clock.tick(60)
			self.input()
	
	def input(self):
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				self.audioHandler.stop()
				self.stop()
				pygame.quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_a:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("C4"))
				elif event.key == pygame.K_w:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("C#4"))
				elif event.key == pygame.K_s:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("D4"))
				elif event.key == pygame.K_e:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("D#4"))
				elif event.key == pygame.K_d:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("E4"))
				elif event.key == pygame.K_f:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("F4"))
				elif event.key == pygame.K_t:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("F#4"))
				elif event.key == pygame.K_g:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("G4"))
				elif event.key == pygame.K_y:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("G#4"))
				elif event.key == pygame.K_h:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("A4"))
				elif event.key == pygame.K_u:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("A#4"))
				elif event.key == pygame.K_j:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("B4"))
				elif event.key == pygame.K_k:
					self.audioHandler.sendNote(audiolazy.lazy_midi.str2freq("C5"))
					
				if event.key == pygame.K_v:
					self.wavetablePainter.bitcrushPixelTable(2)
				if event.key == pygame.K_c:
					self.wavetablePainter.rectificationDistortPixelTable()
				if event.key == pygame.K_x:
					self.wavetablePainter.bendDistortPixelTable()
				if event.key == pygame.K_z:
					self.wavetablePainter.smoothPixelTable()
					
		self.wavetablePainter.input(events)

class wavetablePainter():
	def __init__(self, width, height):
		self.width = width
		self.height = height
		
		self.clickPosition = (0, 0)
		self.releasePosition = (0, 0)
		self.currentPosition = (0, 0)
		self.previousPosition = (0,0)
		self.holdingState = False
		
		self.pixels = numpy.zeros(self.width)
		
		self.waveTableSize = 1024
		self.waveTable = numpy.zeros(self.waveTableSize)
	
	def input(self, events):
		for event in events:
			if event.type == pygame.MOUSEBUTTONDOWN:
				self.clickPosition = pygame.mouse.get_pos()
				self.holdingState = True
			elif event.type == pygame.MOUSEBUTTONUP:
				self.releasePosition = pygame.mouse.get_pos()
				self.holdingState = False
			
			if event.type == pygame.MOUSEMOTION:
				self.previousPosition = self.currentPosition
				self.currentPosition = pygame.mouse.get_pos()
				
				if self.holdingState:
					self.updatePixelTable()
					self.updateWaveTable()
					
	def updatePixelTable(self):
		xPixelCurrent = round(self.currentPosition[0])
		yPixelCurrent = round(self.currentPosition[1])
		xPixelPrevious = round(self.previousPosition[0])
		yPixelPrevious = round(self.previousPosition[1])
		
		xStart = min(xPixelCurrent, xPixelPrevious)
		xEnd = max(xPixelCurrent, xPixelPrevious)
		
		if xStart == xPixelCurrent:
			yStart = yPixelCurrent
			yEnd = yPixelPrevious
		else:
			yStart = yPixelPrevious
			yEnd = yPixelCurrent
		
		for i in range(xStart, xEnd + 1, 1):
			yValue = round(numpy.interp(i, [xStart, xEnd], [yStart, yEnd]))
			self.pixels[i] = yValue
		
		self.updateWaveTable()
			
	def bitcrushPixelTable(self, factor):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		self.pixels = self.pixels * factor
		self.pixels = numpy.ceil(self.pixels)
		self.pixels = self.pixels * (1 / factor)
		
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
		
		self.updateWaveTable()
		
	def tanDistortPixelTable(self, factor):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		for i in range(len(self.pixels)):
			self.pixels[i] = (2 / math.pi) * math.atan(self.pixels[i] * factor)
		
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
		
		self.updateWaveTable()
		
	def rectificationDistortPixelTable(self):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		for i in range(len(self.pixels)):
			if self.pixels[i] > 0:
				pass
			else:
				self.pixels[i] = -self.pixels[i]
		
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
		
		self.updateWaveTable()
		
	def bendDistortPixelTable(self):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		for i in range(1, len(self.pixels) - 1, 1):
			self.pixels[i] = self.pixels[int(math.log(i) * (self.width / math.log(self.width)))]
			
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
		
		self.updateWaveTable()
		
	def smoothPixelTable(self):
		self.pixels = (self.pixels / (self.height / 2))
		self.pixels = self.pixels - 1
		
		for i in range(16):
			for j in range(2, len(self.pixels) - 2, 1):
				p1 = self.pixels[j-2]
				p2 = self.pixels[j - 1]
				p3 = self.pixels[j + 1]
				p4 = self.pixels[j + 2]
				
				sum = p1 + p2 + p3 + p4
				sum /= 4
				self.pixels[j] = (self.pixels[j] + sum) / 2
		
		self.pixels = self.pixels + 1
		self.pixels = self.pixels * (self.height / 2)
		
		self.updateWaveTable()
		
	def render(self):
		return self.pixels
	
	def updateWaveTable(self):
		stepSize = self.width / self.waveTableSize
		increment = 0.0
		
		for i in range(self.waveTableSize):
			frac, whole = math.modf(increment)
			whole = int(whole)
			
			if i != self.waveTableSize - 1:
				yValue = linearInterpolation(self.pixels[whole], self.pixels[whole + 1], frac)
			else:
				yValue = self.pixels[whole]
			
			if yValue != 0:
				yValue = (yValue / (self.height / 2)) - 1
				yValue = -yValue
			else:
				yValue = 1
				
			self.waveTable[i] = yValue
			increment += stepSize
			
		return self.waveTable

if __name__ == '__main__':
	a = audioHandler()
	g = graphicHandler(a)
	g.start()
