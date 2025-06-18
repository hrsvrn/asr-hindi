import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate (Hz)
seconds = 10  # Duration of recording

print("Recording...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file
print("Recording finished.")
