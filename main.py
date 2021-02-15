import pyaudio
import wave
import os
import matplotlib.pyplot as plt
import scipy
import keras
import numpy as np
from matplotlib import image
import librosa
import librosa.display

model = keras.models.load_model('/home/james/Downloads/model.h5')
wheezes = np.load('/home/james/Downloads/wheezedata.npy', allow_pickle=True)
both = np.load('/home/james/Downloads/bothdata.npy', allow_pickle=True)
crackles = np.load('/home/james/Downloads/cracklesdata.npy', allow_pickle=True)
none = np.load('/home/james/Downloads/data.npy', allow_pickle=True)
clip = []
clip.append(none[1][0].reshape(50, 245, 1))
clip.append(wheezes[1][0].reshape(50, 245, 1))
clip.append(wheezes[2][0].reshape(50, 245, 1))
clip.append(wheezes[2][0].reshape(50, 245, 1))
clip.append(both[8][0].reshape(50, 245, 1))
clip.append(both[4][0].reshape(50, 245, 1))
clip.append(none[5][0].reshape(50, 245, 1))
clip.append(crackles[8][0].reshape(50, 245, 1))
clip.append(crackles[4][0].reshape(50, 245, 1))
clip.append(crackles[5][0].reshape(50, 245, 1))

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22000
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "resources/test.wav"

def graph_spectrogram():
    filename = 'resources/test.wav'
    y, sr = librosa.load(filename)
    test, _ = librosa.effects.trim(y)
    n_mels = 90
    n_fft = 512
    hop_length = 512
    plt.tight_layout()
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    S = librosa.feature.melspectrogram(test, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.savefig('sp_xyz.png', dpi=300)
    (f, t, Sxx) = scipy.signal.spectrogram(test, fs=22000, nfft=512, nperseg=512)
    plt.plot(Sxx)
    plt.savefig('sp1_xyz.png', dpi=300)


def record():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    return wave_file


audio = record()
audio_fpath = "resources/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ", len(audio_clips))
graph_spectrogram()

image = image.imread('sp_xyz.png')

audio_clip = image.resize(50, 245, 1)
audio_clip = np.array(image, dtype=np.float32)
clip.append(audio_clip.reshape(50, 245, 1))
clip = np.array(clip)

prediction = model.predict(clip)
classes = np.argmax(prediction, axis=1)
print(classes)