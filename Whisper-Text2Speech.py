# Testing Whisper from OpenAI -- Derrick Joyce
import whisper
import os
import pyttsx3
import pyaudio
import wave

# Load Model
model = whisper.load_model("base")

# Audio Params
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "input_audio.wav"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print ("Recording ...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("Recording.wav",'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#Record Audio
record_audio()

# Perform transcription
result = model.transcribe("Recording.wav")

# Print the transcribed text
print(result["text"])


# Initialize the TTS engine
#engine = pyttsx3.init()

# Convert transcribed text back to speech
#engine.say(result["text"])
#engine.runAndWait()
