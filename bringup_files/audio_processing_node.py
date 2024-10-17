#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyaudio
import wave
import os
from gtts import gTTS
from playsound import playsound

class AudioProcessingNode(Node):
    def __init__(self):
        super().__init__('audio_processing_node')
        
        # Initialize Whisper model
        self.load_whisper_model()
        
        # Audio Params
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 10
        self.WAVE_OUTPUT_FILENAME = "temp_audio.wav"
        
        # Publishers and Subscribers
        self.audio_listen_sub = self.create_subscription(String, '/audio_listen', self.audio_listen_callback, 10)
        self.audio_speak_sub = self.create_subscription(String, '/audio_speak', self.audio_speak_callback, 10)
        self.speech_text_pub = self.create_publisher(String, '/speech_text', 10)
        
        self.get_logger().info('Audio Processing Node initialized')

    def load_whisper_model(self):
        try:
            self.model = whisper.load_model("tiny")
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {str(e)}')
            self.model = None

    def audio_listen_callback(self, msg):
        if msg.data == 'start_listening':
            self.capture_and_process_audio()

    def audio_speak_callback(self, msg):
        self.speak(msg.data)

    def capture_and_process_audio(self):
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
            
            frames = []

            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            if self.model:
                result = self.model.transcribe(self.WAVE_OUTPUT_FILENAME, language="en")
                text = result["text"]
                self.speech_text_pub.publish(String(data=text))
            else:
                self.get_logger().error("Whisper model not initialized")

        except Exception as e:
            self.get_logger().error(f"Error in audio processing: {str(e)}")

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en-GB', slow=False)
            tts.save("temp_speech.mp3")
            playsound("temp_speech.mp3")
            os.remove("temp_speech.mp3")
        except Exception as e:
            self.get_logger().error(f"Error in text-to-speech: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
