#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import pyttsx3
import requests
import pyaudio
import wave
import numpy as np
import os

class AudioProcessingNode(Node):
    def __init__(self):
        super().__init__('audio_processing_node')
        
        # Initialize Whisper model
        self.load_whisper_model()
        
        # Initialize text-to-speech
        self.initialize_tts()
        
        # Set up audio recording
        self.initialize_audio()
        
        # Publishers and Subscribers
        self.speech_trigger_sub = self.create_subscription(String, '/speech_trigger', self.speech_trigger_callback, 10)
        self.speech_text_pub = self.create_publisher(String, '/speech_text', 10)
        self.object_location_pub = self.create_publisher(String, '/object_location', 10)
        
        # Speak initialization message
#        self.speak("Audio Processing Node initialized and ready.")
        
        self.get_logger().info('Audio Processing Node initialized')

    def load_whisper_model(self):
        try:
            self.model = whisper.load_model("base")
            self.get_logger().info('Whisper model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {str(e)}')
            self.model = None

    def initialize_tts(self):
        try:
            self.engine = pyttsx3.init()
            # Configure the TTS engine for more natural speech
            self.engine.setProperty('rate', 108)  # Speed of speech
            self.engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
            
            voices = self.engine.getProperty('voices')
            # Choose a more natural sounding voice (you may need to experiment with different indices)
            self.engine.setProperty('voice', voices[1].id)
            
            self.get_logger().info('Text-to-speech engine initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize text-to-speech: {str(e)}')
            self.engine = None

    def initialize_audio(self):
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.device_index = self.find_input_device()
            if self.device_index is not None:
                self.get_logger().info(f'Audio input device found at index {self.device_index}')
            else:
                self.get_logger().warn('No suitable audio input device found')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize audio: {str(e)}')
            self.audio = None

    def find_input_device(self):
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                return i
        return None

    def speech_trigger_callback(self, msg):
        if msg.data == 'start_conversation':
            self.start_conversation()

    def start_conversation(self):
        self.speak("Hello, is there something you are looking for?")
        
        if self.device_index is None:
            self.get_logger().error("No suitable input device found")
            return

        try:
            # Open the stream only when needed
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                                          input=True, frames_per_buffer=1024, 
                                          input_device_index=self.device_index)
            
            # Record audio
            frames = []
            for _ in range(0, int(16000 / 1024 * 5)):  # Record for 5 seconds
                data = self.stream.read(1024)
                frames.append(data)
            
            # Close the stream after recording
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            # Save audio to file
            with wave.open("temp_audio.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames))
            
            # Transcribe audio using Whisper
            if self.model:
                result = self.model.transcribe("temp_audio.wav")
                text = result["text"]
                self.speech_text_pub.publish(String(data=text))
                
                # Process text with LLM
                object_and_location = self.process_with_llm(text)
                self.object_location_pub.publish(String(data=object_and_location))
            else:
                self.get_logger().error("Whisper model not initialized")
        
        except Exception as e:
            self.get_logger().error(f"Error in audio processing: {str(e)}")

    def speak(self, text):
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                self.get_logger().error(f"Error in text-to-speech: {str(e)}")
        else:
            self.get_logger().warn("Text-to-speech engine not initialized")

    def process_with_llm(self, text):
        # TODO: Implement LLM processing (either via API or local GPU node)
        # For now, we'll use a placeholder API call
        try:
            response = requests.post('http://localhost:5000/process_text', json={'text': text})
            return response.json()['result']
        except Exception as e:
            self.get_logger().error(f"Error in LLM processing: {str(e)}")
            return "Error processing text"

    def __del__(self):
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio is not None:
            self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
