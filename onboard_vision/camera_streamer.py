#!/usr/bin/env python3

import cv2
import socket
import struct
import numpy as np
import time
import threading
import argparse
from collections import deque

class CameraServer:
    def __init__(self, port, device_id):
        self.port = port
        self.device_id = device_id
        self.running = True
        
        # Setup socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('0.0.0.0', port))
        self.socket.listen(5)
        
        # Setup camera capture
        self.camera = cv2.VideoCapture(device_id)
        if not self.camera.isOpened():
            raise Exception(f"Failed to open camera {device_id}")
            
        print(f"Server started on port {port} reading from /dev/video{device_id}")

    def handle_client(self, client_socket):
        print("Client connected")
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Compress frame
                _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                data = encoded.tobytes()
                message_size = struct.pack("L", len(data))
                
                # Send frame
                client_socket.sendall(message_size + data)
                
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()
            print("Client disconnected")

    def run(self):
        try:
            while self.running:
                client_socket, addr = self.socket.accept()
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        self.socket.close()
        if self.camera.isOpened():
            self.camera.release()

def main():
    parser = argparse.ArgumentParser(description='Camera Server')
    parser.add_argument('--side', choices=['left', 'right'], required=True,
                      help='Which camera to serve (left or right)')
    parser.add_argument('--port', type=int, default=65432,
                      help='Port to serve on (default: 65432)')
    
    args = parser.parse_args()
    
    # Use video9 for left, video10 for right
    device_id = 9 if args.side == 'left' else 10
    
    server = CameraServer(args.port, device_id)
    server.run()

if __name__ == "__main__":
    main()
