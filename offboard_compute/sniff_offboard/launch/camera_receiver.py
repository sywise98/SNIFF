#!/usr/bin/env python3

import socket
import cv2
import struct
import numpy as np
import time
import subprocess
import signal
import sys
import threading
from queue import Queue
import os
import argparse

class StreamReceiverSplitter:
    def __init__(self, host, port, side='left', output_devices=['/dev/video10']):
        self.host = host
        self.port = port
        self.side = side.lower()
        self.output_devices = output_devices
        self.running = True
        self.frame_queue = Queue(maxsize=2)
        self.ffmpeg_processes = []
        
        # FPS monitoring
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Setup virtual devices
        self.setup_virtual_devices()
        
    def setup_virtual_devices(self):
        try:
            # Get device numbers from paths
            device_nums = ','.join([dev.split('video')[-1] for dev in self.output_devices])
            
            # Unload existing module
            subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], 
                        stderr=subprocess.DEVNULL)
            
            # Load module with specified devices
            subprocess.run(['sudo', 'modprobe', 'v4l2loopback', 
                        f'devices={len(self.output_devices)}',
                        f'video_nr={device_nums}',
                        'card_label="Virtual Camera"',
                        'exclusive_caps=1'], check=True)
            
            # Wait for devices to be created
            max_attempts = 10
            for attempt in range(max_attempts):
                all_devices_ready = True
                for device in self.output_devices:
                    if not os.path.exists(device):
                        all_devices_ready = False
                        break
                if all_devices_ready:
                    break
                time.sleep(0.5)
                
            if not all_devices_ready:
                raise Exception("Timeout waiting for virtual devices to be created")
                
            # Set permissions
            for device in self.output_devices:
                subprocess.run(['sudo', 'chmod', '666', device])
                
            print(f"Virtual devices created and initialized: {', '.join(self.output_devices)}")
        except subprocess.CalledProcessError as e:
            print(f"Error setting up virtual devices: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    def setup_ffmpeg_process(self, device, frame_shape):
        height, width = frame_shape[:2]
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', '60',
            '-i', '-',
            '-f', 'v4l2',
            device
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_time
        
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            print(f"Receive/Write FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time

    def receive_frames(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            client_socket.connect((self.host, self.port))
            print(f"Connected to {self.side} stream at {self.host}:{self.port}")
            
            data = b""
            payload_size = struct.calcsize("L")
            
            while self.running:
                # Get message size
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet

                if not data:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                # Get data
                while len(data) < msg_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # Decode JPEG data
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if frame is not None:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame)
                    
        except Exception as e:
            print(f"Error in receive_frames: {e}")
        finally:
            client_socket.close()
            self.running = False

    def write_frames(self):
        first_frame = True
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                if first_frame:
                    for device in self.output_devices:
                        process = self.setup_ffmpeg_process(device, frame.shape)
                        self.ffmpeg_processes.append(process)
                    first_frame = False
                
                frame_bytes = frame.tobytes()
                for process in self.ffmpeg_processes:
                    try:
                        process.stdin.write(frame_bytes)
                    except:
                        continue
                
                self.update_fps()
                
            except:
                if self.running:
                    print("No frame received")
                continue

    def run(self):
        receive_thread = threading.Thread(target=self.receive_frames)
        receive_thread.start()
        
        write_thread = threading.Thread(target=self.write_frames)
        write_thread.start()
        
        def signal_handler(sig, frame):
            print("\nShutting down...")
            self.cleanup()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        receive_thread.join()
        write_thread.join()

    def cleanup(self):
        self.running = False
        
        for process in self.ffmpeg_processes:
            try:
                process.stdin.close()
                process.wait(timeout=2)
            except:
                process.kill()
        
        print("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='Stereo Stream Receiver')
    parser.add_argument('--host', type=str, required=True, help='Server IP address')

    args = parser.parse_args()

    receiver = StreamReceiverSplitter(
        host=args.host,
        port=65432,
        side='left',
        output_devices=['/dev/video10', '/dev/video11']
    )
    
    receiver.run()

if __name__ == "__main__":
    main()