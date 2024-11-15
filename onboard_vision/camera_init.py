#!/usr/bin/env python3

import cv2
import numpy as np
import subprocess
import threading
import time
import os
import signal
import sys

class StereoPreprocessor:
    def __init__(self):
        # Camera calibration matrices
        self.K1 = np.array([[597.210779215877, 0, 936.314567624786],
                [0, 596.320127434697, 547.717453300479],
                [0, 0, 1]])
        self.K2 = np.array([[595.407619093856, 0, 962.520311099857],
                    [0, 594.463887937182, 516.695039681949],
                    [0, 0, 1]])
        self.D1 = np.array([-0.00750873048498732, -0.00969497435763337, 0, 0, 0], dtype=np.float32)
        self.D2 = np.array([-0.00838410275653840, -0.00951651705635062, 0, 0, 0], dtype=np.float32)
        self.R = np.array([[0.999692059002392, -0.000905251771451415, 0.0247985420294908],
                    [0.000733108102552455, 0.999975580442755, 0.00694989717085000],
                    [-0.0248042278667995, -0.00692957700048863, 0.999668310612338]])
        self.T = np.array([-59.9379011895022, 0.0927496275234350, 0.311436645049106])

        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FPS, 60.0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.running = True
        self.init_rectification_maps()
        self.setup_virtual_devices()
        
    def setup_virtual_devices(self):
        try:
            subprocess.run(['sudo', 'modprobe', '-r', 'v4l2loopback'], 
                         stderr=subprocess.DEVNULL)
            
            # Create three devices: two for left (8,9) and one for right (10)
            subprocess.run(['sudo', 'modprobe', 'v4l2loopback', 
                          'devices=4',
                          'video_nr=7,8,9,10',
                          'card_label="Virtual Camera"',
                          'exclusive_caps=1'], check=True)
            
            # Wait for devices to be created
            time.sleep(1)
            
            # Set permissions
            for dev_num in [7, 8, 9, 10]:
                device = f'/dev/video{dev_num}'
                subprocess.run(['sudo', 'chmod', '666', device])
            
            print("Virtual devices created: /dev/video7 /dev/video8, /dev/video9, /dev/video10")
            
        except Exception as e:
            print(f"Error setting up virtual devices: {e}")
            sys.exit(1)

    def init_rectification_maps(self):
        img_size = (1920, 1080)
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, img_size, self.R, self.T, 
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, img_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, img_size, cv2.CV_32FC1)

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
            '-pix_fmt', 'yuv420p',
            device
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE)

    def run(self):
        processes = []
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                # Split and process frames
                width = frame.shape[1] // 2
                left_frame = frame[:, :width]
                right_frame = frame[:, width:]

                left_rect = cv2.remap(left_frame, self.map1x, self.map1y, cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_frame, self.map1x, self.map1y, cv2.INTER_LINEAR)

                right_out = cv2.rotate(left_rect, cv2.ROTATE_180)
                left_out = cv2.rotate(right_rect, cv2.ROTATE_180)

                # Initialize ffmpeg processes if needed
                if not processes:
                    processes = [
                        self.setup_ffmpeg_process('/dev/video7', left_out.shape),
                        self.setup_ffmpeg_process('/dev/video8', left_out.shape),
                        self.setup_ffmpeg_process('/dev/video9', left_out.shape),
                        self.setup_ffmpeg_process('/dev/video10', right_out.shape)
                    ]

                # Write to virtual devices
                frame_bytes = left_out.tobytes()
                processes[0].stdin.write(frame_bytes)
                processes[1].stdin.write(frame_bytes)
                processes[2].stdin.write(frame_bytes)
                processes[3].stdin.write(right_out.tobytes())

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup(processes)

    def cleanup(self, processes):
        self.running = False
        if self.camera.isOpened():
            self.camera.release()
            
        for process in processes:
            try:
                process.stdin.close()
                process.wait(timeout=2)
            except:
                process.kill()

def main():
    preprocessor = StereoPreprocessor()
    preprocessor.run()

if __name__ == "__main__":
    main()
