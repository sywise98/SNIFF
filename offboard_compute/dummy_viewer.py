#!/usr/bin/env python3

import cv2
import sys

def test_virtual_device(device_path):
    cap = cv2.VideoCapture(device_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {device_path}")
        return
        
    window_name = f"Feed from {device_path}"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading from {device_path}")
            break
            
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_device.py /dev/videoX")
        sys.exit(1)
        
    test_virtual_device(sys.argv[1])