#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import numpy as np

class BBoxVisualizerNode(Node):
    def __init__(self):
        super().__init__('bbox_visualizer_node')
        
        # Create video capture
        self.cap = cv2.VideoCapture('/dev/video11')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Subscribe to bounding box topic
        self.bbox_subscription = self.create_subscription(
            Int32MultiArray,
            'yolo_bboxes',
            self.bbox_callback,
            10
        )
        
        # Store latest bboxes
        self.current_bboxes = []
        
        # Create timer for visualization loop (30 fps)
        self.timer = self.create_timer(1/30.0, self.visualization_callback)
        
        self.get_logger().info('Visualizer Node initialized')

    def bbox_callback(self, msg):
        # Convert normalized coordinates back to pixel coordinates
        self.current_bboxes = msg.data

    def visualization_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return
            
        height, width = frame.shape[:2]
        
        # Draw all current bboxes
        if self.current_bboxes:
            # Convert normalized coordinates (0-10000) back to pixel coordinates
            x1 = int((self.current_bboxes[0] / 10000.0) * width)
            y1 = int((self.current_bboxes[1] / 10000.0) * height)
            x2 = int((self.current_bboxes[2] / 10000.0) * width)
            y2 = int((self.current_bboxes[3] / 10000.0) * height)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLO Detections', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = BBoxVisualizerNode()
    
    try:
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
