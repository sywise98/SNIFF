#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float32
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class BBoxVisualizerNode(Node):
    def __init__(self):
        super().__init__('bbox_visualizer_node')
        
        # Create video capture
        self.cap = cv2.VideoCapture('/dev/video11')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to bounding box and depth topics
        self.bbox_subscription = self.create_subscription(
            Int32MultiArray,
            'yolo_bboxes',
            self.bbox_callback,
            qos_profile
        )
        
        self.depth_subscription = self.create_subscription(
            Float32,
            '/object_depth',
            self.depth_callback,
            qos_profile
        )
        
        # Store latest values
        self.current_bboxes = []
        self.current_depth = None
        
        # Create timer for visualization loop (30 fps)
        self.timer = self.create_timer(1/30.0, self.visualization_callback)
        
        self.get_logger().info('Visualizer Node initialized')

    def bbox_callback(self, msg):
        self.current_bboxes = msg.data

    def depth_callback(self, msg):
        self.current_depth = msg.data

    def visualization_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return
        
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))     
        height, width = frame.shape[:2]
        
        # Draw all current bboxes with depth
        if self.current_bboxes:
            # Convert normalized coordinates (0-10000) back to pixel coordinates
            x1 = int((self.current_bboxes[0] / 10000.0) * width)
            y1 = int((self.current_bboxes[1] / 10000.0) * height)
            x2 = int((self.current_bboxes[2] / 10000.0) * width)
            y2 = int((self.current_bboxes[3] / 10000.0) * height)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add depth label if available
            if self.current_depth is not None:
                label = f"{self.current_depth:.2f}m"
                # Get text size for background rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width + 10, y1), 
                            (0, 255, 0), 
                            -1)
                
                # Draw text
                cv2.putText(frame, 
                           label, 
                           (x1 + 5, y1 - 5), 
                           font, 
                           font_scale, 
                           (0, 0, 0), 
                           thickness)
        
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
