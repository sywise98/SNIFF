#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String
import cv2
from ultralytics import YOLO
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_world_node')
        
        # Create video capture
        self.cap = cv2.VideoCapture('/dev/video10')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Publishers and Subscribers
        self.publisher = self.create_publisher(
            Int32MultiArray,
            'yolo_bboxes',
            10
        )
        
        self.item_subscription = self.create_subscription(
            String,
            'extracted_item',
            self.item_callback,
            10
        )

        # Initialize YOLO model
        self.model = YOLO('yolov8s-world.pt')
        self.current_item = "person"
        
        # Create timer for inference loop (30 fps)
        self.timer = self.create_timer(1/30.0, self.inference_callback)
        
        self.get_logger().info('YOLO Node initialized')

    def item_callback(self, msg):
        self.current_item = msg.data
        self.get_logger().info(f'Searching for {self.current_item}')

    # Modified part of YoloNode class
    def inference_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return
            
        height, width = frame.shape[:2]
        self.model.set_classes([self.current_item])
        results = self.model(frame)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                bbox_msg = Int32MultiArray()
                x1, y1, x2, y2 = box.xyxy[0]
                # Normalize coordinates to 0-1 range
                norm_coords = [
                    int(x1/width * 10000),
                    int(y1/height * 10000),
                    int(x2/width * 10000),
                    int(y2/height * 10000)
                ]
                bbox_msg.data = norm_coords
                self.publisher.publish(bbox_msg)
                self.get_logger().info(f'Found {self.current_item} at {bbox_msg.data}')
                
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    yolo_node = YoloNode()
    
    try:
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()