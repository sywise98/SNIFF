import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, String
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_world_node')
        
        self.item_subscription = self.create_subscription(
                String,
                'extracted_item',
                self.item_callback,
                10)

        self.publisher = self.create_publisher(Int32MultiArray,
                'yolo_bboxes',
                10)

        # Create named window first
        cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)

        # Open video device
        self.cap = cv2.VideoCapture('/dev/video9')
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open /dev/video9')
            return

        # Create timer that runs at 10Hz
        self.timer = self.create_timer(0.2, self.timer_callback)  # 0.1 seconds = 10Hz

        self.model = YOLO('yolov8s-world.pt')
        self.current_item = "person"

    def item_callback(self, msg):
        self.current_item = msg.data
        self.get_logger().info(f'Searching for {self.current_item}')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return

        self.model.set_classes([self.current_item])
        results = self.model(frame, verbose=False)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                bbox_msg = Int32MultiArray()
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                self.get_logger().info(f'{x1}, {y1}, {x2}, {y2}')
                bbox_msg.data = [int(x1), int(y1), int(x2), int(y2)]
                self.publisher.publish(bbox_msg)
                self.get_logger().info(f'Found {self.current_item} at {bbox_msg}')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('YOLO Detection', frame)
        cv2.waitKey(1)  # Add this line to properly update the window

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

def main(args = None):
    rclpy.init(args=args)
    yolo_node = YoloNode()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
