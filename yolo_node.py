import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_world_node')
        self.subscription = self.create_subscription (
                Image,
                'camera/image_raw',
                self.image_callback,
                10)
        
        self.item_subscription = self.create_subscription (
                String,
                'extracted_item',
                self.item_callback,
                10)

        self.publisher = self.create_publisher(Int32MultiArray,
                'yolo_bboxes',
                10)
        self.cv_bridge = CvBridge()

        self.model = YOLO('yolov8s-world.pt')
        self.current_item = "person"

    def item_callback(self, msg):
        self.current_item = msg.data
        self.get_logger().info(f'Searching for {self.current_item}')

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        self.model.set_classes([self.current_item])

        results = self.model(cv_image)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                bbox_msg = Int32MultiArray()
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                self.get_logger().info(f'{x1}, {y1}, {x2}, {y2}')
                bbox_msg.data = [int(x1), int(y1), int(x2), int(y2)]
                self.publisher.publish(bbox_msg)
                self.get_logger().info(f'Found {self.current_item} at {bbox_msg}')

def main(args = None):
    rclpy.init(args=args)
    yolo_node = YoloNode()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

