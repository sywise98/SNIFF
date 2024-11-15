import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
from PIL import Image as PILImage

class OwlVitNode(Node):
    def __init__(self):
        super().__init__('owl_vit_node')
        self.camera_subscriber = self.create_subscription(
                Image,
                '/stereo/image_rect/compressed',
                self.find_object,
                10)

        self.object_subscriber = self.create_subscription(
                String,
                'extracted_item',
                self.object_callback,
                10)
        
        self.bbox_publisher = self.create_publisher(
                Float32MultiArray,
                'owl_vit/bounding_boxes',
                10)
        self.cv_bridge = CvBridge()
        self.object = [["glasses"]]

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        

    def object_callback(self, msg):
        self.object =[[msg.data]]

        self.get_logger().info(f'Searching for {self.object}')

    def find_object(self,msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)

        inputs = self.processor(text=self.object, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])
        
        results = self.processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            print(f"Detected {self.object[0][label]} at location {box}")

            #Publish bounding boxes
            box_list = box.tolist()
            bbox_msg = Float32MultiArray()
            bbox_msg.data = box_list
            self.bbox_publisher.publish(bbox_msg)

def main(args=None):
    rclpy.init(args=args)
    owl_vit_node = OwlVitNode()
    rclpy.spin(owl_vit_node)
    owl_vit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

