import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class Webcam(Node):
    def __init__(self):
        super().__init__('webcam')
        self.publisher = self.create_publisher(Image,
                'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.cv_bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    webcam = Webcam()
    rclpy.spin(webcam)
    webcam.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
