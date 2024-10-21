#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CompressedImage, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from sensor_msgs_py import point_cloud2

class CameraProcessingNode(Node):
    def __init__(self):
        super().__init__('camera_processing_node')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Initialize camera with high resolution
        self.camera = cv2.VideoCapture(0)  # Use camera index 0
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if not self.camera.isOpened():
            self.get_logger().error('Failed to open camera')
            return

        # Check for calibration files
        self.calibration_file_left = 'camera_left_calibration.yaml'
        self.calibration_file_right = 'camera_right_calibration.yaml'
        self.calibration_files_exist = os.path.exists(self.calibration_file_left) and os.path.exists(self.calibration_file_right)

        if self.calibration_files_exist:
            self.load_calibration_data()

        # Publishers
        self.left_rect_pub = self.create_publisher(CompressedImage, '/camera/left/image_rect_compressed', 10)
        self.right_rect_pub = self.create_publisher(CompressedImage, '/camera/right/image_rect_compressed', 10)
        self.disparity_pub = self.create_publisher(DisparityImage, '/camera/disparity', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/camera/points', 10)
        
        # Timer for camera processing
        self.timer = self.create_timer(0.1, self.process_camera)  # 10 Hz
        
        # Stereo parameters (you may need to adjust these based on your camera setup)
        self.focal_length = 1000.0  # in pixels
        self.baseline = 0.1  # in meters
        
        self.get_logger().info('Camera Processing Node initialized')

    def load_calibration_data(self):
        fs_left = cv2.FileStorage(self.calibration_file_left, cv2.FILE_STORAGE_READ)
        fs_right = cv2.FileStorage(self.calibration_file_right, cv2.FILE_STORAGE_READ)
        
        self.K_left = fs_left.getNode('K').mat()
        self.D_left = fs_left.getNode('D').mat()
        self.K_right = fs_right.getNode('K').mat()
        self.D_right = fs_right.getNode('D').mat()

        fs_left.release()
        fs_right.release()

    def process_camera(self):
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        # Rotate image 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Split image into left and right
        height, width = frame.shape[:2]
        left_image = frame[:, :width//2]
        right_image = frame[:, width//2:]

        # Rectify images
        left_rect = self.rectify_image(left_image, 'left')
        right_rect = self.rectify_image(right_image, 'right')

        # Publish rectified images
        self.left_rect_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(left_rect))
        self.right_rect_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(right_rect))

        # Calculate and publish disparity
        disparity = self.calculate_disparity(left_rect, right_rect)
        self.publish_disparity(disparity)

        # Generate and publish point cloud
        point_cloud = self.generate_point_cloud(disparity, left_rect)
        self.pointcloud_pub.publish(point_cloud)

    def rectify_image(self, image, side):
        if self.calibration_files_exist:
            if side == 'left':
                undistorted = cv2.undistort(image, self.K_left, self.D_left)
            else:
                undistorted = cv2.undistort(image, self.K_right, self.D_right)
            return undistorted
        else:
            # If no calibration files exist, assume the image is already rectified
            return image

    def calculate_disparity(self, left_image, right_image):
        # Convert images to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Create StereoBM object
        stereo = cv2.StereoBM_create(numDisparities=16*10, blockSize=21)

        # Compute the disparity map
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        return disparity

    def publish_disparity(self, disparity):
        msg = DisparityImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.image = self.cv_bridge.cv2_to_imgmsg(disparity, encoding="32FC1")
        msg.f = self.focal_length
        msg.t = self.baseline  # Changed from T to t
        msg.min_disparity = float(np.min(disparity))
        msg.max_disparity = float(np.max(disparity))
        msg.delta_d = 0.125  # You may want to adjust this value
        self.disparity_pub.publish(msg)

    def generate_point_cloud(self, disparity, image):
        height, width = disparity.shape

        # Create a 3D points array
        points = np.zeros((height, width, 3), dtype=np.float32)

        # Generate x and y coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate Z coordinates
        Z = self.focal_length * self.baseline / (disparity + 1e-7)
        X = (x - width / 2) * Z / self.focal_length
        Y = (y - height / 2) * Z / self.focal_length

        points[:,:,0] = X
        points[:,:,1] = Y
        points[:,:,2] = Z

        # Get color information
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create point cloud
        points = points.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)

        # Remove invalid points
        mask = Z.reshape(-1) > 0
        points = points[mask]
        colors = colors[mask]

        # Create PointCloud2 message
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1)
        ]

        # Pack RGB values
        rgb_packed = colors[:, 0] << 16 | colors[:, 1] << 8 | colors[:, 2]

        # Combine XYZ and RGB
        points_with_color = np.c_[points, rgb_packed.astype(np.float32)]

        pc2 = point_cloud2.create_cloud(self.get_clock().now().to_msg(), "camera_link", fields, points_with_color)
        return pc2

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
