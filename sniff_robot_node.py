#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from enum import Enum

class RobotState(Enum):
    LOOKING_FOR_PEOPLE = 1
    APPROACHING_POI = 2
    CONVERSATION = 3
    LOOKING_FOR_OBJECT = 4
    NAVIGATING_TO_OBJECT = 5
    RETURNING_TO_IDLE = 6

class AssistantRobot(Node):
    def __init__(self):
        super().__init__('assistant_robot')
        
        self.state = RobotState.LOOKING_FOR_PEOPLE
        self.cv_bridge = CvBridge()
        
        # Subscribers
        self.create_subscription(CompressedImage, '/camera/left/image_rect_compressed', self.left_image_callback, 10)
        self.create_subscription(CompressedImage, '/camera/right/image_rect_compressed', self.right_image_callback, 10)
        self.create_subscription(Image, '/camera/disparity', self.disparity_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(String, '/speech_text', self.speech_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher('cmd_vel', 10)
        self.speech_trigger_pub = self.create_publisher(String, '/speech_trigger', 10)
        
        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Timer for state machine
        self.create_timer(0.1, self.state_machine_callback)
        
        self.get_logger().info('Assistant Robot initialized')

    def left_image_callback(self, msg):
        # Process left camera image
        left_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        # TODO: Implement image processing for person detection and gesture recognition

    def right_image_callback(self, msg):
        # Process right camera image
        right_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        # TODO: Implement image processing for person detection and gesture recognition

    def disparity_callback(self, msg):
        # Process disparity image
        disparity = self.cv_bridge.imgmsg_to_cv2(msg)
        # TODO: Implement distance estimation using disparity

    def lidar_callback(self, msg):
        # Process LIDAR data
        # TODO: Implement obstacle detection and distance estimation

    def speech_callback(self, msg):
        # Process speech-to-text result
        if self.state == RobotState.CONVERSATION:
            self.process_user_request(msg.data)

    def state_machine_callback(self):
        if self.state == RobotState.LOOKING_FOR_PEOPLE:
            self.look_for_people()
        elif self.state == RobotState.APPROACHING_POI:
            self.approach_poi()
        elif self.state == RobotState.CONVERSATION:
            self.conduct_conversation()
        elif self.state == RobotState.LOOKING_FOR_OBJECT:
            self.look_for_object()
        elif self.state == RobotState.NAVIGATING_TO_OBJECT:
            self.navigate_to_object()
        elif self.state == RobotState.RETURNING_TO_IDLE:
            self.return_to_idle()

    def look_for_people(self):
        # TODO: Implement person detection and distress sign recognition
        # If a person is detected:
        #     self.state = RobotState.APPROACHING_POI

    def approach_poi(self):
        # TODO: Implement navigation to the person of interest
        # Use disparity and LIDAR data to estimate distance
        # Navigate to a suitable conversation distance
        # Once reached:
        #     self.state = RobotState.CONVERSATION

    def conduct_conversation(self):
        # Trigger speech recognition
        self.speech_trigger_pub.publish(String(data='start_conversation'))

    def process_user_request(self, text):
        # TODO: Implement natural language processing to extract object and location
        # Once object and location are identified:
        #     self.state = RobotState.LOOKING_FOR_OBJECT

    def look_for_object(self):
        # TODO: Implement object detection using camera images
        # If object is found:
        #     self.state = RobotState.NAVIGATING_TO_OBJECT

    def navigate_to_object(self):
        # TODO: Implement navigation to the detected object
        # Once reached:
        #     Notify user
        #     self.state = RobotState.RETURNING_TO_IDLE

    def return_to_idle(self):
        # TODO: Implement navigation to idle location
        # Once reached:
        #     self.state = RobotState.LOOKING_FOR_PEOPLE

    def navigate_to_pose(self, pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        
        self.nav_client.wait_for_server()
        self.future = self.nav_client.send_goal_async(goal_msg)
        self.future.add_done_callback(self.navigation_response_callback)

    def navigation_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Navigation finished')

def main(args=None):
    rclpy.init(args=args)
    node = AssistantRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()