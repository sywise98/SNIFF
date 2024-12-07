#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatusArray
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Int32MultiArray, Float32MultiArray, String, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
import math
import handwave_ros
import speech2item
import cv2
import numpy as np
from enum import Enum

class RobotState(Enum):
    LOOKING_FOR_PEOPLE = 1
    APPROACHING_POI = 2
    CONVERSATION = 3
    LOOKING_FOR_OBJECT = 4
    RETURNING_TO_IDLE = 5

class AssistantRobot(Node):
    def __init__(self):
        super().__init__('assistant_robot')
        
        self.state = RobotState.LOOKING_FOR_PEOPLE
        self.cv_bridge = CvBridge()
        
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(String, '/speech_text', self.speech_callback, 10)
        self.create_subscription(String, '/speech_completion', self.audio_listen_callback, 10)
        self.create_subscription(Bool, "/is_waving", self.handwave_callback, 10)
        self.create_subscription(String, '/extracted_item',self.extracted_item_str_callback, 10)
        self.create_subscription(Int32MultiArray,'yolo_bboxes', self.yolo_callback,qos_profile)
        self.create_subscription(Float32, '/object_depth', self.depth_callback, qos_profile)
        self.create_subscription(GoalStatusArray, '/NavigateToPose/_action/status', self.nav_status_callback, 10)

        handwave_ros.ros_main()

        #speech2item.main()
        self.object_found = ""
        self.stop_listen = False
        self.object_depth = 0.0
        self.conversation_started = False

        self.waypoints = [
            [1.0, 0.0, 0],      # Forward 1m
            [0.7, 0.7, 90],     # Diagonal right
            [0.0, 1.0, 180],    # Left side
            [-0.7, 0.7, 270],   # Back diagonal
            [0.0, 0.0, 0]       # Return to start
        ]

        self.home_pose = [0.0, 0.0, 0.0]  # Home coordinates [x, y, yaw]
        self.current_waypoint = 0
        self.nav_complete = True  # Start true to trigger first waypoint
        self.returning_home = False

        self.audio_listen_pub = self.create_publisher(String, '/audio_listen', 10)
        self.audio_speak_pub = self.create_publisher(String, '/audio_speak', 10)
        self.cmd_vel_pub = self.create_publisher(String, 'cmd_vel', 10)
        self.speech_trigger_pub = self.create_publisher(String, '/speech_trigger', 10)
        self.relative_nav_pub = self.create_publisher(Float32MultiArray, 'relative_goal', 10)
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)
   
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(0.1, self.state_machine_callback)
        self.get_logger().info('Assistant Robot initialized')

    def handwave_callback(self, msg):
        if(msg.data == True and self.state == RobotState.LOOKING_FOR_PEOPLE):
            self.state = RobotState.APPROACHING_POI
            self.get_logger().info('Hnadwave detected.')
    
    def extracted_item_str_callback(self, msg):
        if(self.state == RobotState.CONVERSATION):
            self.state = RobotState.LOOKING_FOR_OBJECT
        self.object_found = msg.data

    def state_machine_callback(self):
        if self.state == RobotState.LOOKING_FOR_PEOPLE:
            self.look_for_people()
        elif self.state == RobotState.APPROACHING_POI:
            self.approach_poi()
        elif self.state == RobotState.CONVERSATION:
            if not (self.conversation_started):
                self.conduct_conversation()
                self.conversation_started = True
        elif self.state == RobotState.LOOKING_FOR_OBJECT:
            self.look_for_object()
        elif self.state == RobotState.RETURNING_TO_IDLE:
            self.return_to_idle()

    def look_for_people(self):
        return #YOLO will handle this

    def approach_poi(self):
        if(self.nav_complete):
            self.state = RobotState.CONVERSATION
        return

    def conduct_conversation(self):
        output = String()
        output.data = "Hi. Is there something you are looking for? I can try to find it."
        self.audio_speak_pub.publish(output)
        return
    
    def look_for_object(self):
        if self.nav_complete:
            if not self.returning_home:
                if self.current_waypoint < len(self.waypoints):
                    # Get next waypoint
                    wp = self.waypoints[self.current_waypoint]
                    self.send_goal_pose(wp[0], wp[1], wp[2])
                    self.get_logger().info(f'Navigating to waypoint {self.current_waypoint + 1}/{len(self.waypoints)}')
                    self.current_waypoint += 1
                else:
                    # All waypoints visited, object not found
                    msg = String()
                    msg.data = f"Sorry. I was unable to find the {self.object_found}. I am returning to my home point."
                    self.audio_speak_pub.publish(msg)
                    
                    # Return to home
                    self.returning_home = True
                    self.send_goal_pose(
                        self.home_pose[0],
                        self.home_pose[1],
                        self.home_pose[2]
                    )
        return

    def return_to_idle(self):
        return
    
    def audio_listen_callback(self, msg):
        if(self.stop_listen==False):
            output = msg
            output.data = "start_listening"
            self.audio_listen_pub.publish(output)
            print("NOW LISTENING")
            self.stop_listen = True
    
    def yolo_callback(self, msg):
        if(self.state == RobotState.LOOKING_FOR_OBJECT):
            output = String()
            output.data = "I found it! The " + self.object_found + " is right here."
            print("Found "+ self.object_found)
            self.audio_speak_pub.publish(output)
            self.state = RobotState.RETURNING_TO_IDLE
        
        elif(self.state == RobotState.LOOKING_FOR_PEOPLE):
            print("Found person!")
            
            # Calculate center point of bounding box
            x1 = msg.data[0] / 10000.0  # Denormalize coordinates
            x2 = msg.data[2] / 10000.0
            bbox_center = (x1 + x2) / 2.0
            angle = (bbox_center - 0.5) * 120.0
            
            # Create and publish relative goal message
            goal_msg = Float32MultiArray()
            goal_msg.data = [self.object_distance, angle]
            self.relative_nav_pub.publish(goal_msg)
            self.state = RobotState.APPROACHING_POI
            self.nav_complete = False

    def depth_callback(self, msg):
        self.object_depth = msg.data
    
    def nav_status_callback(self, msg):
        if msg.status_list:
            current_status = msg.status_list[0].status
            if current_status == 4:  # SUCCEEDED
                self.nav_complete = True
                if self.returning_home:
                    self.state = RobotState.RETURNING_TO_IDLE

    def send_goal_pose(self, x, y, yaw_degrees):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set position
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw_rad = math.radians(yaw_degrees)
        q = quaternion_from_euler(0, 0, yaw_rad)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        
        self.goal_publisher.publish(goal_pose)
        self.nav_complete = False
        

    def speech_callback(self, msg):
    #     # Process speech-to-text result
    #     #print("LISTENED")
    #     #output = msg
    #     #self.audio_speak_pub.publish(output)
    #     #if self.state == RobotState.CONVERSATION:
    #         #self.process_user_request(msg.data)
        return

    # def navigate_to_pose(self, pose):
    #     goal_msg = NavigateToPose.Goal()
    #     goal_msg.pose = pose
        
    #     self.nav_client.wait_for_server()
    #     self.future = self.nav_client.send_goal_async(goal_msg)
    #     self.future.add_done_callback(self.navigation_response_callback)

    # def navigation_response_callback(self, future):
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().info('Goal rejected')
    #         return

    #     self.get_logger().info('Goal accepted')
    #     self.result_future = goal_handle.get_result_async()
    #     self.result_future.add_done_callback(self.navigation_result_callback)

    # def navigation_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info('Navigation finished')

    # def process_user_request(self, text):
    #     # TODO: Implement natural language processing to extract object and location
    #     # Once object and location are identified:
    #     #     self.state = RobotState.LOOKING_FOR_OBJECT
    #     return

def main(args=None):
    rclpy.init(args=args)
    node = AssistantRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()