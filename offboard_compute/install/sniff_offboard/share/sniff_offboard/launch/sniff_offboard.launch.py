from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo, GroupAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='localhost',
        description='Server IP address'
    )

    # Create a group for delayed nodes
    delayed_nodes = GroupAction([
        LogInfo(msg='Starting speech and YOLO nodes...'),
        
        Node(
            package='sniff_offboard',
            executable='item_extractor_node',
            name='item_extractor_node',
            output='screen'
        ),
        
        Node(
            package='sniff_offboard',
            executable='yolo_world_node',
            name='yolo_world_node',
            output='screen'
        ),
    ])

    # Left camera receiver command
    left_receiver_cmd = [
        'python3',
        os.path.join(get_package_share_directory('sniff_offboard'), 'launch', 'camera_receiver.py'),
        '--host', LaunchConfiguration('host')
    ]

    return LaunchDescription([
        # Add launch arguments
        host_arg,

        # Print startup message
        LogInfo(msg='\n\n=== Starting Camera Receiver Process ===\n'),
        
        # Start camera receiver
        ExecuteProcess(
            cmd=left_receiver_cmd,
            name='left_camera_receiver',
            output='log'
        ),

        LogInfo(msg='\nWaiting 10 seconds for camera initialization...\n'),

        # Delay other nodes by 10 seconds
        TimerAction(
            period=5.0,
            actions=[delayed_nodes]
        )
    ])