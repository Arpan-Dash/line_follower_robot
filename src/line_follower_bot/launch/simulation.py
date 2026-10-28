"""Launch Gazebo, the robot, bridge, and line follower."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the simulation launch description."""
    package_share = get_package_share_directory("line_follower_bot")
    default_world = os.path.join(package_share, "worlds", "line.sdf")
    world = LaunchConfiguration("world")
    use_rviz = LaunchConfiguration("use_rviz")
    description = PathJoinSubstitution(
        [FindPackageShare("line_follower_bot"), "description", "differential_drive_bot.urdf.xacro"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
                    )
                ),
                launch_arguments={"gz_args": [world, " -r"]}.items(),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": Command(["xacro ", description])}],
                output="screen",
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=["-topic", "robot_description", "-x", "0.0", "-y", "0.0", "-z", "0.12"],
                output="screen",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                parameters=[
                    {"config_file": os.path.join(package_share, "config", "controller.yaml")}
                ],
                output="screen",
            ),
            Node(
                package="line_follower_bot",
                executable="line_follower",
                parameters=[os.path.join(package_share, "config", "line_follower.yaml")],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
