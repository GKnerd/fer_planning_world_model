"""Bring up the world model server (read-only observer of the planning scene).

The node REQUIRES move_group
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs) -> List[Node]:
    
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    log_level = LaunchConfiguration("log_level").perform(context)

    world_model_server = Node(
        package="fer_world_model",
        executable="world_model_server",
        name="fer_planning_scene_world_model",
        output="both",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[{
            "use_sim_time": use_sim_time,
            "stale_ttl": 30.0,       # s without a diff before an object is reported stale
            "fetch_timeout": 5.0,   # s to wait for a /get_planning_scene answer
        }],
    )
    return [world_model_server]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        generate_declared_arguments() + [OpaqueFunction(function=launch_setup)])


def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="If true, use the simulated clock."),
        DeclareLaunchArgument(
            "log_level", default_value="info",
            description="Node log level (debug|info|warn|error|fatal)."),
    ]
