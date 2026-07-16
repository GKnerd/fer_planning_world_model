"""Bring up the world model server on its own.

Standalone smoke test: the node starts and serves ~/reconcile and ~/purge.
Scene mutation goes through MoveIt's /apply_planning_scene service, so without
move_group running those services honestly report failure (after apply_timeout).
You can still verify the node is up and inspect its authoritative view:

    ros2 launch fer_world_model world_model.launch.py
    ros2 topic echo /fer_planning_scene_world_model/world_state   # latched world view
    ros2 service call /fer_planning_scene_world_model/reconcile std_srvs/srv/Trigger {}
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs) -> List[Node]:
    # Resolve to concrete Python types here (inside OpaqueFunction). Passing a
    # LaunchConfiguration straight into a param dict yields a *string*, and
    # bool("false") is True — a classic footgun for boolean params.
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    log_level = LaunchConfiguration("log_level").perform(context)

    world_model_server = Node(
        package="fer_world_model",
        executable="world_model_server",
        name="fer_planning_scene_world_model",
        output="screen",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[{
            "use_sim_time": use_sim_time,
            "planning_frame": "base",
            "perceived_ttl": 2.0,     # s before a perceived object goes stale
            "prune_period": 1.0,      # s between stale sweeps
            "apply_timeout": 2.0,     # s to wait for move_group's ack
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
