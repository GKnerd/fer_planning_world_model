"""Bring up the world model server on its own.

Standalone smoke test: the node starts, optionally seeds a demo object, and
publishes planning-scene diffs on /planning_scene. Those diffs are only *applied*
when move_group's PlanningSceneMonitor is running to consume them — run this
alongside the MoveIt stack to actually see objects in the scene. On its own you
can still verify the node is up and inspect its authoritative view:

    ros2 launch fer_world_model world_model.launch.py
    ros2 topic echo /world_model_server/world          # latched world view
    ros2 service call /world_model_server/reconcile std_srvs/srv/Trigger {}
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
    seed_demo = LaunchConfiguration("seed_demo").perform(context).lower() == "true"
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
            "seed_demo": seed_demo,
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
            "seed_demo", default_value="true",
            description="Spawn a demo cylinder_1 on startup so there is something "
                        "to see when running standalone."),
        DeclareLaunchArgument(
            "log_level", default_value="info",
            description="Node log level (debug|info|warn|error|fatal)."),
    ]
