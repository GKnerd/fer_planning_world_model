"""Bring up the world model server (read-only observer of the planning scene).

The node REQUIRES move_group: at startup it blocks fetching the full scene via
/get_planning_scene (logging while it waits), then mirrors every diff from
/monitored_planning_scene. It never writes the scene.

Smoke test (with your MoveIt stack already running):

    ros2 launch fer_world_model world_model.launch.py
    ros2 topic echo /fer_planning_scene_world_model/world_state    # latched model view
    ros2 service call /fer_planning_scene_world_model/get_objects std_srvs/srv/Trigger {}

Then add/remove a collision object (RViz planning-scene panel or a script that
calls /apply_planning_scene) and watch the model follow.
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
            "stale_ttl": 2.0,       # s without a diff before an object is reported stale
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
