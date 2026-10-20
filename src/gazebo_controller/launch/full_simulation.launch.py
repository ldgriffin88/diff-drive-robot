import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('gazebo_controller')

    # Declare maze selection argument
    maze_arg = DeclareLaunchArgument(
        'maze',
        default_value='basic_maze',
        description='Maze to load: basic_maze or Maze_hr'
    )
    
    def launch_setup(context):
        maze_folder = context.perform_substitution(LaunchConfiguration('maze'))
        world_file = os.path.join(pkg_share, 'sdf', maze_folder, 'maze_world.sdf')

        # Launch Gazebo with the world
        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
                )
            ]),
            launch_arguments={'gz_args': f'-r {world_file}'}.items(),
        )

        # Include spawn entities launch file (robot, goals, map, static transforms)
        spawn_entities = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'spawn_entities.launch.py')
            ),
            launch_arguments={'maze': maze_folder}.items()
        )

        # Bridge, PID, RViz
        bridge = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'ros_gz_bridge.launch.py')
            )
        )

        pid_controller = Node(
            package='gazebo_controller',
            executable='diffdrive_pid',
            name='diffdrive_pid',
            output='screen',
            parameters=[
                {'kp': 0.5},
                {'kd': 0.05},
                {'ki': 0.0},
                {'lookahead': 0.03},
                {'publish_rate': 30.0},
                {'max_linear_vel': 0.7},
                {'max_angular_vel': 1.0},
                # {'use_sim_time': True}
            ]
        )

        mapper = Node(
            package='gazebo_controller',
            executable='mapper2',
            name='mapper2',
            output='screen',
            # parameters=[{'use_sim_time': True}]
        )

        planner = Node(
            package='gazebo_controller',
            executable='planner',
            name='planner',
            output='screen',
            # parameters=[{'use_sim_time': True}]
        )

        rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_share, 'rviz', 'rviz_view.rviz')],
            # parameters=[{'use_sim_time': True}]
        )

        return [
            gazebo,
            spawn_entities,
            bridge,
            pid_controller,
            mapper,
            planner,
            rviz
        ]

    return LaunchDescription([
        # SetParameter(name='use_sim_time', value=True),
        maze_arg,
        OpaqueFunction(function=launch_setup),
    ])