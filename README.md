# diff-drive-robot

A ROS 2 navigation project for a simulated differential-drive robot in Gazebo. The robot builds an occupancy grid from lidar measurements, estimates its pose using wheel odometry and IMU data, plans paths with A*, and follows waypoints using a PID-based controller.

Developed for **Introduction to Robots (EN.605.613)** at **Johns Hopkins University**, Fall 2025.

> **Implementation note:** Mapping and path planning use an Extended Kalman Filter (EKF) pose estimate. In this version of the simulation, the PID controller uses the simulated robot's **ground-truth pose** via the `odom` to `vehicle_blue` TF transform. The system therefore does not use EKF localization throughout the entire control loop, and it is **not a full EKF-SLAM implementation**.

## Technologies

- **Python** and **ROS 2 Jazzy**
- **Gazebo** for robot and environment simulation
- **RViz** for visualizing the robot and occupancy grid
- **NumPy** for numerical operations
- **Docker** for the provided ROS 2 development environment

## Project Overview

I implemented three main ROS 2 nodes: `mapper2`, `planner`, and `diffdrive_pid`. The course-provided framework included the simulated robot, maze environments, map publisher, and goal-point publisher.

The navigation pipeline is:

1. The **mapper** predicts the robot's motion from wheel odometry and updates its orientation estimate using IMU measurements in an EKF.
2. Lidar scans are projected into the maze frame using the estimated pose to build an **occupancy grid**.
3. The **planner** uses the occupancy grid and EKF-derived pose to select a goal and compute a path with **A***.
4. The planner publishes an intermediate waypoint to the **PID controller**.
5. The controller converts the waypoint into linear and angular velocity commands, using **simulation ground-truth pose feedback** rather than the EKF estimate.

### Main ROS 2 interfaces

| Node | Subscribes to / reads | Publishes |
| --- | --- | --- |
| `mapper2` | `/odom`, `/imu`, `lidar`, TF | `/est_pose`, `/occ_map` |
| `planner` | `/est_pose`, `/occ_map`, `/goal_points` | `/goal_pose` |
| `diffdrive_pid` | `/goal_pose`, TF (`odom` to `vehicle_blue`) | `/cmd_vel` |

## Implementation

### 1. Localization and Mapping (`mapper2.py`)

The mapper maintains an estimated robot state of **x, y, and heading**.

- **EKF prediction:** Integrates linear and angular velocities from wheel odometry using a differential-drive motion model.
- **EKF measurement update:** Uses IMU yaw as an orientation measurement to correct the predicted state.
- **Lidar mapping:** Projects scan rays from the robot's estimated pose into an occupancy grid aligned with the `maze_world` frame.
- **Grid updates:** Accumulates ray traversal counts (free-space observations) and endpoint counts (obstacle observations), then classifies cells as free, occupied, or unknown.
- **Obstacle inflation:** Applies a configurable buffer around occupied cells to provide additional clearance for path planning.

The mapper publishes `/est_pose` and `/occ_map` for the planner. The lidar observations update the map but **do not feed back into the EKF** to correct localization. As a result, pose drift or scan/pose timing differences can affect map alignment.

### 2. Path Planning (`planner.py`)

The planner receives the occupancy grid and estimated robot pose from the mapper, then:

- Selects the **closest remaining goal** based on the estimated robot position.
- Converts robot and goal positions from world coordinates into grid coordinates.
- Runs **A*** with four-directional movement and a Manhattan-distance heuristic.
- Clears a small region around the selected goal in its planning copy of the grid, helping A* find a route to goals near inflated obstacles.
- Chooses a configurable waypoint farther along the computed path instead of commanding every adjacent cell.
- Publishes the selected waypoint as `/goal_pose` in `maze_world` coordinates.

**Path fallback:** If the start-cell occupancy check detects an obstacle, the planner attempts to reuse the remaining portion of its previous path. A search that otherwise finds no route returns an empty path; it does not always retain the old path.

### 3. Motion Control (`diffdrive_pid.py`)

The controller receives `/goal_pose`, transforms it into the `odom` frame, and computes velocity commands using a **lookahead-point (trailer-hitch) PID approach**.

- Calculates a point ahead of the robot using its current position and heading.
- Computes position error between this lookahead point and the requested waypoint.
- Converts the control output into linear and angular velocity commands.
- Limits commanded speeds and publishes `/cmd_vel`.

**Simulation limitation:** The controller looks up the `odom` to `vehicle_blue` transform, which supplies ground-truth pose in this simulation. It **does not subscribe to `/est_pose`**. This helped separate controller behavior from EKF localization error, but it also gives the controller position information that a physical robot would not have. The planner still uses the EKF-derived pose, so localization errors can influence the mapped obstacles and generated waypoints.

## Running the Simulation

### Environment

- ROS 2 **Jazzy**
- Docker image: `tiryoh/ros2-desktop-vnc:jazzy`
- ROS 2 workspace containing the `gazebo_controller` package

### Build and launch

From the workspace directory (the directory containing `src`):

```bash
colcon build
source install/setup.bash
ros2 launch gazebo_controller full_simulation.launch.py
```

The launch file starts Gazebo and RViz.

After editing code, rebuild with `colcon build` and source `install/setup.bash` again.

## Design Decisions and Limitations

- **EKF localization is separate from mapping:** The EKF combines odometry and IMU orientation, while lidar is used to update occupancy cells. This is EKF-based localization with occupancy grid mapping, **not joint EKF-SLAM**.
- **Mapping depends on pose accuracy:** Lidar observations are placed using the current estimated pose. Localization drift and differences between scan and pose timestamps can introduce inconsistencies or distortion in the map.
- **Ground-truth control feedback:** The PID controller relies on simulator pose feedback while the planner relies on EKF localization. Replacing ground truth with the EKF estimate would provide a more realistic end-to-end autonomy test.
- **Obstacle inflation and waypoint lookahead:** Both help tune navigation behavior, with tradeoffs between clearance, smoothness, and corner cutting.
- **Path fallback is limited:** The previous path is reused for a particular occupied-start condition, but not for every failed A* search.

## Potential Next Steps

- Route EKF-estimated pose into the controller and compare its performance against ground-truth feedback.
- Synchronize lidar scans with the corresponding pose estimate during mapping.
- Quantify localization drift against Gazebo ground truth.
- Improve A* fallback handling and validate grid boundary checks.
- Explore lidar-based scan matching or landmark corrections to feed environmental observations back into localization.

## Acknowledgments

Created as the final project for **EN.605.613 Introduction to Robots**, Johns Hopkins University, Fall 2025. The robot simulation, maze environments, and supporting map and goal publishers were provided as course infrastructure. I implemented the mapper, planner, and differential-drive PID controller nodes.
