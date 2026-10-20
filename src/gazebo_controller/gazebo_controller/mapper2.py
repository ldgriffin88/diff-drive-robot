import numpy as np
from time import sleep
import os
import yaml
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs 
from sensor_msgs.msg import LaserScan, Imu
import tf2_ros
from ament_index_python.packages import get_package_share_directory

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    # Clamp sinp to avoid domain errors in arcsin
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler angle to quaternion
    
    :param roll: euler roll angle
    :param pitch: euler pitch angle
    :param yaw: euler yaw angle
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qx, qy, qz, qw

class Mapper2(Node):
    """
    Mapper node for differential drive robot. Processes lidar scans, constructs occupancy grid, and estimates robot pose in map coordinates
    """
    def __init__(self):
        """
        Creates mapper object
        """
        super().__init__('mapper2')
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # publisher for estimated pose for planner
        self.pose_publisher = self.create_publisher(PoseStamped, '/est_pose', 10) 

        # publisher for estimated pose for controller
        # self.pose_odom_publisher = self.create_publisher(PoseStamped, '/est_pose_odom', 10)

        self.est_robot_state = None # starting state as none
        self.first_pose_received = False # to get first loc in map
        
        self._to_frame = "odom" # transform frames
        self._from_frame = "base_link" # transform frames

        # odom topic subscription
        self.odom_sub = self.create_subscription(Odometry, '/odom',
                                                 self.odom_callback,
                                                 10)
        self.last_odom = None # starting odom as none
        self.last_odom_stamp = None # store last odom time

        # EKF info
        self.last_imu = None
        self.currnt_imu = None
        self.imu_sub = self.create_subscription(Imu, '/imu', 
                                                 self.imu_callback,
                                                 10)
        # covariance and noise
        self.P = np.diag([0.1**2, 0.1**2, 0.1**2])
        self.V = np.diag([0.01, 0.01, 0.3]) # noise for odom
        self.W_imu = ([[0.00121847]]) # 2 degree noise for imu

        # get parameters from map for occupancy grid
        self.declare_parameter('maze', 'basic_maze')
        maze_folder = self.get_parameter('maze').value
        pkg_share = get_package_share_directory('gazebo_controller')
        map_file = os.path.join(pkg_share, 'sdf', maze_folder, 'map.yaml')
        with open(map_file, 'r') as f:
            map_config = yaml.safe_load(f)
        self.resolution = map_config['resolution']
        self.width = map_config['width']
        self.height = map_config['height']
        self.origin_x = map_config['origin_x']
        self.origin_y = map_config['origin_y']
        self.grid_width = int(self.width  / self.resolution) # get grid size
        self.grid_height = int(self.height / self.resolution)

        # arrays to track hits and misses for occupancy
        self.hits = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        self.misses = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # for debugging
        self.get_logger().info(f'Mapper started with width and height {self.grid_width} and {self.grid_height}')

        # subscribe to lidar scans
        self.lidar_offset = 0.16
        self.scan_sub = self.create_subscription(LaserScan, 'lidar', 
                                                 self.scan_callback,
                                                 10)
        self.last_scan = None # store last lidar scan
        self.newest_scan = None # store new lidar scan

        # init occupancy grid and publisher
        self.occ = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        self.publisher_ = self.create_publisher(OccupancyGrid, '/occ_map', 10)

        self.occ_threshold = 0.6
        self.buffer = 0.4

    def odom_callback(self, msg):
        """
        Called when new odom message received. The linear and angular velocity here are used in the EKF prediction
    
        :param msg: Odometry message
        """
        t = msg.header.stamp

        # if no state in odom yet
        if self.est_robot_state is None:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            _, _, theta = euler_from_quaternion(msg.pose.pose.orientation)
            self.est_robot_state = np.array([x, y, theta])
            self.last_odom_stamp = t
            return

        # if last stamp not stored
        if self.last_odom_stamp is None:
            self.last_odom_stamp = t
            return
        
        # get time difference
        dt = (t.sec - self.last_odom_stamp.sec) + (t.nanosec - self.last_odom_stamp.nanosec) * 1e-9

        # set curr stamp as last stamp
        self.last_odom_stamp = t

        # if backwards in time return
        if dt <= 0.0:
            return

        # get V and omega
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        # call predict
        self.predict(v, omega, dt)

    def imu_callback(self, msg):
        """
        Called when new imu message received
        
        :param msg: Imu message
        """
        # just correct yaw
        _, _, yaw = euler_from_quaternion(msg.orientation)

        # update for ekf
        self.update_imu(yaw)

    def scan_callback(self, msg):
        """
        Called when new lidar scan received
        
        :param msg: LaseScan message
        """ 
        # if first scan, set as both old and new
        if self.newest_scan is None:
            # first scan received
            self.last_scan = msg
            self.newest_scan = msg
        else:
            # every scan after first
            self.last_scan = self.newest_scan
            self.newest_scan = msg

        # to update occupancy grid
        self.update_map()        

    def predict(self, v, omega, dt):
        """
        Predict step for EKF using the V and omega obtained from odometry message
        
        :param v: linear velocity of robot from odom
        :param omega: angular velocity of robot form odom
        :param dt: time difference between odom stamps
        """
        # if no state, return
        if self.est_robot_state is None:
            return

        # currest estimate
        x, y, theta = self.est_robot_state

        # motion model for prediction
        x_pred = x + v * np.cos(theta) * dt
        y_pred = y + v * np.sin(theta) * dt
        th_pred = theta + omega * dt

        # update state
        self.est_robot_state = np.array([[x_pred],
                    [y_pred],
                    [self.wrap_angle(th_pred)]])
        self.est_robot_state = self.est_robot_state.flatten()
        
        # jacobian
        A = np.array([[1.0, 0.0, -v * np.sin(theta) * dt],
                      [0.0, 1.0,  v * np.cos(theta) * dt],
                      [0.0, 0.0,  1.0]])

        # update covariance
        self.P = A @ self.P @ A.T + self.V
        
    def update_imu(self, yaw):
        """
        Update step for the EKF.
        
        :param yaw: orientation from Imu message
        """
        # if no estimated state, return
        if self.est_robot_state is None:
            return
        
        # get estimated state
        x, y, theta = self.est_robot_state

        # jacobian
        H = np.array([[0.0, 0.0, 1.0]])

        # differnce between prediction and imu measurement
        ang_diff = self.wrap_angle(yaw - theta)

        # innovation covariance
        S = H @ self.P @ H.T + self.W_imu
        R = self.P @ H.T @ np.linalg.inv(S)

        # update state
        state_col = self.est_robot_state.reshape(3,1)
        state_col = state_col + R * ang_diff
        self.est_robot_state = state_col.flatten()
        self.est_robot_state[2] = self.wrap_angle(self.est_robot_state[2])
    
        # update covariance
        self.P = self.P - R @ H @ self.P

    def map_to_grid(self, x, y):
        """
        Converts x and y map coordinates into occupancy grid coordinates
        
        :param x: x-coordinate in maze world
        :param y: y-coordinate in maze world
        """
        # use maze origin to get grid cells
        x_cell = int((x - self.origin_x) / self.resolution)
        y_cell = int((y - self.origin_y) / self.resolution)

        return x_cell, y_cell
    
    def ray_march(self, start_x, start_y, end_x, end_y):
        """
        Marches along the ray from stawrt to end coordinates, marking misses along the way and a hit at the end coordinate.
        
        :param start_x: maze x-coordinate start of ray
        :param start_y: maze y-coordinate start of ray
        :param end_x: maze x-coordinate end of ray
        :param end_y: maze y-coordinate end of ray
        """
        # take the start and end point and get cells
        col1, row1 = self.map_to_grid(start_x, start_y)
        col2, row2 = self.map_to_grid(end_x, end_y)

        # how far to go in each direction
        dr = abs(row2 - row1)
        dc = abs(col2 - col1)

        # directions
        sr = 1 if row1 < row2 else -1
        sc = 1 if col1 < col2 else -1

        # overall error
        error = dc - dr
        
        row = row1
        col = col1

        # size of map
        rows, cols = self.hits.shape

        # bresenham walk
        while True:

            # if not valid starting
            if not (0 <= row < rows and 0 <= col < cols):
                break

            # if at end mark hit, else mark miss
            if row == row2 and col == col2:
                self.hits[row, col] += 1
                break
            else:
                self.misses[row, col] += 1

            # error 
            e2 = 2 * error
            if e2 > -dr:
                error -= dr
                col += sc
            if e2 < dc:
                error += dc
                row += sr

    def build_occ_grid(self):
        """
        Builds an occupancy grid based on the estimated pose of the robot in the maze world.
        """
        # get size of grid
        rows = self.grid_height
        cols = self.grid_width

        # start with unknown everywhere
        self.occ[:] = -1

        # for each cell in the grid
        for r in range(rows):
            for c in range(cols):

                # get hits and misses for that cell
                hits = self.hits[r, c]
                misses = self.misses[r, c]
                total = hits + misses

                # if no hits or misses, leave as unknown
                if total == 0:
                    continue

                # basic probability
                p_occ = hits / total

                # threshold for setting as occupied.
                # can be tuned for different mazes
                if p_occ > self.occ_threshold:
                    self.occ[r, c] = 100  # occupied
                else:
                    self.occ[r, c] = 0  # free
        
        # add a buffer around hit cells since resolution is high
        # can be tuned for different mazes
        buffer = math.ceil(self.buffer / self.resolution)

        # get copy of occupancy grid so only buffering on actual hit cells
        base = self.occ.copy()

        # go through the grid
        for r in range(rows):
            for c in range(cols):
                # skip cells that are unknown or misses in base
                if base[r, c] != 100:
                    continue

                # get range for buffer cells in each direction
                r_min = max(0, r - buffer)
                r_max = min(rows - 1, r + buffer)
                c_min = max(0, c - buffer)
                c_max = min(cols - 1, c + buffer)

                # if the buffer cell is not occupied, mark it occupied
                for r_cell in range(r_min, r_max + 1):
                    for c_cell in range(c_min, c_max + 1):
                        if self.occ[r_cell, c_cell] != 100:
                            self.occ[r_cell, c_cell] = 100

    def update_map(self):
        """ 
        Take location and update occupancy grid based on lidar scans
        """
        # if don't have starting estimate, get position
        if self.est_robot_state is None:
            return
        
        # if no lidar scan yet
        if self.newest_scan is None:
            return
        
        # new robot state estimate has been set at this point
        x, y, yaw = self.est_robot_state.flatten()

        # publish new pose estimate as odom
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        # self.pose_odom_publisher.publish(pose_msg)

        # transform message to maze frame
        try:
            pose_maze = self._tf_buffer.transform(pose_msg, 'maze_world',    timeout=rclpy.duration.Duration(seconds=0.1))
    
            x_m = pose_maze.pose.position.x
            y_m = pose_maze.pose.position.y
            _, _, yaw_m = euler_from_quaternion(pose_maze.pose.orientation)

            self.pose_publisher.publish(pose_maze)
            # print(f"robot in maze coordinates is: {x_m, y_m, yaw_m}")

        except Exception as e:
            self.get_logger().warn(f"TF transform odom->maze failed: {e}")
            return

        
        # get lidar coordinates
        cos_t = math.cos(yaw_m)
        sin_t = math.sin(yaw_m)
        lidar_x = x_m + self.lidar_offset * cos_t
        lidar_y = y_m + self.lidar_offset * sin_t
        
        # go through scan and march along the rays
        ranges = self.newest_scan.ranges
        angle_min = self.newest_scan.angle_min
        angle_inc = self.newest_scan.angle_increment

        for i, r in enumerate(ranges):
            
            if r < 0.05 or r > 10.0:
                continue

            ray_angle = angle_min + i * angle_inc
            map_angle = yaw_m + ray_angle

            # end coordinates of ray
            map_x_end = lidar_x + r * math.cos(map_angle)
            map_y_end = lidar_y + r * math.sin(map_angle)

            self.ray_march(lidar_x, lidar_y, map_x_end, map_y_end)

        # build the occupancy grid
        self.build_occ_grid()

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'maze_world'
        
        # Map metadata
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Flatten the grid (row-major order) and convert to list
        msg.data = self.occ.flatten().tolist()
        
        self.publisher_.publish(msg)

    def wrap_angle(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
def main(args=None):
    rclpy.init(args=args)
    node = Mapper2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()