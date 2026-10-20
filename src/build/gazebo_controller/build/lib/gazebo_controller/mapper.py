import numpy as np
from time import sleep
import os
import yaml
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
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

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper')
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.robot_state = None # starting state as none
        self._to_frame = "maze_world" # transform frames
        self._from_frame = "vehicle_blue" # transform frames
        self.dt = 1.0 / 10.0 # frequency
        self.timer = self.create_timer(self.dt, self.update_map) # timer loop

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

        # arrays to track hits and misses
        self.hits = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        self.misses = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # for debugging
        self.get_logger().info(f'Mapper started with width and height {self.grid_width} and {self.grid_height}')

        qos_profile = QoSProfile(depth=10)
        self.scan_sub = self.create_subscription(LaserScan, 'lidar', 
                                                 self.scan_callback,
                                                 10)
        self.last_scan = None

        self.occ = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        self.publisher_ = self.create_publisher(OccupancyGrid, '/occ_map', 10)

    def get_robot_state(self):
        # Listen for transform of robot's position  
        try:
            when = rclpy.time.Time()
            trans = self._tf_buffer.lookup_transform(
                self._to_frame, self._from_frame,
                when, timeout=Duration(seconds=5.0))
        except tf2_ros.LookupException:
            self.get_logger().warn('Transform isn\'t available, waiting...')
            return
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {str(e)}')
            return
            
        pose = trans.transform.translation
        orientation = trans.transform.rotation
        roll, pitch, yaw = euler_from_quaternion(orientation)
        self.robot_state = np.array([pose.x, pose.y, yaw]) 

    def scan_callback(self, msg):
        """ Obtain lidar scan information""" 
        self.last_scan = msg
        # print(msg)
    
    def map_to_grid(self, x, y):
        """ Convert map coordinates into occ grid cells"""
        x_cell = int((x - self.origin_x) / self.resolution)
        y_cell = int((y - self.origin_y) / self.resolution)

        return x_cell, y_cell
    
    def ray_march(self, start_x, start_y, end_x, end_y):
        """ use Bresenham to mark hits and misses """
        # take the start and end point and get cells
        col1, row1 = self.map_to_grid(start_x, start_y)
        col2, row2 = self.map_to_grid(end_x, end_y)

        dr = abs(row2 - row1)
        dc = abs(col2 - col1)

        sr = 1 if row1 < row2 else -1
        sc = 1 if col1 < col2 else -1

        error = dc - dr
        
        row = row1
        col = col1

        rows, cols = self.hits.shape

        while True:
            if not (0 <= row < rows and 0 <= col < cols):
                break

            if row == row2 and col == col2:
                self.hits[row, col] += 1
                break
            else:
                self.misses[row, col] += 1

            e2 = 2 * error
            if e2 > -dr:
                error -= dr
                col += sc
            if e2 < dc:
                error += dc
                row += sr

    def build_occ_grid(self):
        # Initialize grid as free space (0 = free, 100 = occupied, -1 = unknown)
        rows = self.grid_height
        cols = self.grid_width

        # start with unknown everywhere
        self.occ[:] = -1

        for r in range(rows):
            for c in range(cols):
                hits = self.hits[r, c]
                misses = self.misses[r, c]
                total = hits + misses

                # if no hits or misses, leave as -1
                if total == 0:
                    continue

                p_occ = hits / total

                # if more hits than misses, mark occupied
                if p_occ > 0.5:
                    self.occ[r, c] = 100  # occupied
                else:
                    self.occ[r, c] = 0    # free
        
        # buffer for hit cells
        buffer = math.ceil(0.2 / self.resolution)
        base = self.occ.copy()

        for r in range(rows):
            for c in range(cols):
                # skip cells that are unknown or misses in base
                if base[r, c] != 100:
                    continue

                r_min = max(0, r - buffer)
                r_max = min(rows - 1, r+ buffer)
                c_min = max(0, c - buffer)
                c_max = min(cols - 1, c + buffer)

                for r_cell in range(r_min, r_max + 1):
                    for c_cell in range(c_min, c_max + 1):
                        if self.occ[r_cell, c_cell] != 100:
                            self.occ[r_cell, c_cell] = 100

    def update_map(self):
        """ 
        Take location and update occupancy grid based on lidar scans
        """
        
        self.get_robot_state()

        if self.last_scan is None:
            return
        if self.robot_state is None:
            return
        
        x, y, yaw = self.robot_state


        ranges = self.last_scan.ranges
        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment

        for i, r in enumerate(ranges):
            
            if r < 0.3 or r > 15.0:
                continue

            ray_angle = angle_min + i * angle_inc
            map_angle = yaw + ray_angle

            # end coordinates of ray
            map_x_end = x + r * math.cos(map_angle)
            map_y_end = y + r * math.sin(map_angle)

            self.ray_march(x, y, map_x_end, map_y_end)


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

def main(args=None):
    rclpy.init(args=args)
    node = Mapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()