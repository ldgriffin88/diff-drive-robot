import numpy as np
from time import sleep
import os
import math
from typing import List, Tuple
from heapq import heappush, heappop
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import MarkerArray

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


class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.robot_state = None # starting state
        self._to_frame = "maze_world" # transform frames
        self._from_frame = "vehicle_blue" # transform frames
        self.dt = 1.0 / 10.0 # frequency
        self.timer = self.create_timer(self.dt, self.update_planner) # timer loop
        # for debugging
        self.get_logger().info(f'Planner started***************')
        self.grid_sub = self.create_subscription(OccupancyGrid, '/occ_map', 
                                                 self.grid_callback,
                                                 10)
        self.last_grid = None
        self.cols = None
        self.rows = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        self.goal_publisher = self.create_publisher(PoseStamped, 'goal', 10)

        # to store goals and subscribe to goal points
        self.goals = {}
        # self.goals_visited = set()
        self.current_goal_id = None
        self.goal_sub = self.create_subscription(MarkerArray, 'goal_points', self.goal_callback, 10)

        self.path = []
        self.path_step = 0

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
 
    def grid_callback(self, msg):
        """ Obtain occupancy grid""" 
        self.cols = msg.info.width
        self.rows = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        occ = np.array(msg.data, dtype=np.int8).reshape(self.rows, self.cols)
        self.last_grid = occ
    
    def goal_callback(self, msg):
        # in maze world coordinates
        for marker in msg.markers:
            goal_x = marker.pose.position.x
            goal_y = marker.pose.position.y

            # add to goals dict
            self.goals[marker.id] = (goal_x, goal_y)
        
        self.get_logger().info(f"Received {len(self.goals)} goals.")

    def grid_to_world(self, row, col):
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution

        return float(x), float(y)

    def map_to_grid(self, x, y):
        """ Convert map coordinates into occ grid cells"""
        x_cell = int((x - self.origin_x) / self.resolution)
        y_cell = int((y - self.origin_y) / self.resolution)

        return x_cell, y_cell

    def select_goal(self):

        if not self.goals:
            return None
        
        x, y, _ = self.robot_state

        min_dist = math.inf
        closest = None

        for goal, (gx, gy) in self.goals.items():
            robot_pos = (x, y)
            goal_pos = (gx, gy)
            dist = math.dist(robot_pos, goal_pos)

            if dist < min_dist:
                min_dist = dist
                closest = goal

        self.curret_goal_id = closest

        return closest

    def manhattan(loc:Tuple[int, int], goal: Tuple[int, int]):
            """
            Helper to return manhattan distance between two points.
            """
            return (abs(loc[0] - goal[0]) + abs(loc[1] - goal[1]))

    def a_star(self, start:Tuple[int, int], goal: Tuple[int, int]):

        # start of astar
        open = []
        closed = set()
        nodes = {}

        heappush(open, (0, start))
        nodes[start] = {"Parent": None, "Cost": 0}

        # Still nodes to expand
        while open:
            # Choose the best from open to expand
            curr_f, curr = heappop(open)

            # Skip if duplicate from heap
            if curr in closed:
                continue

            # Add to closed
            closed.add(curr)

            # If goal, get path
            if curr == goal:
                path = []
                path.append(curr)
                while nodes[curr]["Parent"] is not None:
                    path.append(nodes[curr]["Parent"])
                    curr = nodes[curr]["Parent"]
                
                return path[::-1]

        # Expand nodes 
        expansions = []

        r, c = curr

        # Down (row + 1)
        if r + 1 < self.rows and self.last_grid[r + 1, c] != 100:
            expansions.append((r + 1, c))

        # Up (row - 1)
        if r - 1 >= 0 and self.last_grid[r - 1, c] != 100:
            expansions.append((r - 1, c))

        # Left (col - 1)
        if c - 1 >= 0 and self.last_grid[r, c - 1] != 100:
            expansions.append((r, c - 1))

        # Right (col + 1)
        if c + 1 < self.cols and self.last_grid[r, c + 1] != 100:
            expansions.append((r, c + 1))

        for expansion in expansions:

            # skip if already in closed list
            if expansion in closed:
                continue

            # cost to current state
            curr_cost = nodes[curr]["Cost"]

            # new values
            g = curr_cost + 1
            h = self.manhattan(expansion, goal)
            f = g + h

            # if already in nodes/open
            if expansion in nodes:
                prev_cost = nodes[expansion]["Cost"]

                # if new cost to reach is cheapter, update
                if g < prev_cost:
                    nodes[expansion]["Parent"] = curr
                    nodes[expansion]["Cost"] = g
                    heappush(open, (f, expansion))
            
            else:
                # add to nodes and open list
                nodes[expansion] = {"Parent": curr, "Cost": g}
                heappush(open, (f, expansion))

        return []

    def path_reduce(self):
        step = 3
        reduced_path = self.path[::step]

        if reduced_path[-1] != self.path[-1]:
            reduced_path.append(self.path[-1])

        self.path = reduced_path

    def update_planner(self):
        # if no occ grid to go off of, retrun
        if self.last_grid is None:
            return
        if not self.goals:
            return
        
        # try to get robot state
        self.get_robot_state()
        # skip if robot state is not known yet
        if self.robot_state is None:
            return
        x, y, yaw = self.robot_state

        goal_id = self.select_goal()
        if goal_id is None:
            return
        
        print(f'goal id is {goal_id} and goals are: {self.goals}')
        
        goal_x, goal_y = self.goals[goal_id]

        print(f"therefore, goal coordinates are: {goal_x}, {goal_y}")

        robot_col, robot_row = self.map_to_grid(x, y)
        goal_col, goal_row = self.map_to_grid(goal_x, goal_y)

        # flip to row, col
        robot = (robot_row, robot_col)
        goal = (goal_row, goal_col)

        self.path = self.a_star(robot, goal)

        if self.path is None or len(self.path) == 0:
            return
        
        self.path_reduce()

        print(f"path to goal is: {self.path}")
        
        # now self.path has been reduced


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()