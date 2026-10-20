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

def manhattan(loc: Tuple[int, int], goal: Tuple[int, int]):
        """
        Helper to return manhattan distance between two points
        """
        return (abs(loc[0] - goal[0]) + abs(loc[1] - goal[1]))

class Planner(Node):
    def __init__(self):
        """
        Creates planner object
        """
        super().__init__('planner')

        self.robot_state = None # starting state

        self.dt = 1.0 / 2 # frequency
        self.timer = self.create_timer(self.dt, self.update_planner) # timer loop
        # for debugging
        self.get_logger().info(f'Planner started***************')
        self.grid_sub = self.create_subscription(OccupancyGrid, '/occ_map', 
                                                 self.grid_callback,
                                                 10)
        self.current_grid = None
        self.last_grid = None
        self.cols = None
        self.rows = None
        self.resolution = None
        self.origin_x = None
        self.origin_y = None

        # to publish goal pose
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # to store goals and subscribe to goal points
        self.goals = {}

        # get goals from map
        self.current_goal_id = None
        self.goal_sub = self.create_subscription(MarkerArray, '/goal_points', self.goal_callback, 10)

        # to receive pose estimate
        self.pose_sub = self.create_subscription(PoseStamped, '/est_pose', self.pose_callback, 10)

        # to store path
        self.path = []

        # to keep consistent occupancy grids for search
        self.astar_running = False

        self.advance = 5

    def grid_callback(self, msg):
        """
        Called when an occupancy map has been published
        
        :param msg: OccupancyGrid message
        """ 
        # get and set parameters
        self.cols = msg.info.width
        self.rows = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        occ = np.array(msg.data, dtype=np.int8).reshape(self.rows, self.cols)
        self.current_grid = occ
    
    def goal_callback(self, msg):
        """
        Called when the initial goals are published by the map publisher
        
        :param msg: MarkerArray message containing goal locations
        """
        # only get goals once
        if self.goals:
            return
        
        # in maze world coordinates
        for marker in msg.markers:
            goal_x = marker.pose.position.x
            goal_y = marker.pose.position.y

            # add to goals dict
            self.goals[marker.id] = (goal_x, goal_y)
        
        self.get_logger().info(f"Received {len(self.goals)} goals.")
        
        # destroy after goals received
        self.destroy_subscription(self.goal_sub)

    def pose_callback(self, msg):
        """
        Called when an estimated pose is received
        
        :param msg: PoseStamped message from mapper
        """
        # get maze pose estimate from mapper node
        x = msg.pose.position.x
        y = msg.pose.position.y
        _, _, theta = euler_from_quaternion(msg.pose.orientation)

        # set as state
        self.robot_state = np.array([x, y, theta])

    def grid_to_world(self, row, col):
        """
        Converts a grid cell coordinate to maze frame
        
        :param row: row location in grid. Origin at bottom left corner
        :param col: column location in grid
        """
        # take grid and convert to maze world
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution

        return float(x), float(y)

    def map_to_grid(self, x, y):
        """
        Converts a coordinate from maze frame to occupancy grid
        
        :param x: maze world x-coordinate
        :param y: maze world y-coordinate
        """
        # take maze coordinate and convert to grid cell
        x_cell = int((x - self.origin_x) / self.resolution)
        y_cell = int((y - self.origin_y) / self.resolution)

        return x_cell, y_cell

    def select_goal(self):
        """
        This method chooses a goal from the self.goals dictionary as a target by calculating the distance from the robot's estimated pose. The closest goal becomes the new target. When a goal is reached, it is deleted from the self.goals dictionary.
        """
        # if no goals, return None
        if not self.goals:
            return None
        
        # get estimated state
        x, y, _ = self.robot_state

        # start at infinite distance
        min_dist = math.inf
        closest = None

        # to store goals reached
        to_delete = []

        # for each goal, find closest
        for goal, (gx, gy) in self.goals.items():
            robot_pos = (x, y)
            goal_pos = (gx, gy)
            dist = math.dist(robot_pos, goal_pos)

            # if made it to goal, delete it from dict
            if dist < 0.1:
                to_delete.append(goal)
                continue

            # if goal is closer than current closest
            if dist < min_dist:
                min_dist = dist
                closest = goal

        # delete goal if robot made it there
        for g in to_delete:
            self.goals.pop(g, None)

        # set goal id
        self.curret_goal_id = closest

        return closest

    def a_star(self, start:Tuple[int, int], goal: Tuple[int, int]):
        """
        Runs A* search algorithm to find a path from the starting coordinates to the goal coordinates. Returns the path as a list. If no path is found, an empty list is returned. If the start location is occupied, return the previous path instead of trying to find a new one. 
        
        :param start: starting location of search
        :type start: Tuple[int, int]
        :param goal: goal location of search
        :type goal: Tuple[int, int]
        """
        self.astar_running = True
        self.last_grid = self.current_grid.copy()

        # if robot is in occupied cell, return last path
        x_start, y_start = start
        if self.last_grid[x_start-1, y_start-1] == 100:
            return self.path[1:]
        
        # goal is (row, col) in grid coordinates
        buffer = 6 # resolution
        rows, cols = self.last_grid.shape
        gr, gc = goal   # row, col

        # mark the goal cell free
        self.last_grid[gr, gc] = 0

        # clamp buffer region
        r_min = max(0, gr - buffer)
        r_max = min(rows - 1, gr + buffer)
        c_min = max(0, gc - buffer)
        c_max = min(cols - 1, gc + buffer)

        # clear occupied cells in that box
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if self.last_grid[r, c] == 100:   # occupied
                    self.last_grid[r, c] = 0      # mark free

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
                h = manhattan(expansion, goal)
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
        """
        ***This method is not used.***

        It's intention is to reduce the path length if the map has high resolution to provide separation between waypoints.
        """
        # get every stepth grid cell
        step = 3 
        reduced_path = self.path[::step]

        # if goal is not last cell, append it
        if reduced_path[-1] != self.path[-1]:
            reduced_path.append(self.path[-1])

        # set as path
        self.path = reduced_path

    def pick_goal(self):
        """
        Chooses an intermediate goal coordinate along the path to the overall goal by advancing the robot's estimated current location. Returns the minimum between 5 steps ahead and the end of the path.
        """
        # if no path, return no goal
        if len(self.path) == 0:
            return None
        
        # can be tuned for different performance on maps
        closest_path_id = min(self.advance, len(self.path) - 1)

        return closest_path_id

    def publish_goal(self, x:float, y:float):
        """
        Publishes a goal as a PoseStamped message in the maze_world frame
        
        :param x: x-coordinate in the maze_world frame
        :type x: float
        :param y: y-coordinate in the maze_world
        :type y: float
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "maze_world"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.goal_publisher.publish(msg)

    def update_planner(self):
        """
        Main method to use an occupancy grid, estimated pose, and goal location to run A* search and publish a goal location i nthe maze_world frame.
        """
        # if no occ grid to go off of, retrun
        if self.current_grid is None:
            return
        if not self.goals:
            return

        # skip if robot state is not known yet
        if self.robot_state is None:
            return
        
        # get robot state
        x, y, yaw = self.robot_state

        # choose a goal from the goal dictionary
        goal_id = self.select_goal()
        if goal_id is None:
            return
        
        goal_x, goal_y = self.goals[goal_id]

        # get robot location and goal location i ngrid
        robot_col, robot_row = self.map_to_grid(x, y)
        goal_col, goal_row = self.map_to_grid(goal_x, goal_y)

        # flip to row, col
        robot = (robot_row, robot_col)
        goal = (goal_row, goal_col)
        
        # return if in middle of a star
        if self.astar_running:
            return
        
        # do a star search
        self.path = self.a_star(robot, goal)
        self.astar_running = False

        # if no path or length of path is zero, exit
        if self.path is None or len(self.path) == 0:
            return
        
        # pick goal cell based on robot's position
        index = self.pick_goal()

        # get advanced point in path and conver to maze world
        gx, gy = self.path[index]
        x_world, y_world = self.grid_to_world(gx, gy)

        # publish
        self.publish_goal(x_world, y_world)

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