#!/usr/bin/env python3
import numpy as np
from time import sleep

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros

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


class DiffDrivePID(Node):
    """
    PID controller for differential drive robot using trailer hitch approach.
    Based on EN613 midterm solution.
    """
    def __init__(self):
        super().__init__('diffdrive_pid')

        # PID gains
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('lookahead', 0.5)  # trailer hitch distance
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('angular_scale', 0.7)  # Scale down angular commands

        self.k_p = float(self.get_parameter('kp').get_parameter_value().double_value)
        self.k_d = float(self.get_parameter('kd').get_parameter_value().double_value)
        self.k_i = float(self.get_parameter('ki').get_parameter_value().double_value)
        self.length = float(self.get_parameter('lookahead').get_parameter_value().double_value)
        self.publish_rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').get_parameter_value().double_value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').get_parameter_value().double_value)
        self.dt = 1.0 / self.publish_rate

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # I/O
        qos_profile = QoSProfile(depth=10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_received, 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        # State
        self.desired_goal = np.array([0.0, -2.5])
        self.robot_state = np.array([0.0, 0.0, 0.0])
        self.Xp_last = None

        # TF frames
        self._to_frame = 'maze_world' # originally odom
        self._from_frame = 'vehicle_blue' # originally base_link

        # Timer loop
        self.timer = self.create_timer(self.dt, self.publish_robot_cmd)

        self.get_logger().info(f'DiffDrivePID started: kp={self.k_p}, kd={self.k_d}, ki={self.k_i}, lookahead={self.length}, rate={self.publish_rate} Hz')

    def goal_received(self, msg):
        """Callback for goal pose messages"""
        # Check if goal is in the correct frame (odom)
        goal_frame = msg.header.frame_id
        
        if goal_frame != 'odom' and goal_frame != '':
            self.get_logger().warn(f'Goal received in frame "{goal_frame}" but expecting "odom". Attempting to transform...')
            try:
                # Try to transform the goal to odom frame
                when = rclpy.time.Time()
                trans = self._tf_buffer.lookup_transform(
                    'odom', goal_frame,
                    when, timeout=Duration(seconds=1.0))
                
                # Get the goal position in the source frame
                goal_x = msg.pose.position.x
                goal_y = msg.pose.position.y
                
                # Get translation
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y
                
                # Get rotation (convert quaternion to yaw)
                quat = trans.transform.rotation
                _, _, yaw = euler_from_quaternion(quat)
                
                # Apply full transformation: rotate then translate
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                
                goal_x_odom = cos_yaw * goal_x - sin_yaw * goal_y + tx
                goal_y_odom = sin_yaw * goal_x + cos_yaw * goal_y + ty
                
                self.desired_goal = np.array([goal_x_odom, goal_y_odom])
                self.get_logger().info(f'Goal transformed to odom frame: [{self.desired_goal[0]:.3f}, {self.desired_goal[1]:.3f}]')
            except Exception as e:
                self.get_logger().error(f'Failed to transform goal: {str(e)}')
                return
        else:
            # Goal is already in odom frame (or no frame specified, assume odom)
            self.desired_goal = np.array([msg.pose.position.x, msg.pose.position.y])
            self.get_logger().info(f'Goal set to: [{self.desired_goal[0]:.3f}, {self.desired_goal[1]:.3f}] in odom frame')
        
        # Reset the integral term when a new goal is set
        self.Xp_last = None

    def compute_vel(self):
        """Compute velocity commands using PID control on trailer hitch point"""
        x_r = self.robot_state[0]
        y_r = self.robot_state[1]
        th_r = self.robot_state[2]

        # Compute a trailer hitch point in front of the agent
        X_p = np.array([x_r + self.length * np.cos(th_r),
                        y_r + self.length * np.sin(th_r)])

        # Compute the proportional error between the desired position and the final position
        # The [:,None] prefix reshapes the vector from (2,) to (2,1)
        p_err = (self.desired_goal - X_p)[:, None]
        

        if self.Xp_last is None:
            # If this is the first timestep we do not compute the derivative or integral terms
            d_err = 0
            i_err = 0
            p_err_last = None
        else:
            # Compute previous error
            p_err_last = (self.desired_goal - self.Xp_last)[:, None]
            
            # Derivative is the rate of change of error
            d_err = (p_err - p_err_last) / self.dt

            # We use the Trapezoidal rule to integrate the error
            i_err = self.dt * (p_err + p_err_last) / 2

        inv_rot = np.array([[np.cos(th_r), np.sin(th_r)],
                           [-np.sin(th_r), np.cos(th_r)]])
        
        V = inv_rot @ (self.k_p * p_err - self.k_d * d_err + self.k_i * i_err)
        linear_x = V[0, 0]
        angular_z = V[1, 0] / self.length

        # Clamp velocities
        linear_x = np.clip(linear_x, -self.max_linear_vel, self.max_linear_vel)
        angular_z = np.clip(angular_z, -self.max_angular_vel, self.max_angular_vel)

        self.Xp_last = X_p

        # ******************************************************************
        err_dist = np.linalg.norm(p_err)  # distance from hitch point to goal

        if err_dist < 0.001:
            linear_x = 0.0
            angular_z = 0.0
            self.Xp_last = None  

        # print(f"robot state is: {self.robot_state}")
        # print(f"hitch location is {X_p} and goal location is {self.desired_goal}")
        # print(f"ERROR IS: {p_err}. commanding robot to move forward with {linear_x} and angular {angular_z}")

        return np.array([linear_x, angular_z])

    def publish_robot_cmd(self):
        """Main control loop callback"""
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

        desired_vel = self.compute_vel()

        msg = Twist()
        msg.linear.x = float(desired_vel[0])
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(desired_vel[1])

        self.vel_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = DiffDrivePID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
