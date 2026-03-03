#!/usr/bin/env python3

"""
eBot Navigation for Task 2A + UR5 synchronization
- Pauses after P1 until UR5 drops fertilizer
- Listens to /ur5_status
- Publishes DOCK_STATION at P1 (mandatory)
- Controls Fruit Detection pause/resume per waypoint (ONE-TIME publish)
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion

# ===================== WAYPOINTS =====================

waypoints_exec = [
    [-1.53, -6.10, 1.57],
    [0.26, -1.95, 1.57],
    [-1.63, 1.27, 3.34],
    [-1.70, 1.24, 3.14],
    [-1.48, -0.67, -1.45],
    # [-1.53, -6.61, -1.57]
    [-1.53, -6.30, 2.94],
    [-2.75, -6.30, 1.37],
    [-2.75, 1.20, -0.2],
    # [-2.75, 1.24, -0.2],
    
    
    [-1.70, 1.24,-0.2],
    [0.26,1.27,0.0],
    
    
    
    [0.26, -2.00, -1.70],
    [0.26, -1.95, -1.57],
    [0.26, -6.10, -1.57],
    [-1.58, -6.10, 3.14],
    [-1.53, -6.61, -1.57]
    
    
     
]


class EbotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav_task2a')

        self.waypoints = waypoints_exec

        # Navigation parameters
        self.pos_tol = 0.2
        self.yaw_tol = math.radians(10)
        self.max_lin_vel = 0.5
        self.max_ang_vel = 1.0
        self.K_lin = 0.9
        self.K_ang = 1.5

        # State variables
        self.current_wp = 0
        self.state = 'navigate_xy'
        self.waypoint_stage = 'init'
        self.target_yaw = None

        self.x = self.y = self.yaw = None
        self.scan = None

        # Shape detection pause from detector
        self.paused = False

        # Dock publish flag
        self.dock_published = False
        self.second_dock_done = False


        # UR5 sync flags
        self.waiting_for_ur5 = False
        self.ur5_done = False

        # Fruit detection control (ONE-TIME per WP)
        self.last_wp_published = None

        # ================= SUBSCRIPTIONS =================
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/pause_navigation', self.pause_callback, 10)
        self.create_subscription(String, '/ur5_status', self.ur5_status_callback, 10)

        # ================= PUBLISHERS =================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.fruit_control_pub = self.create_publisher(String, '/fruit_detection_control', 10)

        self.create_timer(0.1, self.loop)

        self.get_logger().info("✓ eBot Navigator + UR5 sync + Fruit control started")

    # ================= CALLBACKS =================

    def ur5_status_callback(self, msg):
        if msg.data == "UR5_FERTILIZER_DONE":
            self.ur5_done = True
            self.waiting_for_ur5 = False
            self.get_logger().info("🟢 UR5 finished fertilizer task")

    def pause_callback(self, msg):
        if msg.data == "PAUSE":
            self.paused = True
            self.stop()
        elif msg.data == "RESUME":
            self.paused = False

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, msg):
        self.scan = msg

    # ================= HELPERS =================

    def clamp(self, v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def angle_diff(self, a, b):
        d = b - a
        while d > math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_lidar_range_at_angle(self, angle_deg):
        if not self.scan:
            return float('inf')
        angle_rad = math.radians(angle_deg)
        idx = int((angle_rad - self.scan.angle_min) / self.scan.angle_increment)
        idx = max(0, min(idx, len(self.scan.ranges) - 1))
        r = self.scan.ranges[idx]
        return r if self.scan.range_min < r < self.scan.range_max else float('inf')

    def turn_to_yaw(self, target):
        yaw_err = self.angle_diff(self.yaw, target)
        if abs(yaw_err) < self.yaw_tol:
            return True, Twist()
        cmd = Twist()
        cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
        return False, cmd

    def stop(self):
        self.cmd_pub.publish(Twist())

    # ================= FRUIT CONTROL =================

    def handle_fruit_detection_control(self):
        pause_wps = {0,1,3, 6,8}
        resume_wps = {2, 4, 5, 7}

        if self.current_wp == self.last_wp_published:
            return

        msg = String()

        if self.current_wp in pause_wps:
            msg.data = "Fruit_detection_pause"
            self.fruit_control_pub.publish(msg)
            # self.get_logger().info(f"🍎 Fruit detection PAUSED at WP {self.current_wp}")
            self.last_wp_published = self.current_wp

        elif self.current_wp in resume_wps:
            msg.data = "Fruit_detection_resume"
            self.fruit_control_pub.publish(msg)
            # self.get_logger().info(f"🍏 Fruit detection RESUMED at WP {self.current_wp}")
            self.last_wp_published = self.current_wp

    # ================= DOCK =================

    def publish_dock_station(self):
        if not self.dock_published:
            msg = String()
            msg.data = f"DOCK_STATION,{self.x:.2f},{self.y:.2f},0"
            self.detection_pub.publish(msg)
            self.dock_published = True
            self.get_logger().info(f"✓ Published dock: {msg.data}")

    # ================= MAIN LOOP =================

    def loop(self):
        if None in (self.x, self.y, self.yaw, self.scan):
            return

        # ONE-TIME fruit control per waypoint
        self.handle_fruit_detection_control()

        if self.paused:
            self.stop()
            return

        if self.waiting_for_ur5 and not self.ur5_done:
            self.stop()
            return

        if self.current_wp >= len(self.waypoints):
            self.stop()
            return

        tx, ty, tyaw = self.waypoints[self.current_wp]
        pos_err = self.distance((self.x, self.y), (tx, ty))
        yaw_err = self.angle_diff(self.yaw, tyaw)

        cmd = Twist()

        # ================= WAYPOINT 0 =================
        if self.current_wp == 1:
            if self.waypoint_stage == 'init':
                self.target_yaw = -0.1
                self.waypoint_stage = 'turn_right'

            elif self.waypoint_stage == 'turn_right':
                done, cmd = self.turn_to_yaw(self.target_yaw)
                if done:
                    self.waypoint_stage = 'move_obstacle'

            elif self.waypoint_stage == 'move_obstacle':
                if self.get_lidar_range_at_angle(0) < 0.30:
                    self.waypoint_stage = 'turn_left'
                else:
                    cmd.linear.x = self.max_lin_vel

            elif self.waypoint_stage == 'turn_left':
                done, cmd = self.turn_to_yaw(1.72)
                if done:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.get_logger().info("✓ P1 reached — waiting for UR5")
                    self.publish_dock_station()
                    self.waiting_for_ur5 = True
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                    self.stop()
                    return
                else:
                    cmd.linear.x = self.clamp(self.K_lin * (pos_err + 0.05), 0, self.max_lin_vel)

        # ================= WAYPOINT 3 =================
        elif self.current_wp == 4:
            if self.waypoint_stage == 'init':
                self.target_yaw = -1.45
                self.waypoint_stage = 'turn_left'

            elif self.waypoint_stage == 'turn_left':
                done, cmd = self.turn_to_yaw(self.target_yaw)
                if done:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)
                    
                    
        # ================= SECOND DOCK (FERTILIZER UNLOAD) =================
        elif self.current_wp == 11 and not self.second_dock_done:

            if pos_err < self.pos_tol:
                self.get_logger().info("✓ SECOND DOCK reached — waiting for UR5 to unload fertilizer")

                # IMPORTANT: allow re-publishing DOCK_STATION
                self.dock_published = False
                self.publish_dock_station()
                
                self.ur5_done = False   # <<< VERY IMPORTANT


                # Stop and wait
                self.waiting_for_ur5 = True
                self.stop()

                # Mark so this block runs only once
                self.second_dock_done = True
                return
            else:
                cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)

                    
        
        # ================= WAYPOINT 13 =================
       
        elif self.current_wp == 13:
            if self.waypoint_stage == 'init':
                self.target_yaw = 3.00
                self.waypoint_stage = 'turn_left'
                
            elif self.waypoint_stage == 'turn_left':
                done, cmd = self.turn_to_yaw(self.target_yaw)
                if done:
                    self.waypoint_stage = 'move_to_wp'
                
            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < 0.3:
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)
                    
                    
                    
        # ================= WAYPOINT 14 =================
        elif self.current_wp == 14:
            if self.waypoint_stage == 'init':
                self.target_yaw = -1.37
                self.waypoint_stage = 'turn_left'

            elif self.waypoint_stage == 'turn_left':
                done, cmd = self.turn_to_yaw(self.target_yaw)
                if done:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)

        # ================= DEFAULT =================
        else:
            range_front = self.get_lidar_range_at_angle(0)
            range_left = self.get_lidar_range_at_angle(90)
            range_right = self.get_lidar_range_at_angle(-90)
            range_side = max(range_left, range_right)

            if range_front == float('inf') or range_front > range_side or range_front > 0.45:
                if self.state == 'navigate_xy':
                    if pos_err < self.pos_tol:
                        self.state = 'navigate_yaw'
                    else:
                        cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)

                elif self.state == 'navigate_yaw':
                    if abs(yaw_err) < self.yaw_tol:
                        self.current_wp += 1
                        if self.current_wp < len(self.waypoints):
                            self.state = 'navigate_xy'
                    else:
                        cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)

            elif range_front < range_side and range_front < 0.45:
                if self.state != 'navigate_yaw':
                    self.state = 'navigate_yaw'

                if abs(yaw_err) >= self.yaw_tol:
                    cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
                else:
                    if pos_err < self.pos_tol:
                        self.current_wp += 1
                        self.state = 'navigate_xy'
                    else:
                        cmd.linear.x = self.clamp(self.K_lin * pos_err, 0, self.max_lin_vel)

        self.cmd_pub.publish(cmd)


# ================= MAIN =================

def main():
    rclpy.init()
    node = EbotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
