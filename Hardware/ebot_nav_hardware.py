import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion


# Execution waypoints with intermediates
waypoints_exec = [



    [ 2.450, -1.656,  0.0],   # P1
    [ 4.716, 0.04 ,  1.77],   # Intermediate
    [4.716, 0.04 ,  1.57],   # Intermediate
    [ 2.639,  0.0, -3.14],   # P2
    [ 0.005,  0.0, 1.37],   #P3
    [0.0, 1.682, -0.15],  #P4
    [4.716, 1.682, -1.77],  #P5
    [4.716, -0.04, -1.57],  #P6
    [ 0.100,  0.0, -3.14]  #P7

]

# ---------- Detection trigger points ----------
DETECTION_POINTS = [
    {
        "label": "BAD_HEALTH",
        "x": 3.870,
        "y": -1.551,
        "tolerance": 0.20,
        "triggered": False,
        "p_id":4
    },
    {
        "label": "BAD_HEALTH",
        "x": 2.290,
        "y": -0.040,
        "tolerance": 0.15,
        "triggered": False,
        "p_id":6
    },
    # {
    #     "label": "FERTILIZER_REQUIRED",
    #     "x": 1.574,
    #     "y": -0.020,
    #     "tolerance": 0.15,
    #     "triggered": False,
    #     "p_id":1
    # },
    # {
    #     "label": "FERTILIZER_REQUIRED",
    #     "x": 4.600,
    #     "y": 1.666,
    #     "tolerance": 0.23,
    #     "triggered": False,
    #     "p_id":8
    # },
    # 👉 Add more points like this
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

        # State management
        self.current_wp = 0
        self.state = 'navigate_xy'
        self.waypoint_stage = 'init'
        self.target_yaw = None

        # Robot pose
        self.x = self.y = self.yaw = None
        self.scan = None

        # Task 2A: Shape detection integration
        self.paused = False
        self.dock_published = False

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/pause_navigation', self.pause_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)

        # Timer
        self.create_timer(0.1, self.loop)

        self.get_logger().info('✓ Task Navigator Started')
        self.get_logger().info('  Integrating with shape detection system')

    def check_and_publish_detection(self):
        """
        Check if robot is near any predefined detection point
        and publish detection_status once.
        """
        for point in DETECTION_POINTS:
            if point["triggered"]:
                continue

            dist = math.hypot(self.x - point["x"], self.y - point["y"])

            if dist <= point["tolerance"]:
                msg = String()
                msg.data = f"{point['label']},{self.x:.2f},{self.y:.2f},{point['p_id']}"
                self.detection_pub.publish(msg)

                self.get_logger().info(f"🟢 Detection published: {msg.data}")

                point["triggered"] = True

                # Mandatory 2 second wait
                self.stop()
                time.sleep(2.0)


    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, msg):
        self.scan = msg

    def pause_callback(self, msg):
        """Handle pause/resume from shape detector"""
        if msg.data == "PAUSE":
            self.paused = True
            self.stop()
            self.get_logger().info(" Navigation PAUSED - Shape detected")
        elif msg.data == "RESUME":
            self.paused = False
            self.get_logger().info(" Navigation RESUMED")

    def clamp(self, v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def angle_diff(self, a, b):
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_lidar_range_at_angle(self, angle_deg):
        if not self.scan:
            return float('inf')
        angle_min = self.scan.angle_min
        angle_inc = self.scan.angle_increment
        n = len(self.scan.ranges)
        angle_rad = math.radians(angle_deg)
        idx = int(round((angle_rad - angle_min) / angle_inc)) % n
        r = self.scan.ranges[idx]
        return r if self.scan.range_min < r < self.scan.range_max else float('inf')

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def turn_to_yaw(self, target_yaw):
        """Turn robot to target yaw"""
        yaw_err = self.angle_diff(self.yaw, target_yaw)

        if abs(yaw_err) < self.yaw_tol:
            return True, Twist()
        else:
            cmd = Twist()
            cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
            return False, cmd

    def publish_dock_station(self):
        """Publish DOCK_STATION at P1 (MANDATORY)"""
        if not self.dock_published and self.current_wp >= 1:
            # Use original P1 coordinates
            tx, ty = self.x, self.y
            msg = String()
            msg.data = f"DOCK_STATION,{tx:.2f},{ty:.2f}"
            self.detection_pub.publish(msg)
            self.dock_published = True
            self.get_logger().info(f"✓ Published: {msg.data}")
            self.stop()
            time.sleep(2.0)

    def loop(self):
        if None in (self.x, self.y, self.yaw, self.scan):
            return

        # Paused by shape detector - don't move
        if self.paused:
            self.stop()
            time.sleep(2.0)
            return

        # Check detection locations (independent of waypoints)
        self.check_and_publish_detection()


        if self.current_wp >= len(self.waypoints):
            if self.state != 'done':
                self.stop()
                self.state = 'done'
                self.get_logger().info("✓ Mission Complete!")
            return

        tx, ty, tyaw = self.waypoints[self.current_wp]
        pos_err = self.distance((self.x, self.y), (tx, ty))
        yaw_err = self.angle_diff(self.yaw, tyaw)

        range_front = self.get_lidar_range_at_angle(0)
        range_left_80 = self.get_lidar_range_at_angle(90)
        range_right_80 = self.get_lidar_range_at_angle(-90)
        range_side = max(range_left_80, range_right_80)

        cmd = Twist()

        # ============ WAYPOINT 0 (P1 - Dock Station) ============
        if self.current_wp == 0:

            if self.waypoint_stage == 'init':
                self.target_yaw = -1.57  # -1.27
                self.waypoint_stage = 'turn_right'

            elif self.waypoint_stage == 'turn_right':
                turn_complete, cmd = self.turn_to_yaw(self.target_yaw)
                if turn_complete:
                    self.waypoint_stage = 'move_obstacle'

            elif self.waypoint_stage == 'move_obstacle':
                if range_front < 0.45:
                    self.waypoint_stage = 'turn_left'
                else:
                    cmd.linear.x = self.max_lin_vel

            elif self.waypoint_stage == 'turn_left':
                self.target_yaw = 0.15
                turn_complete, cmd = self.turn_to_yaw(self.target_yaw)
                if turn_complete:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.get_logger().info(f"✓ P1 (Dock Station) reached!")
                    # MANDATORY: Publish DOCK_STATION
                    self.current_wp += 1
                    self.publish_dock_station()
                    self.waypoint_stage = 'init'
                    cmd.linear.x = 0.0
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)

        # ============ WAYPOINT 3 (P2) ============
        elif self.current_wp == 3:

            if self.waypoint_stage == 'init':
                self.target_yaw = -3.00
                self.waypoint_stage = 'turn_left'

            elif self.waypoint_stage == 'turn_left':
                turn_complete, cmd = self.turn_to_yaw(self.target_yaw)
                if turn_complete:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)




        # ============ WAYPOINT 8 (P7) ============
        elif self.current_wp == 8:

            if self.waypoint_stage == 'init':
                self.target_yaw = 2.99
                self.waypoint_stage = 'turn_right'

            elif self.waypoint_stage == 'turn_right':
                turn_complete, cmd = self.turn_to_yaw(self.target_yaw)
                if turn_complete:
                    self.waypoint_stage = 'move_to_wp'

            elif self.waypoint_stage == 'move_to_wp':
                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.waypoint_stage = 'init'
                else:
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)




        # ============ OTHER WAYPOINTS (Normal Navigation) ============
        elif range_front == float('inf') or range_front > range_side or range_front > 0.45:

            if self.state == 'navigate_xy':
                if pos_err < self.pos_tol:
                    self.state = 'navigate_yaw'
                    cmd.linear.x = 0.0
                else:
                    heading = math.atan2(ty - self.y, tx - self.x)
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)

            elif self.state == 'navigate_yaw':
                if abs(yaw_err) < self.yaw_tol:
                    self.current_wp += 1
                    if self.current_wp >= len(self.waypoints):
                        self.state = 'done'
                    else:
                        self.state = 'navigate_xy'
                    cmd.linear.x = 0.0
                else:
                    cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
                    cmd.linear.x = 0.0

        elif range_front < range_side and range_front < 0.45:

            if self.state != 'navigate_yaw':
                self.state = 'navigate_yaw'

            if abs(yaw_err) >= self.yaw_tol:
                cmd.angular.z = self.clamp(self.K_ang * yaw_err, -self.max_ang_vel, self.max_ang_vel)
                cmd.linear.x = 0.0
            else:
                if self.state != 'navigate_xy':
                    self.state = 'navigate_xy'

                if pos_err < self.pos_tol:
                    self.current_wp += 1
                    self.state = 'navigate_xy' if self.current_wp < len(self.waypoints) else 'done'
                    cmd.linear.x = 0.0
                else:
                    heading = math.atan2(ty - self.y, tx - self.x)
                    cmd.linear.x = self.clamp(self.K_lin * pos_err, 0.0, self.max_lin_vel)

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())


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
