import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion

class ShapeDetector(Node):
    def __init__(self):
        super().__init__('shape_detector_task2a')

        # --- State Variables ---
        self.x = None
        self.y = None
        self.yaw = None
        self.detection_paused = False
        self.processing = False
        self.detected_objects = []

        # --- Tuned Parameters (From Motion-Robust Code) ---
        # 1. Clustering
        self.cluster_thresh = 0.15
        
        # 2. Size Filters
        self.min_span = 0.05
        self.max_span = 0.75

        # 3. Geometry Simplification (RDP)
        self.rdp_epsilon = 0.02  # 2cm tolerance

        # 4. Point Counts
        self.min_points = 4
        self.max_points = 70

        # --- Subscribers ---
        self.create_subscription(Odometry, '/odom', self.odom_callback, 30)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/fruit_detection_control', self.control_callback, 10)

        # --- Publishers ---
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.pause_pub = self.create_publisher(String, '/pause', 10)

        self.get_logger().info('✓ Shape Detector Task 3B (Motion-Robust Logic) Initialized')

    def control_callback(self, msg):
        """Pause or resume detection based on /fruit_detection_control"""
        if msg.data == "Fruit_detection_pause":
            self.detection_paused = True
            self.get_logger().info("⏸️ Fruit detection paused")
        elif msg.data == "Fruit_detection_resume":
            self.detection_paused = False
            self.get_logger().info("▶️ Fruit detection resumed")

    def odom_callback(self, msg):
        """Track robot global position"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, msg):
        """Main Detection Loop (Replaced with Motion-Robust Logic)"""
        # 1. Safety Checks
        if self.detection_paused or self.processing or self.x is None:
            return

        # 2. Polar -> Cartesian Conversion
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Filter valid ranges (0.05m to 3.0m)
        valid_mask = (ranges > 0.05) & (ranges < 3.0) & (np.isfinite(ranges))
        if not np.any(valid_mask): 
            return

        xs = ranges[valid_mask] * np.cos(angles[valid_mask])
        ys = ranges[valid_mask] * np.sin(angles[valid_mask])
        points = np.column_stack((xs, ys))

        # 3. Clustering
        clusters = self.get_clusters(points)

        # 4. Shape Analysis
        for cluster in clusters:
            # --- FILTER A: Basic Point Counts ---
            if len(cluster) < self.min_points or len(cluster) > self.max_points:
                continue
            
            # --- FILTER B: Physical Size ---
            min_x, min_y = np.min(cluster, axis=0)
            max_x, max_y = np.max(cluster, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            span = math.hypot(width, height)
            
            # Reject huge walls or tiny speckles
            if span > self.max_span or span < self.min_span: 
                continue 

            # --- FILTER C: Linearity Check (The "Turning Fix") ---
            # Rejects walls that look like shapes during turns
            if self.is_linear(cluster):
                continue

            # --- FILTER D: Aspect Ratio ---
            # Aspect ratio check to kill long skinny wall segments
            short_side = max(min(width, height), 0.01)
            aspect_ratio = span / short_side
            if aspect_ratio > 3.5:
                continue

            # --- GEOMETRY: Count Corners (RDP Algorithm) ---
            vertices = self.ramer_douglas_peucker(cluster, self.rdp_epsilon)
            
            # Check closure (is it a loop?)
            gap = np.linalg.norm(vertices[0] - vertices[-1])
            is_closed = gap < 0.12  # 12cm gap allowed

            num_sides = 0
            if is_closed:
                num_sides = max(0, len(vertices) - 1)
            else:
                continue 

            # --- CLASSIFICATION ---
            shape_type = None
            status_msg = ""
            
            # Logic: Identify shape based on sides
            # Note: You can adjust these mappings based on specific Task 3B rules
            
            if num_sides == 3:
                # 4 Sides = Square
                shape_type = "SQUARE"
                status_msg = "BAD_HEALTH" 
            elif num_sides >= 5 and num_sides <= 6:
                shape_type = "PENTAGON"
                # status_msg = "DOCK_STATION"
            
            # Only proceed if we found a target shape (BAD_HEALTH)
            if shape_type == "SQUARE": # Currently targeting Squares based on your uploaded file
                
                # Check for duplicates based on ROBOT position (self.x, self.y)
                if self.is_duplicate_detection(self.x, self.y):
                    continue

                self.get_logger().info(f"✓ Found {shape_type} (Sides: {num_sides})")
                self.handle_detection(status_msg)
                break 

    # --- Helper Functions (Motion Robust Logic) ---

    def is_linear(self, points):
        """
        Returns True if points fit a straight line (Correlation Coeff).
        Used to reject walls misidentified as shapes during turns.
        """
        if len(points) < 3: return True
        
        # Center points
        centered = points - np.mean(points, axis=0)
        
        # PCA / SVD to find primary axis variance
        U, S, Vt = np.linalg.svd(centered)
        
        # Ratio of smallest variance to largest variance
        ratio = S[1] / S[0] if S[0] > 0 else 0
        
        # Threshold: 0.04 implies the object is very thin/flat (Line)
        return ratio < 0.04

    def get_clusters(self, points):
        clusters = []
        if len(points) == 0: return clusters
        
        curr = [points[0]]
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i-1]) < self.cluster_thresh:
                curr.append(points[i])
            else:
                clusters.append(np.array(curr))
                curr = [points[i]]
        clusters.append(np.array(curr))
        return clusters

    def perpendicular_distance(self, pt, line_start, line_end):
        if np.all(line_start == line_end):
            return np.linalg.norm(pt - line_start)
        return np.abs(np.cross(line_end-line_start, line_start-pt)) / np.linalg.norm(line_end-line_start)

    def ramer_douglas_peucker(self, points, epsilon):
        dmax = 0.0
        index = 0
        end = len(points) - 1
        for i in range(1, end):
            d = self.perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        if dmax > epsilon:
            rec1 = self.ramer_douglas_peucker(points[:index+1], epsilon)
            rec2 = self.ramer_douglas_peucker(points[index:], epsilon)
            result = np.vstack((rec1[:-1], rec2))
        else:
            result = np.vstack((points[0], points[end]))
        return result

    def is_duplicate_detection(self, x, y):
        """Check against previous detection locations"""
        for (hx, hy) in self.detected_objects:
            if math.hypot(x - hx, y - hy) < 1.0: # 1 meter radius duplicate check
                return True
        return False

    def handle_detection(self, status):
        """Publishes the detection result in Task 3B format"""
        self.processing = True
        
        # 1. Pause Navigation
        pause_msg = String()
        pause_msg.data = "PAUSE"
        self.pause_pub.publish(pause_msg)
        self.get_logger().info("⏸️  Pausing 2 seconds for stability...")
        time.sleep(2.0)
        
        # 2. Capture latest coordinates
        latest_x = self.x
        latest_y = self.y
        
        # 3. Publish Detection
        # Format: status, x, y, 8 (Altitude/Marker)
        msg = String()
        msg.data = f"{status},{latest_x:.2f},{latest_y:.2f},8"
        self.detection_pub.publish(msg)
        self.get_logger().info(f"✓✓✓ PUBLISHED: {msg.data} ✓✓✓")
        
        # 4. Save to history
        self.detected_objects.append((latest_x, latest_y))
        
        # 5. Resume Navigation
        time.sleep(2.0) # Wait a bit more
        resume_msg = String()
        resume_msg.data = "RESUME"
        self.pause_pub.publish(resume_msg)
        self.get_logger().info("▶️  Resumed")
        
        # Cooldown to prevent immediate re-detection
        time.sleep(5.0) 
        self.processing = False

def main():
    rclpy.init()
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
