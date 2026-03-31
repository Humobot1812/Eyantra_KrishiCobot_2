
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
        super().__init__('shape_detector_task')

        self.x = None
        self.y = None
        self.yaw = None
        self.scan = None
        self.detection_paused = False

        self.last_detection_time = 0
        self.detection_cooldown = 0.0
        self.detected_shapes = []

        self.DETECTION_RADIUS = 1.0

        # RANSAC parameters
        self.ransac_n = 2
        self.ransac_k = 100
        self.ransac_t = 0.05
        self.ransac_d = 3

        # Clustering parameters
        self.min_points_per_cluster = 3
        self.det=False
        # Connectivity parameters
        # Note: value kept as in your code; unit comment was misleading earlier
        self.connection_threshold = 3.0  # edges within this distance are "connected"

        self.create_subscription(Odometry, '/odom', self.odom_callback, 30)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/fruit_detection_control', self.control_callback, 10)


        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        self.pause_pub = self.create_publisher(String, '/pause_navigation', 10)

        self.create_timer(0.2, self.detect_shapes)

        # self.get_logger().info('✓ RANSAC-Enhanced + Connectivity Check (Connected-pair angles only)')
        # self.get_logger().info('  Angle test uses only connected edge pairs')

    
    def control_callback(self, msg):
        """Pause or resume detection based on /fruit_detection_control"""
        if msg.data == "Fruit_detection_pause":
            self.detection_paused = True
            self.get_logger().info("⏸️ Fruit detection paused")
        elif msg.data == "Fruit_detection_resume":
            self.detection_paused = False
            self.get_logger().info("▶️ Fruit detection resumed")
            
            
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, msg):
        self.scan = msg

    def polar_to_cartesian(self, ranges, angles):
        """Convert polar to Cartesian"""
        points = []
        for r, theta in zip(ranges, angles):
            if 0.05 < r < 3.0:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, theta])

        return np.array(points) if len(points) > 0 else np.array([])

    def group_contiguous_points(self, points, max_angle_gap=np.radians(5)):
        """Group points into edge clusters"""
        if len(points) == 0:
            return []

        clusters = []
        current_cluster = [points[0]]

        for i in range(1, len(points)):
            angle_gap = abs(points[i][2] - points[i-1][2])
            spatial_gap = np.linalg.norm(points[i][:2] - points[i-1][:2])

            if angle_gap < max_angle_gap and spatial_gap < 0.3:
                current_cluster.append(points[i])
            else:
                if len(current_cluster) >= self.min_points_per_cluster:
                    clusters.append(np.array(current_cluster))
                current_cluster = [points[i]]

        if len(current_cluster) >= self.min_points_per_cluster:
            clusters.append(np.array(current_cluster))

        return clusters

    def ransac_line_fit(self, points):
        """RANSAC algorithm for robust line fitting"""
        if len(points) < self.ransac_n:
            return None, []

        best_inliers = []
        best_model = None
        best_error = np.inf

        points_2d = points[:, :2]

        for _ in range(self.ransac_k):
            if len(points_2d) < self.ransac_n:
                break

            idx = np.random.choice(len(points_2d), self.ransac_n, replace=False)
            maybe_inliers = points_2d[idx]

            p1, p2 = maybe_inliers
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue

            # Line: ax + by + c = 0
            a = dy
            b = -dx
            c = -(a * p1[0] + b * p1[1])

            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-6:
                continue
            a, b, c = a/norm, b/norm, c/norm

            distances = np.abs(a * points_2d[:, 0] + b * points_2d[:, 1] + c)
            inlier_mask = distances < self.ransac_t
            inliers = points_2d[inlier_mask]

            if len(inliers) > len(best_inliers):
                if len(inliers) >= self.ransac_d:
                    better_model = (a, b, c)
                    this_error = np.mean(distances[inlier_mask]**2)

                    if this_error < best_error:
                        best_error = this_error
                        best_model = better_model
                        best_inliers = inliers

        return best_model, best_inliers

    def fit_line_to_cluster_ransac(self, cluster):
        """Fit line to cluster using RANSAC"""
        if len(cluster) < self.ransac_n:
            return None

        model, inliers = self.ransac_line_fit(cluster)

        if model is None or len(inliers) < self.ransac_d:
            return None

        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid

        U, S, Vt = np.linalg.svd(centered)
        direction = Vt[0]

        # Endpoints from projections along direction
        projections = np.dot(centered, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        endpoint1 = centroid + min_proj * direction
        endpoint2 = centroid + max_proj * direction

        return {
            'direction': direction,
            'centroid': centroid,
            'endpoint1': endpoint1,
            'endpoint2': endpoint2,
            'model': model,
            'inliers': inliers,
            'num_inliers': len(inliers),
            'cluster': cluster
        }

    def check_edges_connected(self, lines, return_pairs=False):
        """
        Check if consecutive edges (i,(i+1)%N) are connected by endpoint proximity.
        Returns:
          - connections (int)
          - connected_pairs (list of (i, j)) if return_pairs=True
        """
        n = len(lines)
        if n < 2:
            return (0, []) if return_pairs else 0

        connections = 0
        connected_pairs = []

        for i in range(n):
            j = (i + 1) % n  # adjacent around contour

            ep1_1 = lines[i]['endpoint1']; ep1_2 = lines[i]['endpoint2']
            ep2_1 = lines[j]['endpoint1']; ep2_2 = lines[j]['endpoint2']

            d11 = np.linalg.norm(ep1_1 - ep2_1)
            d12 = np.linalg.norm(ep1_1 - ep2_2)
            d21 = np.linalg.norm(ep1_2 - ep2_1)
            d22 = np.linalg.norm(ep1_2 - ep2_2)

            min_dist = min(d11, d12, d21, d22)

            if min_dist < self.connection_threshold:
                connections += 1
                if return_pairs:
                    connected_pairs.append((i, j))
                self.get_logger().info(f"  ✓ Edge {i}-{j} connected: {min_dist:.3f}m")
            else:
                self.get_logger().info(f"  ✗ Edge {i}-{j} NOT connected: {min_dist:.3f}m")

        if return_pairs:
            return connections, connected_pairs
        return connections

    def calculate_angle_between_lines(self, line1_dir, line2_dir):
        """Calculate angle between two lines (0-90°)"""
        d1 = line1_dir / np.linalg.norm(line1_dir)
        d2 = line2_dir / np.linalg.norm(line2_dir)

        cos_angle = np.abs(np.dot(d1, d2))
        cos_angle = np.clip(cos_angle, 0, 1)

        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def classify_by_edges_and_angles(self, clusters):
        """
        Classify shape using:
        1. 4 edges
        2. Connectivity satisfied
        3. Angles computed ONLY between connected edge pairs
        """
        n_edges = len(clusters)

        self.get_logger().info(f"DEBUG: Detected {n_edges} edge clusters")

        if n_edges != 4:
            self.get_logger().info(f"  → Not 4 edges (need exactly 4 for square)")
            return None

        # Fit RANSAC lines
        lines = []
        for i, cluster in enumerate(clusters):
            line = self.fit_line_to_cluster_ransac(cluster)
            if line is not None:
                lines.append(line)
                self.get_logger().info(f"  Cluster {i}: {line['num_inliers']}/{len(cluster)} inliers")

        self.get_logger().info(f"DEBUG: Fitted {len(lines)} RANSAC lines")

        if len(lines) < 4:
            return None

        # Connectivity with connected pair indices
        self.get_logger().info("DEBUG: Checking edge connectivity...")
        connections, connected_pairs = self.check_edges_connected(lines, return_pairs=True)

        self.get_logger().info(f"DEBUG: Connections found: {connections}/4")

        # Keep your strictness (requires all 4 connected); adjust if desired
        if connections < 4:
            self.get_logger().info(f"  → Only {connections} edges connected (need at least 4)")
            return None

        # Compute angles ONLY for connected pairs
        angles = []
        for (i, j) in connected_pairs:
            ang = self.calculate_angle_between_lines(lines[i]['direction'], lines[j]['direction'])
            angles.append(ang)
            self.get_logger().info(f"  Connected angle {i}-{j}: {ang:.1f}°")

        if len(angles) == 0:
            self.get_logger().info("  → No connected edge pairs to evaluate")
            return None

        avg_angle = np.mean(angles)
        self.get_logger().info(f"DEBUG: Average angle (connected only): {avg_angle:.1f}°")

        # Right-angle count from connected-only angles
        angles_close_to_90 = [abs(a - 90) < 10 for a in angles]
        num_right_angles = sum(angles_close_to_90)
        self.get_logger().info(f"  Right angles (connected only): {num_right_angles}/{len(angles)}")

        # Preserve your existing acceptance logic
        if num_right_angles >= 2 and len(angles) == 4:
            self.get_logger().info("✓ 4 edges + ~90° angles (connected-only) + connected → SQUARE")
            return 'square'
        elif len(angles) == 4:
            self.get_logger().info("✓ 4 edges (connected) → SQUARE")
            return 'square'

        return None

    def calculate_shape_center(self, points):
        """Center of all detected points"""
        if len(points) == 0:
            return None
        return np.mean(points[:, :2], axis=0)

    def calculate_distance_from_robot(self, shape_center):
        """Distance to shape"""
        if shape_center is None:
            return float('inf')
        return np.linalg.norm(shape_center)

    def is_duplicate_detection(self, x, y, shape_type):
        """Check duplicate"""
        for prev_x, prev_y, prev_shape in self.detected_shapes:
            distance = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
            if distance < 0.5 and shape_type == prev_shape:
                return True
        return False

    def detect_shapes(self):
        """Main detection loop"""
        
        if self.detection_paused:
            return
        
        if self.scan is None or self.x is None:
            return

        ranges = np.array(self.scan.ranges)
        angles = np.linspace(
            self.scan.angle_min,
            self.scan.angle_max,
            len(ranges)
        )

        points = self.polar_to_cartesian(ranges, angles)

        if len(points) < 10:
            return

        clusters = self.group_contiguous_points(points)

        if len(clusters) < 4:
            self.get_logger().debug(f"Not enough clusters: {len(clusters)}")
            return

        self.get_logger().info(f"DEBUG: Total clusters detected: {len(clusters)}")

        shape_type = self.classify_by_edges_and_angles(clusters)

        if shape_type is None:
            return

        shape_center = self.calculate_shape_center(points)
        distance = self.calculate_distance_from_robot(shape_center)

        self.get_logger().info(f"SHAPE: {shape_type} at distance {distance:.2f}m")

        if distance > self.DETECTION_RADIUS:
            self.get_logger().info(f"  → Too far ({distance:.2f}m > {self.DETECTION_RADIUS}m)")
            return

        # current_time = time.time()
        # if current_time - self.last_detection_time < self.detection_cooldown:
        #     return

        if self.is_duplicate_detection(self.x, self.y, shape_type):
            self.get_logger().info("  → Duplicate detection")
            return
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            return

        if shape_type == 'square':
            status = "BAD_HEALTH"
        else:
            return

        
        msg = String()
        
        time.sleep(4.0)
        latest_x = self.x
        latest_y = self.y

        if self.det==False:
            msg.data = f"{status},{latest_x:.2f},{latest_y:.2f},4"
            self.det=True
        else:
            msg.data = f"{status},{latest_x:.2f},{latest_y:.2f},5"
            
        self.detection_pub.publish(msg)

        self.detected_shapes.append((latest_x, latest_y, shape_type))
        self.last_detection_time = current_time

        self.get_logger().info(f"✓✓✓ PUBLISHED: {msg.data} ✓✓✓")

        # Pause/Resume
        pause_msg = String()
        pause_msg.data = "PAUSE"
        self.pause_pub.publish(pause_msg)
        self.get_logger().info("⏸️  Pausing 2 seconds...")
        time.sleep(4.0)

        resume_msg = String()
        resume_msg.data = "RESUME"
        self.pause_pub.publish(resume_msg)
        self.get_logger().info("▶️  Resumed")
        time.sleep(20.0)


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
