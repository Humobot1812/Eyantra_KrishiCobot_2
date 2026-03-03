#!/usr/bin/python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import TransformBroadcaster
from typing import Optional

SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Node for fruit detection and TF publishing.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.team_id = "1732" # CHANGED to string for consistency
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # ... (Rest of __init__ is same) ...
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info(f"FruitsTF node started - Team ID: {self.team_id}")

    # ... (depthimagecb and colorimagecb are same) ...
    def depthimagecb(self, data):
        try:
            self.original_depth_msg = data
            # self.get_logger().info(f"Depth image encoding: {data.encoding}") # Reduced logging

            if data.encoding == '32FC1':
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
                self.depth_image = depth_data
            elif data.encoding == '16UC1':
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
                depth_meters = depth_data.astype(np.float32) / 1000.0
                self.depth_image = depth_meters
            else:
                depth_data = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
                self.depth_image = depth_data.astype(np.float32)

            if self.depth_image is not None:
                self.depth_image[self.depth_image <= 0] = np.nan
                self.depth_image[np.isinf(self.depth_image)] = np.nan

        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")
            self.depth_image = None

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    # ... (detect_aruco_markers is same) ...
    def detect_aruco_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        image_with_aruco = image.copy()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_with_aruco, corners, ids)
            # ... (Existing drawing code) ...
        return image_with_aruco

    def process_aruco_markers(self, image, depth_image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return image

        # ... (Camera matrix setup is same) ...
        camera_matrix = np.array([[915.3003540039062, 0.0, 642.724365234375],
                                  [0.0, 914.0320434570312, 361.9780578613281],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
        focalX = camera_matrix[0, 0]
        focalY = camera_matrix[1, 1]
        centerCamX = camera_matrix[0, 2]
        centerCamY = camera_matrix[1, 2]

        image_out = image.copy()

        for i, corner in enumerate(corners):
            pts = corner[0]
            cX = int(np.mean(pts[:, 0]))
            cY = int(np.mean(pts[:, 1]))
            marker_id = int(ids[i][0])

            distance = None
            if depth_image is not None:
                distance = self.get_accurate_depth(cX, cY, depth_image)

            if distance is None:
                continue

            optical_pos = self.pixel_to_3d(cX, cY, distance, focalX, focalY, centerCamX, centerCamY)
            base_pos = self.transform_optical_to_base_frame(optical_pos)

            # FIX: Ensure frame names match AutoEval expectation (just "1732_...")
            if marker_id == 3:
                frame_name = f"{self.team_id}_fertilizer_1" # "1732_fertilizer_1"
            elif marker_id == 6:
                frame_name = f"{self.team_id}_aruco_6"
            else:
                continue
            
            transform = self.create_transform_stamped('base_link', frame_name, base_pos)
            self.tf_broadcaster.sendTransform(transform)
            
            # self.get_logger().info(f"Published TF for ArUco {marker_id}") # Reduced logging

            cv2.circle(image_out, (cX, cY), 6, (255, 0, 0), -1)
            cv2.putText(image_out, f"Aruco {marker_id}", (cX - 30, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return image_out

    def bad_fruit_detection(self, rgb_image):
        bad_fruits = []
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # ... (Your existing HSV ranges) ...
        lower_green = np.array([35, 100, 80])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        lower_grey = np.array([0, 0, 50])
        upper_grey = np.array([180, 50, 200])
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)
        mask_green_dilated = cv2.dilate(mask_green, None, iterations=3)
        mask_grey_dilated = cv2.dilate(mask_grey, None, iterations=3)
        overlap_mask = cv2.bitwise_and(mask_green_dilated, mask_grey_dilated)
        
        contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- FIX: SORT CONTOURS LEFT-TO-RIGHT ---
        if len(contours) > 0:
            # Sort contours based on the x-coordinate of the bounding box
            contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        # ----------------------------------------

        fruit_id = 0 # Start from 0 for 0-based indexing if preferred, or 1. 
                     # Usually 0 is better for arrays, but let's stick to your logic if 1-based is needed.
                     # AutoEval often expects 0, 1, 2. Let's use 0.
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.4 < aspect_ratio < 2.5:
                        distance = None
                        if self.depth_image is not None:
                            distance = self.get_accurate_depth(cX, cY, self.depth_image)
                        fruit_info = {
                            'center': (cX, cY),
                            'distance': distance,
                            'angle': 0,
                            'width': w,
                            'id': fruit_id,
                            'area': area,
                            'contour': contour
                        }
                        bad_fruits.append(fruit_info)
                        fruit_id += 1
        return bad_fruits

    # ... (get_accurate_depth, pixel_to_3d, transform_optical_to_base_frame, create_transform_stamped are same) ...
    def get_accurate_depth(self, x: int, y: int, depth_image: np.ndarray) -> Optional[float]:
        if depth_image is None: return None
        height, width = depth_image.shape
        if x < 0 or x >= width or y < 0 or y >= height: return None
        depth_samples = []
        valid_samples = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                sample_x = x + dx
                sample_y = y + dy
                if 0 <= sample_x < width and 0 <= sample_y < height:
                    depth_val = depth_image[sample_y, sample_x]
                    if not np.isnan(depth_val) and 0.1 < depth_val < 3.0:
                        depth_samples.append(depth_val)
                        valid_samples += 1
        if valid_samples < 5: return None
        depth_samples = np.array(depth_samples)
        Q1 = np.percentile(depth_samples, 25)
        Q3 = np.percentile(depth_samples, 75)
        IQR = Q3 - Q1
        filtered_depths = depth_samples[(depth_samples >= (Q1 - 1.5 * IQR)) & (depth_samples <= (Q3 + 1.5 * IQR))]
        if len(filtered_depths) == 0: return None
        return np.median(filtered_depths)

    def pixel_to_3d(self, u, v, depth, focalX, focalY, centerCamX, centerCamY):
        z = depth
        x = (u - centerCamX) * z / focalX
        y = (v - centerCamY) * z / focalY
        return (x, y, z)

    def transform_optical_to_base_frame(self, point_optical):
        x_opt, y_opt, z_opt = point_optical
        x_cam = z_opt
        y_cam = -x_opt
        z_cam = -y_opt
        pitch = -0.733
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        x_base = (x_cam * cos_p - z_cam * sin_p) - 1.095239
        y_base = y_cam
        z_base = (x_cam * sin_p + z_cam * cos_p) + 1.10058
        return (x_base, y_base, z_base)

    def create_transform_stamped(self, parent_frame, child_frame, translation):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.w = 1.0
        return t

    def process_image(self):
        # ... (Constants same) ...
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312

        if self.cv_image is None or self.depth_image is None:
            return

        try:
            # Process ArUco
            vis_image = self.process_aruco_markers(self.cv_image, self.depth_image)

            # Process Bad Fruits
            detections = self.bad_fruit_detection(self.cv_image)
            
            if not detections and SHOW_IMAGE:
                cv2.imshow("fruits_tf_view", vis_image)
                cv2.waitKey(1)
                return

            published_count = 0
            for fruit_info in detections:
                cX, cY = fruit_info['center']
                distance = fruit_info['distance']
                fruit_id = fruit_info['id']
                # ... (Visualization code) ...
                contour = fruit_info['contour']
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis_image, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(vis_image, "bad fruit", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if distance is None:
                    continue
                
                # Compute Transform
                optical_pos = self.pixel_to_3d(cX, cY, distance, focalX, focalY, centerCamX, centerCamY)
                base_link_pos = self.transform_optical_to_base_frame(optical_pos)
                
                # FIX: Naming convention "1732_bad_fruit_0" (using 0-based index logic from loop)
                frame_name = f"{self.team_id}_bad_fruit_{fruit_id}" 
                transform = self.create_transform_stamped('base_link', frame_name, base_link_pos)
                self.tf_broadcaster.sendTransform(transform)
                
                self.get_logger().info(f"Published {frame_name}")
                published_count += 1

            if SHOW_IMAGE:
                cv2.imshow("fruits_tf_view", vis_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
