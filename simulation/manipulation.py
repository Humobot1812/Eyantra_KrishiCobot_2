#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
*****************************************************************************************
* eYRC Krishi CoBot 2025-26 | Team ID: 1732
* UR5 Pick & Place Extended Task (FINAL FIXED VERSION)
*****************************************************************************************
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import numpy as np
import time
from scipy.spatial.transform import Rotation
from linkattacher_msgs.srv import AttachLink, DetachLink

BASE_FRAME = "base_link"
EEF_FRAME = "wrist_3_link"

# --- Motion Constants ---
LINEAR_KP = 1.5
MAX_LINEAR_VEL = 30.0
POSITION_TOLERANCE = 0.08
ANGULAR_KP = 0.8
ORIENTATION_TOLERANCE = 0.1
WAIT_AT_WAYPOINT = 0.8

# --- Waypoint Orientations ---
PICK_ORIENTATION = Rotation.from_euler('x', 90, degrees=True)
DROP_ORIENTATION = Rotation.from_euler('y', 180, degrees=True)
FRUIT_PICK_ORIENTATION = Rotation.from_euler('x', 180, degrees=True) * Rotation.from_euler('z', 90, degrees=True)
INTERMEDIATE_P2_ORN = Rotation.from_quat(np.array([0.029, 0.997, 0.045, 0.033]))

TRASH_BIN_POS = np.array([-0.806, 0.010, 0.182])
INTERMEDIATE_1_POS = np.array([-0.159, 0.501, 0.600])
INTERMEDIATE_2_POS = np.array([-0.150, 0.501, 0.600])
INTERMEDIATE_0_POS = np.array([0.150, 0, 0.600])

# Offsets
HOVER_OFFSET = np.array([0.0, 0.0, 0.2])
FERTILIZER_PICK_OFFSET = np.array([0.0, 0.0, 0.02])
FRUIT_PICK_OFFSET = np.array([0.0, 0.0, -0.04])

# Object names
FERTILIZER_MODEL = "fertiliser_can"
BAD_FRUIT_MODEL = "bad_fruit"
ROBOT_MODEL = "ur5"
ROBOT_GRIP_LINK = "wrist_3_link"
OBJECT_LINK = "body"


class UR5PickPlace(Node):
    def __init__(self):
        super().__init__("ur5_pick_place")
        self.get_logger().info("=== UR5 Pick & Place Node Started (Team 1732) ===")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        self.status_pub = self.create_publisher(String, "/ur5_status", 10)

        # Services
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Attach service not available, waiting...')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Detach service not available, waiting...')

        # SUBSCRIBE TO DOCK_STATION
        self.create_subscription(String, "/detection_status", self.dock_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        # Internal vars
        self.team_id = "1732"
        self.state = "WAIT_FOR_DOCK"
        self.tf_positions = {}
        self.sequence = []
        self.current_index = 0
        self.reached_target = False
        self.last_reach_time = None
        self.service_call_in_progress = False
        self.service_future = None
        
        
        
        self.phase = "INITIAL"     # INITIAL → FRUIT_DONE → FINAL_FERTILIZER
        self.waiting_for_second_dock = False


    # ===================== DOCK CALLBACK =====================
    # def dock_callback(self, msg):
    #     if "DOCK_STATION" in msg.data:
    #         self.get_logger().info("📩 DOCK RECEIVED — Starting sequence")
    #         self.state = "WAIT_FOR_TFS"
            
            
            
    def dock_callback(self, msg):
        if "DOCK_STATION" not in msg.data:
            return

        # FIRST DOCK → normal start
        if self.phase == "INITIAL":
            self.get_logger().info("📩 FIRST DOCK RECEIVED — Starting initial sequence")
            self.state = "WAIT_FOR_TFS"

        # SECOND DOCK → final fertilizer pickup
        elif self.phase == "FRUIT_DONE":
            self.get_logger().info("📩 SECOND DOCK RECEIVED — Starting final fertilizer disposal")
            self.build_final_fertilizer_sequence()


    # ===================== TF HELPERS =====================
    def get_tf_pos(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time())
            return np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
        except:
            return None

    def stop(self):
        self.cmd_pub.publish(Twist())

    def get_eef_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EEF_FRAME, rclpy.time.Time())
            pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z
            ])
            orn = Rotation.from_quat([
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w
            ])
            return pos, orn
        except:
            return None, None

    # ===================== MOTION CONTROL =====================
    def move_to_pose(self, target_pos, target_orn):
        pos, orn = self.get_eef_pose()
        if pos is None:
            return False

        # Linear
        error = target_pos - pos
        dist = np.linalg.norm(error)
        linear_cmd = LINEAR_KP * error if dist > POSITION_TOLERANCE else np.zeros(3)

        if np.linalg.norm(linear_cmd) > MAX_LINEAR_VEL:
            linear_cmd = linear_cmd / np.linalg.norm(linear_cmd) * MAX_LINEAR_VEL

        # Angular
        rot_err = target_orn * orn.inv()
        ang_err = rot_err.as_rotvec()
        ang_dist = np.linalg.norm(ang_err)
        angular_cmd = ANGULAR_KP * ang_err if ang_dist > ORIENTATION_TOLERANCE else np.zeros(3)

        # Reached
        if dist < POSITION_TOLERANCE and ang_dist < ORIENTATION_TOLERANCE:
            self.stop()
            return True

        # Publish motion
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = linear_cmd
        msg.angular.x, msg.angular.y, msg.angular.z = angular_cmd
        self.cmd_pub.publish(msg)
        return False

    # ===================== SERVICE CALL =====================
    def call_gripper_service(self, action, model_name):
        req = AttachLink.Request() if action == "attach" else DetachLink.Request()
        req.model1_name = model_name
        req.link1_name = OBJECT_LINK
        req.model2_name = ROBOT_MODEL
        req.link2_name = ROBOT_GRIP_LINK

        if action == "attach":
            self.get_logger().info(f"🔗 Attaching {model_name}...")
            self.service_future = self.attach_client.call_async(req)
        else:
            self.get_logger().info(f"🔓 Detaching {model_name}...")
            self.service_future = self.detach_client.call_async(req)

        self.service_call_in_progress = True

    # ===================== CONTROL LOOP =====================
    def control_loop(self):
        if self.state == "WAIT_FOR_DOCK":
            return
        if self.state == "WAIT_FOR_TFS":
            self.collect_all_tfs()
            return
        if self.state == "MOVE_SEQUENCE":
            self.follow_sequence()
            return
        if self.state == "DONE":
            self.stop()
            return

    # ===================== COLLECT TFs =====================
    def collect_all_tfs(self):
        frames = [
            f"{self.team_id}_fertilizer_1",
            f"{self.team_id}_aruco_6",
            f"{self.team_id}_bad_fruit_0",
            f"{self.team_id}_bad_fruit_1",
            f"{self.team_id}_bad_fruit_2",
        ]

        for f in frames:
            pos = self.get_tf_pos(f)
            if pos is None:
                self.get_logger().info(f"Waiting for TF: {f}")
                return
            self.tf_positions[f] = pos

        # Fixed landmarks
        self.tf_positions["trash_bin"] = TRASH_BIN_POS
        self.tf_positions["intermediate_0"] = INTERMEDIATE_0_POS
        self.tf_positions["intermediate_1"] = INTERMEDIATE_1_POS

        self.get_logger().info("✅ All TF positions collected.")

        # ===================== BUILD SEQUENCE =====================
        self.sequence = []

        # PICK FERTILIZER
        pick = self.tf_positions[f"{self.team_id}_fertilizer_1"]
        self.sequence.append({
            'pos': pick + FERTILIZER_PICK_OFFSET,
            'orn': PICK_ORIENTATION,
            'label': "Pick Fertilizer",
            'action': "attach",
            'model': FERTILIZER_MODEL
        })

        # INTERMEDIATE-0
        p0 = self.tf_positions["intermediate_0"]
        self.sequence.append({
            'pos': p0,
            'orn': DROP_ORIENTATION,
            'label': "Intermediate 0",
            'action': "none"
        })

        # DROP FERTILIZER
        drop = self.tf_positions[f"{self.team_id}_aruco_6"]
        self.sequence.append({
            'pos': drop + HOVER_OFFSET,
            'orn': DROP_ORIENTATION,
            'label': "Drop Fertilizer",
            'action': "detach",
            'model': FERTILIZER_MODEL
        })

        # PUBLISH FERTILIZER DONE
        self.sequence.append({
            'pos': drop + HOVER_OFFSET,
            'orn': DROP_ORIENTATION,
            'label': "Notify Done",
            'action': "notify",
            'model': ""
        })

        # INTERMEDIATE-1
        p2 = self.tf_positions["intermediate_1"]
        self.sequence.append({
            'pos': p2,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Intermediate-1",
            'action': "none"
        })

        # BAD FRUIT LOOP
        for i in range(3):
            fruit = self.tf_positions[f"{self.team_id}_bad_fruit_{i}"]

            # Pick fruit
            self.sequence.append({
                'pos': fruit + FRUIT_PICK_OFFSET,
                'orn': FRUIT_PICK_ORIENTATION,
                'label': f"Pick Fruit {i}",
                'action': "attach",
                'model': BAD_FRUIT_MODEL
            })

            # P2
            self.sequence.append({
                'pos': p2,
                'orn': INTERMEDIATE_P2_ORN,
                'label': "P2",
                'action': "none"
            })

            # DROP fruit
            trash = self.tf_positions["trash_bin"]
            self.sequence.append({
                'pos': trash + HOVER_OFFSET,
                'orn': DROP_ORIENTATION,
                'label': f"Drop Fruit {i}",
                'action': "detach",
                'model': BAD_FRUIT_MODEL
            })

            if i < 2:
                self.sequence.append({
                    'pos': p2,
                    'orn': INTERMEDIATE_P2_ORN,
                    'label': "Return P2",
                    'action': "none"
                })

        p2 = self.tf_positions["intermediate_1"]
        self.sequence.append({
            'pos': p2 ,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Intermediate-1",
            'action': "none"
        })
        
        # pl = self.tf_positions[f"{self.team_id}_aruco_6"]
        # self.sequence.append({
        #     'pos': pl + +np.array([0.0, 0.0, 0.2]),
        #     'orn': DROP_ORIENTATION,
        #     'label': "Intermediate 0",
        #     'action': "none"
        # })

        self.current_index = 0
        self.state = "MOVE_SEQUENCE"
        
        
        
    def build_final_fertilizer_sequence(self):
        frames = [
            f"{self.team_id}_fertilizer_1",
        ]

        for f in frames:
            pos = self.get_tf_pos(f)
            if pos is None:
                self.get_logger().info(f"Waiting for TF: {f}")
                return
            self.tf_positions[f] = pos

        # Fixed landmarks
        self.tf_positions["trash_bin"] = TRASH_BIN_POS
        self.tf_positions["intermediate_0"] = INTERMEDIATE_0_POS
        self.tf_positions["intermediate_1"] = INTERMEDIATE_1_POS

        self.get_logger().info("✅ All TF positions collected.")
        self.sequence = []
        
        # pick fertilizer from eBot
        pick = self.tf_positions[f"{self.team_id}_fertilizer_1"]
        fertilizer_pos = pick + np.array([0.0, -0.05, 0.0025])
        self.sequence.append({
            'pos': fertilizer_pos,
            'orn': DROP_ORIENTATION,
            'label': "Pick Fertilizer From eBot",
            'action': "attach",
            'model': FERTILIZER_MODEL
        })

        
        # back to intermediate_0
        p0 = self.tf_positions["intermediate_0"]
        self.sequence.append({
            'pos': p0,
            'orn': DROP_ORIENTATION,
            'label': "Intermediate 0",
            'action': "none"
        })
        
        self.sequence.append({
            'pos': p0,
            'orn': DROP_ORIENTATION,
            'label': "Notify Done",
            'action': "notify",
            'model': ""
        })

        p2 = self.tf_positions["intermediate_1"]
        self.sequence.append({
            'pos': p2,
            'orn': FRUIT_PICK_ORIENTATION,
            'label': "Intermediate-1",
            'action': "none"
        })
        
        # drop fertilizer in trash
        trash = self.tf_positions["trash_bin"]
        self.sequence.append({
            'pos': trash + HOVER_OFFSET,
            'orn': DROP_ORIENTATION,
            'label': "Drop Fertilizer In Trash",
            'action': "detach",
            'model': FERTILIZER_MODEL
        })

        self.current_index = 0
        self.reached_target = False
        self.phase = "FINAL_FERTILIZER"
        self.state = "MOVE_SEQUENCE"


    # ===================== FOLLOW SEQUENCE (WITH FIXED DETACH) =====================
    def follow_sequence(self):

        # --- WAIT FOR ONGOING SERVICE ---
        if self.service_call_in_progress:
            if self.service_future.done():
                self.get_logger().info("✔ Gripper service completed")
                self.service_call_in_progress = False
                self.current_index += 1
                self.reached_target = False
            return

        # END OF SEQUENCE
        # if self.current_index >= len(self.sequence):
        #     self.state = "DONE"
        #     return
        
        if self.current_index >= len(self.sequence):
            if self.phase == "INITIAL":
                self.get_logger().info("🍎 All bad fruits dumped. Waiting for second dock.")
                self.phase = "FRUIT_DONE"
                self.state = "WAIT_FOR_DOCK"
                return

            if self.phase == "FINAL_FERTILIZER":
                self.get_logger().info("♻ Fertilizer dumped. Task 4C complete.")
                self.state = "DONE"
                return


        step = self.sequence[self.current_index]
        action = step.get("action", "none")

        # ===================== FORCE DETACH FIX =====================
        if action == "detach":
            pos, _ = self.get_eef_pose()
            if pos is not None:
                dist = np.linalg.norm(step['pos'] - pos)
                if dist < POSITION_TOLERANCE:
                    self.get_logger().info("📌 Position close enough → FORCING DETACH")
                    self.call_gripper_service("detach", step['model'])
                    return
        # =============================================================

        reached = self.move_to_pose(step['pos'], step['orn'])
        if not reached:
            return

        if not self.reached_target:
            self.reached_target = True
            self.last_reach_time = time.time()
            return

        if time.time() - self.last_reach_time < WAIT_AT_WAYPOINT:
            return

        # ATTACH
        if action == "attach":
            self.call_gripper_service("attach", step['model'])
            return

        # NOTIFY
        if action == "notify":
            msg = String()
            msg.data = "UR5_FERTILIZER_DONE"
            self.status_pub.publish(msg)
            self.get_logger().info("🌱 Published UR5_FERTILIZER_DONE")
            self.current_index += 1
            self.reached_target = False
            return

        # NORMAL waypoint
        self.current_index += 1
        self.reached_target = False


# ===================== MAIN =====================
def main(args=None):
    rclpy.init(args=args)
    node = UR5PickPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
