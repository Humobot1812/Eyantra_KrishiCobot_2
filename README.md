# 🌱 KrishiCobot – eYRC 2025–26

### Multi-Robot Autonomous System using ROS 2

> Developed for **e-Yantra Robotics Competition (eYRC) 2025–26**
> Theme: **KrishiCobot (KC)**

---

## 🚀 Project Overview

KrishiCobot is a complete **multi-robot autonomous agricultural warehouse system** built using **ROS 2**.

The system integrates:

* 🚗 eBot (Mobile Robot Navigation)
* 🦾 UR5 Manipulator (Dock-based)
* 👁️ Vision + Depth Perception
* 📡 TF2 Frame Broadcasting
* 🔺 LiDAR-based Shape Detection
* 🌾 Fertilizer Handling & Fruit Sorting

The mission executes in a **single continuous autonomous run**.

---

# 🏗 System Architecture

## 1️⃣ eBot (Mobile Robot)

* Starts from home position
* Detects dock station (Pentagon using LiDAR)
* Receives fertilizer from UR5
* Navigates all greenhouse lanes
* Detects plant health shapes:

  * 🔺 Triangle → `FERTILIZER_REQUIRED`
  * ◼ Square → `BAD_HEALTH`
* Publishes detection messages
* Returns to dock
* Returns to home position

---

## 2️⃣ UR5 Manipulator (Stationed at Dock)

* Loads fertilizer onto eBot
* Detects and removes only defective fruits
* Uses servo-based control
* Uses intermediate waypoints to avoid singularity
* Uses Attach/Detach link services
* Unloads fertilizer after mission completion

---

# 📂 Repository Structure

```
KrishiCobot/
│
├── Hardware/
│   ├── Videos        # Glampse of Hardware
│   ├── ebot_nav_hardware.py        # Real robot navigation
│   └── perception_hardware.py      # Real camera perception
│
├── simulation/
│   ├── Images
│   ├── Videos        
│   ├── ebot_nav.py                 # eBot navigation (simulation)
│   ├── manipulation.py             # UR5 pick & place logic
│   ├── perception.py               # Vision + TF publishing
│   └── shape_detector.py           # LiDAR RANSAC shape detection
│
└── README.md
```

---

# 🧠 Core Concepts Implemented

### 🔹 Navigation

* Waypoint-based motion
* LiDAR obstacle avoidance
* Dock detection
* Collision-aware execution

### 🔹 Shape Detection

* Polar → Cartesian conversion
* Clustering of LiDAR points
* RANSAC line fitting
* Edge connectivity verification
* Angle-based square classification

### 🔹 Vision System

* ArUco marker detection
* HSV-based fruit segmentation
* Contour-based centroid extraction
* Depth filtering (IQR outlier rejection)
* TF publishing for:

  * Fertilizer
  * Bad fruits
  * Dock marker

### 🔹 Manipulation

* Delta twist servo control
* Position + orientation PID logic
* Intermediate waypoints
* Singularity avoidance
* Attach/Detach services

---

# ⚙️ Tech Stack

* ROS 2
* Gazebo
* TF2
* OpenCV
* NumPy
* SciPy (Rotation transforms)
* Python

---

# 🛠 Installation & Setup

### 1️⃣ Building the workspace and Clone the Repository

```bash
cd ~/colcon_ws
git clone https://github.com/eYantra-Robotics-Competition/eyrc-25-26-krishi-cobot.git ./src/
git checkout tags/v1.0.1 .
git checkout main

cd ~/colcon_ws
colcon build
source install/setup.bash
echo "source ~/colcon_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc      
```
Clone my codes:
```bash
git clone https://github.com/Humobot1812/KrishiCobot.git
cd KrishiCobot
```

### 3️⃣ Launch Simulation

```bash
ros2 launch eyantra_warehouse task4c.launch.py
```

---

# 🎥 Demonstration

📺 YouTube Demo:
👉 https://youtu.be/KQTkCjBqOxc?si=yl_s3IojYleHSmcs

---
📺 YouTube playlist:
👉 https://youtube.com/playlist?list=PLQcgql__dXre4uqwJgKRXPLFkSnAbvKIY&si=tBioVZB9U1Ie4ruB


---

# 📚 Key Learnings

This project strengthened my understanding of:

* Multi-robot coordination
* TF frame management
* Sensor fusion (LiDAR + Camera + IMU)
* Real-time decision publishing
* Manipulator servo control
* Robotics system architecture
* Writing deterministic autonomous logic

---

---

# 🙏 Acknowledgement

Grateful to **e-Yantra, IIT Bombay** for designing a real-world agricultural robotics challenge that integrates perception, manipulation, and navigation in a single system.

---

