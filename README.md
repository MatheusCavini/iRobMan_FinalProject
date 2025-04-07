# Intelligent Robotic Manipulation (WiSe 2024/25) - Final Project

### Authors:
- **Daniel Bellardi Kerzner** - 2445155  
- **Matheus Latorre Cavini** - 2261960  

---

## Project Overview

This project simulates a robotic manipulation task using a 7-DOF robotic arm in a simulated environment. The robot is tasked with grasping, transporting, and placing an object while avoiding obstacles. The simulation is implemented in Python using the `pybullet` library for physics-based robotics simulation.

### Key Features:
- **Perception**: Object detection and 6D pose estimation using RGB-D cameras.
- **Control**: End-effector pose control using Jacobian-based inverse kinematics.
- **Grasping**: Grasp generation using the GIGA library and point cloud data.
- **Localization and Tracking**: Obstacle tracking using Kalman filters.
- **Planning**: Trajectory planning with collision avoidance using RRT*.

---

## Dependencies

The project requires the following Python libraries:

- **Core Libraries**:
  - `numpy`
  - `pybullet`
  - `pybullet_object_models`
  - `open3d`
  - `trimesh`
  - `scipy`
  - `yaml`

- **Computer Vision**:
  - `opencv-python`

- **Kalman Filtering**:
  - `filterpy`

- **Grasp Generation**:
  - `giga` (Grasp generation library)

---

## Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/irobman-wise-2425-final-project.git
   cd irobman-wise-2425-final-project
   ```

2. **Set Up a Python Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Manually install the dependencies:
   ```bash
   pip install numpy pybullet pybullet_object_models open3d trimesh scipy yaml opencv-python filterpy giga
   ```

4. **Download YCB Object Models**:
   The simulation uses YCB object models. Ensure the `pybullet_object_models` library is installed and the YCB dataset is downloaded. If not, follow the instructions [here](https://github.com/bulletphysics/bullet3/tree/master/data).

---

## Running the Simulation

1. **Edit Configuration**:
   The simulation settings can be modified in the `test_config.yaml file`. 

2. **Run the Main Script**:
   Execute the simulation by running:
   ```bash
   python main.py
   ```

3. **Simulation Output**:
   - The simulation will display the robot's actions in the PyBullet GUI (if enabled in the configuration).
   - Logs will be printed to the console, detailing the robot's state, object detection, and trajectory planning.

---

## Project Structure

```
irobman-wise-2425-final-project/
├── configs/
│   └── test_config.yaml         # Configuration file for the simulation
├── src/
│   ├── simulation.py            # Main simulation class
│   ├── robot.py                 # Robot control and kinematics
│   ├── objects.py               # Object and obstacle definitions
│   ├── obstacleDetection.py     # Obstacle detection and tracking
│   ├── grasp_generator.py       # Grasp generation using GIGA
│   ├── trajectoryGeneration.py  # Trajectory planning (RRT*)
│   ├── stateMachine.py          # State machine for task execution
│   └── utils.py                 # Utility functions
├── main.py                      # Entry point for the simulation
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

---

## Notes

- **Performance Optimization**:
  - The simulation disables unnecessary rendering and shadows for better performance.
  - Camera rendering is limited to specific intervals to reduce computational load.

- **Debugging**:
  - Debugging information, such as obstacle positions and robot states, is printed to the console.
  - Uncomment visualization functions in the code to enable 3D visualizations of point clouds and trajectories.

---

