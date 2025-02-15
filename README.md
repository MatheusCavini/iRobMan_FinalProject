# Intelligent Robotic Manipulation (WiSe 2024/25) - Final Project

### Student 1: Daniel Bellardi Kerzner - XXXXXX
### Student 2: Matheus Latorre Cavini - 2261960 


## Brief Introduction to the Project

The overall goal of the project consists in perfoming a task of grasping, transporting and placing a object in a simulated environment, using a 7-DOF robotic arm. 

The simulation, as well as all the code implemented for the task, shall be done in Python language, employing the `pybullet` library, which offers support for robotics simulation. 

The elements of the simulation are:
- A table 
- A Franka robot arm placed on top of the table always at a fixed position.
- An object from the YCB dataset, which is spawned at region of the table which some randomness on its position and orientation.
- A basket placed on the oposite side of the table always at a fixed position.
- Two spherical moving obstacles above the table, following a linear trajectory with random jittering.
- Two RGB-D cameras, which provide color, depth and also segmentation information, being one placed at the end effector of the robot arm, and other placed above the table.

The general outline for the tasks that should be performed in order to achive the overall goal are:
1. **Perception:** estimate the 6D pose of the object to be grasped.
2. **Control**: control robot arm end effector pose using one of the avaiable Jacobian methods.
3. **Grasping:**: sample and perform a proper grasp to firmly catch the object.
4. **Localization and Tracking**: use computer vision and tracking methods avaiable to keep track of the obstacles positions at the environment.
5. **Planning**: plan a trajectory to carry the grasped object to the basket, taking in account obstacle positions, in order to avoid collisions.

All of the tasks shall be performed using sensing information provided only by the 2 cameras and the internal sensors of the robot (joint positions, velocities, etc.), that meaning no ground truth information about object or obstacles is avaiable.

## Task 1: Perception

## Task 2: Control

To perform the pose control for the robot end effector, the method applied was the **Pseudoinverse of Jacobian** [1]. This control method is described by the law:

$$ \Delta \theta =  \text{J}^{\dagger}\cdot \Delta p  $$

Where:
- $\Delta p$ is the difference from the desired to the current 6D pose of end effector.
- $\text{J}^{\dagger}$ is the pseudoinverse of the jacobian matrix for the current joint configuration of the robot.
- $\Delta \theta$ is the position variation that should be applied to joints, given as a $N$-dimensional vector, where $N$ is the number of active joints.

In order to implement this controller in code, a method for the already existing class `Robot` was created named `IK_Solver`, which takes as argument the desired pose of the end effector and returns the new joint position. It makes use of a helping method, `compute_pose_error`, which as the name suggests, calculate the error between the desired and current pose. Lastly, the method `move_to_pose` uses the `IK_Solver` together with the already existing `position_control` to move the joints.


```Python
def compute_pose_error(self, target_pos, current_pos, target_ori, current_ori):
    position_error = np.array(target_pos) - np.array(current_pos)
    orientation_error_quat = p.getDifferenceQuaternion(current_ori, target_ori)
    orientation_error = 2 * np.array(orientation_error_quat[:3])  # Extract vector part
    
    return np.hstack((position_error, orientation_error))


def IK_solver(self, target_pos, target_ori):
    #Calculate the error DeltaP between the current and desired end effector pose
    current_ee_pos, current_ee_ori = self.get_ee_pose()
    joint_positions = self.get_joint_positions()
    joint_velocities = self.get_joint_velocites()
    error = self.compute_pose_error(target_pos, current_ee_pos, target_ori, current_ee_ori)

    #If position error is small enough, stop updating joint positions
    if np.linalg.norm(error[:3]) < 0.01 and np.linalg.norm(error[4:]) <0.01:
            return joint_positions
        

    #Calculate the Jacobian
    zeros = np.zeros(len(joint_positions))
    jacobian_linear, jacobian_angular = p.calculateJacobian(
        self.id, 
        self.ee_idx,
        localPosition=[0, 0, 0], 
        objPositions=joint_positions.tolist()+[0,0],   
        objVelocities=joint_velocities.tolist()+[0,0], 
        objAccelerations=zeros.tolist()+[0,0]
    )
    jacobian = np.vstack((jacobian_linear, jacobian_angular))[:, :7]

    #Calculate the pseudo-inverse of the Jacobian
    jacobian_pseudo_inv = np.linalg.pinv(jacobian)

    #Calculate joint_increment DeltaTheta
    delta_theta = jacobian_pseudo_inv @ error

    #Update the joint positions
    joint_positions += delta_theta

    return joint_positions


def move_to_pose(self, target_pos, target_ori):
    joint_postions = self.IK_solver(target_pos, target_ori)
    self.position_control(joint_postions)
```


## References
[1] Buss, Samuel R. "Introduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods." IEEE Journal of Robotics and Automation 17.1-19 (2004): 16.