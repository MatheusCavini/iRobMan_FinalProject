import pybullet as p
import numpy as np
from typing import Tuple, List


TABLE_SCALING = 2.0


class Robot:
    """Robot Class.

    The class initializes a franka robot.

    Args:
        urdf: Path to the robot URDF file
        init_position: Initial position (x,y,z) of the robot base
        orientation: Robot orientation in axis angle representation
                        (roll,pitch,yaw)
        arm_index: List of joint indices for the robot arm
        gripper_index: List of joint indices for the gripper
        ee_index: Index of the end effector link
        arm_default: List of default joint angles for the robot arm and gripper
        table_scaling: Scaling parameter for the table height ^ size
    """
    def __init__(self,
                 urdf: str,
                 init_position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float],
                 arm_index: List[int],
                 gripper_index: List[int],
                 ee_index: int,
                 arm_default: List[float],
                 table_scaling: float = TABLE_SCALING):

        # load robot
        self.pos = init_position
        self.axis_angle = orientation
        self.tscale = table_scaling

        if self.tscale != 1.0:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] * self.tscale]
        self.ori = p.getQuaternionFromEuler(self.axis_angle)

        self.arm_idx = arm_index
        self.default_arm = arm_default
        self.gripper_idx = gripper_index

        self.ee_idx = ee_index

        self.id = p.loadURDF(urdf, self.pos, self.ori,
                             useFixedBase=True)

        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.set_default_position()

        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

    def set_default_position(self):
        for idx, pos in zip(self.arm_idx, self.default_arm):
            p.resetJointState(self.id, idx, pos)

    def get_joint_limits(self):
        lower = []
        upper = []
        for idx in self.arm_idx:
            joint_info = p.getJointInfo(self.id, idx)
            lower.append(joint_info[8])
            upper.append(joint_info[9])
        return lower, upper

    def print_joint_infos(self):
        num_joints = p.getNumJoints(self.id)
        print('number of joints are: {}'.format(num_joints))
        for i in range(0, num_joints):
            print('Index: {}'.format(p.getJointInfo(self.id, i)[0]))
            print('Name: {}'.format(p.getJointInfo(self.id, i)[1]))
            print('Typ: {}'.format(p.getJointInfo(self.id, i)[2]))

    def get_joint_positions(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return np.array([state[0] for state in states])

    def get_joint_velocites(self):
        states = p.getJointStates(self.id, self.arm_idx)
        return np.array([state[1] for state in states])

    def get_ee_pose(self):
        ee_info = p.getLinkState(self.id, self.ee_idx)
        ee_pos = ee_info[0]
        ee_ori = ee_info[1]
        return np.asarray(ee_pos), np.asarray(ee_ori)

    def position_control(self, target_positions):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )

    

    ###[MC: 2025-02-09] Implement Jacobian IK-Controller for robot ###
    ###########################################################
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

###################################################################################

