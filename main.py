import os
import glob
import yaml
import pybullet as p

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.utils import * 
from src.trajectoryGeneration import *
from src.obstacleDetection import *
#from src.CV import *
from src.objectDetection import *
from src.grasp_generator import *

import cv2



def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")

            depth_threshhold = 0.02
            grabing_distance = True
            global_object_position = False


            ###[MC: 2025-02-15] Test of Jacobian IK-Controller ###
            ###########################################################
            target_position = np.array([ -0.3, -0.3, 1.5])
            axis0 = [0,1,0]
            angle0 =  np.pi
            axis1 = [0, 0, 1]
            angle1 = 0
            target_orientation = concatenate_quaternions(axis_angle_to_quaternion(axis0, angle0), axis_angle_to_quaternion(axis1, angle1)) #...keeping default initial rotation in quaternion
            robot = sim.get_robot()
            
            rgb_static, depth_static, seg_static = sim.get_static_renders()
            rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()

            positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 1000)
            
            ###########################################################
            near = config['world_settings']['camera']['near']
            far = config['world_settings']['camera']['far']
           
            for i in range(10000):
                sim.step()
                
                
    
                

                # for getting renders
                #[MC:2025-02-10] PERFORMANCE: change render FPS
                if i%10 == 0: #Only get renders every 10 steps (240/10 = 24fps on image processing)
                    rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
                    #rgb_static, depth_static, seg_static = sim.get_static_renders()
               

                '''#[MC:2025-02-16] Testing obstacle detection and measuring
                ###########################################################
                    depth_real = real_depth(depth_static, near, far)
                    obstacles_2D_info = detect_obstacle_2D(rgb_static)
                    
                    
                obs_position_guess = np.zeros((2, 3))
                obs_position_guess[0] = obstacle_3D_estimator(obstacles_2D_info, depth_real, sim.projection_matrix, sim.stat_viewMat, 0)
                obs_position_guess[1] = obstacle_3D_estimator(obstacles_2D_info, depth_real, sim.projection_matrix, sim.stat_viewMat, 1)

                obj_2D_info = center_object((seg_static*60).astype(np.uint8)) # X and Y value
                obj_position_guess = np.zeros((1, 3))
                obj_position_guess = object_3D_estimator(obj_2D_info, depth_real, sim.projection_matrix, sim.stat_viewMat)
                ###########################################################
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")'''
       

                if i == 10:
                    target_position = target_position + [0, 0.3, 0]
                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 200)
                
                if i == 210:
                    obj_2D_info = center_object((seg_static*60).astype(np.uint8)) # X and Y value
                    
                    #[DK:2025-02-26] Object detection and coordinate estimation
                    ############################################################################
                    obj_position_guess = np.zeros((1, 3))
                    real_depth_static = real_depth(depth_static, near, far)
                    obj_position_guess = object_3D_estimator(obj_2D_info, real_depth_static, sim.projection_matrix, sim.stat_viewMat)
                    ############################################################################
                    
                    target_position = obj_position_guess + [0, 0, 0.3]
                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 400)

                if i == 610:
                   
                    robot.open_gripper()

                    #[DK:2025-02-26] Object depth detection
                    ############################################################################
                    real_depth_ee = real_depth(depth_ee, near, far)
                    axis2 = [0, 0, 1]
                    angle2 = np.deg2rad(180 - grasping_generator( real_depth_ee , (seg_ee*60).astype(np.uint8)))
                    target_orientation = concatenate_quaternions(target_orientation, axis_angle_to_quaternion(axis2, angle2))
                    ############################################################################

                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 100)
                    grabing_distance = False
                
                if i == 710:
                    target_position = target_position + [0, 0, -2]
                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 10000)
               
                if i == 2200:
                    target_position = target_position + [0, 0, -0.05]
                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 100)
                   
                if i == 2300:
                    target_position = target_position + [0, 0, 0.3]
                    positions, orientations = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 400)
                    robot.close_gripper()

               

                moveAlongTrajectory(robot, positions, orientations,  10, 200, i)
                moveAlongTrajectory(robot, positions, orientations,  210, 400, i)
                moveAlongTrajectory(robot, positions, orientations,  610, 100, i)

                #[DK:2025-02-26] Object depth detection
                ############################################################################
                if not grabing_distance:
                    moveAlongTrajectory(robot, positions, orientations,  710, 10000, i)
                    real_depth_ee = real_depth(depth_ee, near, far)
                    if np.min(real_depth_ee) < depth_threshhold:
                        grabing_distance = True
                        target_position = robot.get_ee_pose()[0]
                 ############################################################################

                
                moveAlongTrajectory(robot, positions, orientations,  2200, 100, i)
                moveAlongTrajectory(robot, positions, orientations,  2300, 400, i)

                print("Step: ", i)
                print("Target position: ", target_position)
                print("Target orientation: ", quaternion_to_axis_angle(target_orientation))
                       
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)

