import os
import glob
import yaml
import pybullet as p
import random
import numpy as np
from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

#Import modules from source
from src.simulation import Simulation
from src.utils import * 
from src.trajectoryGeneration import *
from src.obstacleDetection import *
from src.stateMachine import states, events, transitions
from src.grasp_generator import *

def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    

    ###[MC: 2025-03-06] To get a random object each time
    #random.shuffle(obj_names)
    #####################################################

    sim = Simulation(config)
    

    for obj_name in obj_names:
        #if obj_name != 'c:\\users\\danie\\teste\\pybullet-object-models\\pybullet_object_models\\ycb_objects\\YcbPowerDrill':
        #    continue
      
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



            ###[MC: 2025-03-06] Global variables and definitions for the simulation
            ##########################################################
            # Axis and angle corresponding to the neutral orientation
            axis0 = [0,1,0] 
            angle0 =  np.pi
            robot = sim.get_robot()
            near = config['world_settings']['camera']['near']
            far = config['world_settings']['camera']['far']

            #Related to state machine
            state = states["INIT"]
            event = events["NONE"]
            position_trajectory = None
            orientation_trajectory = None
            start_step = 0

            #Kalman filters for each obstacle
            kf0 = ObstacleKalmanFilter()
            kf1 = ObstacleKalmanFilter()

            #for debugging
            EXECUTE = True
            ###########################################################
           
           
            for i in range(10000):
                sim.step()
    
                
                # for getting renders
                #[MC:2025-02-10] PERFORMANCE: change render FPS
                if i%10 == 0: #Only get renders every 10 steps (240/10 = 24fps on image processing)
                    #rgb_ee, depth_ee, seg_ee = sim.get_ee_renders() ##UNUSED
                    rgb_static, depth_static, seg_static = sim.get_static_renders()
               

                #[MC:2025-02-16] Testing obstacle detection and measuring
                ###########################################################
                    depth_real_static = real_depth(depth_static, near, far)

                    obstacles_2D_info = detect_obstacle_2D(rgb_static)

                    obs_position_guess = np.zeros((2, 3))
                    obstacle0_measure = obstacle_3D_estimator(obstacles_2D_info, depth_real_static, sim.projection_matrix, sim.stat_viewMat, 0)
                    obstacle1_measure = obstacle_3D_estimator(obstacles_2D_info, depth_real_static, sim.projection_matrix, sim.stat_viewMat, 1)
                    obs_position_guess[0] = kf0.update(obstacle0_measure)
                    obs_position_guess[1] = kf1.update(obstacle1_measure)

                    

                


                ###########################################################

                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")


                ##[MC:2025-03-06] SIMULATION LOOP
                ###########################################################

                if EXECUTE:

                    #STEP 1: WAIT FOR SIMULATION TO STABILIZE
                    if i == 120:
                        event = events["SIMULATION_STABLE"]

                    #STEP 2: SEARCH FOR OBJECT USING STATIC CAMERA
                    if state == states["SEARCHING_OBJECT"]:
                        obj_2D_info = center_object((seg_static*60).astype(np.uint8)) # X and Y value
                        
                        #[DK:2025-02-26] Object detection and position estimation
                        ############################################################################
                        obj_position_guess = np.zeros((1, 3))
                        real_depth_static = real_depth(depth_static, near, far)
                        obj_position_guess = point_3D_estimator(obj_2D_info, real_depth_static, sim.projection_matrix, sim.stat_viewMat)
                       
                        if compare_arrays(obj_position_guess, [0, 0, 0]):
                            obj_position_guess = np.array([-0.1, -0.5, 1.4])
                        target_position = obj_position_guess + np.array([-0.3, 0.1, 0.1])
                        ############################################################################
                        
                        target_orientation = axis_angle_to_quaternion(axis0, angle0)
                        goal_guess = np.hstack((target_position, target_orientation))
                        position_trajectory, orientation_trajectory = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 400)
                        event = events["OBJECT_POSITION_ESTIMATED"]
                        start_step = i + 100

                    
                    #STEP 3: MOVE ROBOT ARM OVER OBJECT
                    if state == states["MOVING_TO_OBJECT"]:
                        finished = moveAlongTrajectory(robot, position_trajectory, orientation_trajectory, start_step, i)
                        if finished:
                            event = events["REACHED_OBJECT"]
                            robot.open_gripper()

                    #STEP 4: GENERATE GRASP 
                    if state == states["GENERATING_GRASP"]:

                        #[DK:2025-03-25] Grasping generation
                        ############################################################################
                        # Obtain the ideal position and orientation for the grasp
                        obj_position_guess, obj_orientation_guess_Quaternion = grasp_point_cloud_2((seg_static*60).astype(np.uint8), depth_real_static, sim.projection_matrix, sim.stat_viewMat, robot)

                        # Extract the grasp direction from the ideal orientation
                        grasp_direction = quaternion_to_direction(obj_orientation_guess_Quaternion)
                        # Normalize the direction vector (to ensure unit length)
                        grasp_direction = grasp_direction / np.linalg.norm(grasp_direction)
                        
                        # Compute pre-grasp position
                        target_position = obj_position_guess - 0.15 * grasp_direction

                        target_orientation =  obj_orientation_guess_Quaternion
                        ############################################################################

                        position_trajectory, orientation_trajectory = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 1000)
                        event = events["GRASP_GENERATED"]
                        start_step = i + 100

                    
                    #STEP 5: PRE-GRASP OBJECT
                    if state == states["PRE_GRASPING"]:
                        finished = moveAlongTrajectory(robot, position_trajectory, orientation_trajectory, start_step, i)
                        if finished:
                            event = events["PREGRASP_SUCCESS"]
                            
                            # Compute grasp position
                            target_position =  obj_position_guess + 0.05 * grasp_direction
                            position_trajectory, orientation_trajectory = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 400)
                            start_step = i + 100
                            
                   

                    #STEP 6: GRASP OBJECT
                    if state == states["GRASPING"]:
                        finished = moveAlongTrajectory(robot, position_trajectory, orientation_trajectory, start_step, i)
                        if finished and i > start_step + 500:
                            robot.close_gripper()
                        if finished and i > start_step + 600:
                            event = events["GRASP_SUCCESS"]
                            target_position = robot.get_ee_pose()[0] + np.array([0, 0, 0.15])
                            position_trajectory, orientation_trajectory = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 400)
                            start_step = i + 100

                    #STEP 7: LIFT OBJECT
                    if state == states["LIFTING"]:
                        finished = moveAlongTrajectory(robot, position_trajectory, orientation_trajectory, start_step, i)
                        if finished:
                            event = events["OBJECT_LIFTED"]

                        
                        

                    #STEP 8: GENERATE TRAJECTORY TO TARGET
                    if state == states["GENERATING_TRAJECTORY"]:
                        target_position = np.array(config["world_settings"]["default_goal_pos"]) + np.array([-0.10, -0.10, 0.35])
                        waypoint_orientation = concatenate_quaternions(axis_angle_to_quaternion(axis0, angle0), axis_angle_to_quaternion([0,0,1], +np.pi/2))
                        target_orientation = concatenate_quaternions(waypoint_orientation, axis_angle_to_quaternion([1,0,0], +np.pi/2))
                        
                        bounds = [[-1.5, 1.5], [-1.5, 1.5], [0.5,2 ]]  # Define 3D workspace limit

                        rrt_star = RRTStar3D(start=robot.get_ee_pose()[0], goal=target_position, world_bounds=bounds)
                        
                        path = rrt_star.run(num_points=1000)
                        
                        # Visualize the trajectory in PyBullet
                        if len(path) > 10:
                            _, orientation_trajectory = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 1000)
                        
                            position_trajectory = path

                            for j in range(len(path) - 1):
                                p.addUserDebugLine(path[j], path[j+1], lineColorRGB=[1, 0, 0], lifeTime=0)
                            event = events["TRAJECTORY_GENERATED"]
                        start_step = i + 100

                    #STEP 9: MOVE TO TARGET AND DROP OBJECT
                    if state == states["MOVING_TO_TARGET"]:
                        finished = moveAlongTrajectory(robot, position_trajectory, orientation_trajectory, start_step, i)
                        if finished and i > start_step + 1100:
                            robot.open_gripper()
                            event = events["TARGET_REACHED"]

                    #After each step, updates the state machine
                    state = transitions[state][event]
                    event = events["NONE"]
                    print(f"[{i}] State: {state}")
                    print(f"[{i}] Event: {event}")
                ###########################################################'''
                       
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)

    
