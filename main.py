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

            ###[MC: 2025-02-15] Test of Jacobian IK-Controller ###
            ###########################################################
            target_position = [ 0.0, -0.6, 1.8] 
            axis0 = [0,1,0]
            angle0 =  np.pi
            axis1 = [0, 0, 1]
            angle1 = -np.pi/4
            target_orientation = concatenate_quaternions(axis_angle_to_quaternion(axis0, angle0), axis_angle_to_quaternion(axis1, angle1)) #...keeping default initial rotation in quaternion
            robot = sim.get_robot()

            positions1, orientations1 = interpolateLinearTrajectory( robot.get_ee_pose()[0], robot.get_ee_pose()[1], target_position, target_orientation, 1000)
            positions2, orientations2 = interpolateLinearTrajectory( target_position, target_orientation, [ 0.0, -0.6, 1.4], target_orientation, 500)
            ###########################################################
           
            for i in range(10000):
                sim.step()
                
                ###########################################################
                moveAlongTrajectory(robot, positions1, orientations1,  250, 1000, i)
                moveAlongTrajectory(robot, positions2, orientations2,  1300, 500, i)
                ###########################################################
                

                # for getting renders
                #[MC:2025-02-10] PERFORMANCE: change render FPS
                if i%10 == 0: #Only get renders every 10 steps (240/10 = 24fps on image processing)
                    #rgb, depth, seg = sim.get_ee_renders()
                    rgb, depth, seg = sim.get_static_renders()
                

                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
