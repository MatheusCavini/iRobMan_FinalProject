import numpy as np
import pybullet as p

#[MC:2025-02-15] Some functions to test how well the Jacobian Controller performs in interpolated trajectories
###########################################################
def interpolateLinearTrajectory(initial_position, initial_orientation, end_position, end_orientation, n_points):
    positions = []
    orientations = []
    for i in range(n_points):
        position = ((np.array(end_position) - np.array(initial_position))/n_points * i) + np.array(initial_position)
        orientation = np.array(p.getQuaternionSlerp(initial_orientation, end_orientation, i/n_points))

        positions.append(position)
        orientations.append(orientation) 

    return positions, orientations



def moveAlongTrajectory(robot, positions, orientations, step_start, steps_duration, current_step):
    if current_step > step_start and current_step < step_start+steps_duration:
        robot.move_to_pose(positions[current_step-step_start], orientations[current_step-step_start])


###########################################################