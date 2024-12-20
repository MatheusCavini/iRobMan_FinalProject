from typing import Dict, Any, Optional

import numpy as np
import pybullet as p
import pybullet_data

from src.robot import Robot
from src.objects import Obstacle, Table, Box, YCBObject, Goal
from src.utils import pb_image_to_numpy


class Simulation:
    """Simulation Class.

    The class initializes a franka robot with static and moving
    obstacles on a large table.

    Args:
        exp_settings: Dictionary containing experiment settings.
        spawn_object: Which object to spawn (cube, or YCB object name).
        seed: Random seed for reproducibility.
    """
    def __init__(self,
                 exp_settings: Dict[str, Any],
                 spawn_object: str = "cube",
                 seed: int = 42):
        # settings here
        sim_settings = exp_settings["world_settings"]
        self.gravity = sim_settings["gravity"]
        self.obstacles_flag = sim_settings["turn_on_obstacles"]
        self.timestep = 1.0 / sim_settings["timestep_freq"]
        self.mode = sim_settings["mode"]
        self.gui_mode = p.GUI if sim_settings["mode"] > 0 else p.DIRECT
        self.render_mode = p.ER_TINY_RENDERER
        self.target_object = spawn_object
        self.rng = np.random.RandomState(seed)

        # camera general
        camera_settings = sim_settings["camera"]
        self.width = camera_settings["width"]
        self.height = camera_settings["height"]
        self.cam_render_flag = camera_settings["cam_render_flag"]
        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            camera_settings["fov"],
            aspect,
            camera_settings["near"],
            camera_settings["far"])

        # ee camera
        self.ee_cam_offset = np.array(camera_settings["ee_cam_offset"])
        self.ee_cam_orientation = np.array(
            camera_settings["ee_cam_orientation"])

        # stat cam
        self.stat_cam_pos = camera_settings["stat_cam_pos"]
        self.stat_cam_target_pos = camera_settings["stat_cam_target_pos"]
        self.stat_viewMat = p.computeViewMatrix(
            self.stat_cam_pos,
            self.stat_cam_target_pos,
            cameraUpVector=[0, 0, 1])

        # starting global_settings
        p.connect(self.gui_mode)
        p.setTimeStep(self.timestep)
        # p.setRealTimeSimulation(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # starting the simulation
        self.exp_settings = exp_settings

    def reset(self, new_obj_name: Optional[str] = None):
        if new_obj_name:
            self.target_object = new_obj_name
        p.resetSimulation()
        p.setGravity(0, 0, self.gravity)
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1)
        self._load_objects(self.exp_settings)

    @property
    def get_ground_tuth_position_object(self):
        return self.initial_position

    def close(self):
        p.disconnect()

    def check_goal(self):
        min_lim, max_lim = self.goal._get_goal_lims()
        obj_ids = p.getOverlappingObjects(min_lim, max_lim)
        obj_idx = [ob[0] for ob in obj_ids]
        if self.object.id in obj_idx:
            return True
        return False

    def check_goal_obj_pos(self, estimated_pos: np.array):
        full_obj_pos = np.hstack([self.object.pos, self.object.ori])
        assert estimated_pos.shape == full_obj_pos.shape
        return np.linalg.norm(full_obj_pos - estimated_pos, 2)

    def check_obstacle_position(self, estimated_pos: np.array):
        assert estimated_pos.shape == (2, 3)
        if not self.obstacles_flag:
            return 0
        diff = 0
        for idx, obstacle in enumerate(self.obstacles):
            pos = np.asarray(obstacle._get_pos())
            diff += np.linalg.norm(pos - estimated_pos[idx])
        return diff

    def _load_objects(self, exp_settings: Dict[str, Any]):

        sim_settings = exp_settings["world_settings"]
        robot_settings = exp_settings["robot_settings"]
        p.loadURDF(sim_settings["base_urdf"])
        self.wall = p.loadURDF(sim_settings["background_urdf"],
                               basePosition=[-0.6, 0., 0.],
                               baseOrientation=p.getQuaternionFromEuler(
                                   [0, np.pi/2, 0]))
        self.table = Table()
        self.robot = Robot(urdf=robot_settings["urdf"],
                           init_position=robot_settings["default_init_pos"],
                           orientation=robot_settings["default_init_ori"],
                           arm_index=robot_settings["arm_idx"],
                           gripper_index=robot_settings["gripper_idx"],
                           ee_index=robot_settings["ee_idx"],
                           arm_default=robot_settings["default_arm"],
                           table_scaling=robot_settings["table_scaling"])

        self.initial_position = sim_settings["default_obj_pos"]
        self.obj_jitter = self.rng.uniform([-0.2, -0.2, 0], [0.2, 0.2, 0])
        self.initial_position = self.initial_position + self.obj_jitter
        self.goal = Goal(position=sim_settings["default_goal_pos"])

        if self.target_object == "cube":
            self.object = Box(position=self.initial_position)
        else:
            self.object = YCBObject(obj_name=self.target_object,
                                    position=self.initial_position)
        if self.obstacles_flag:
            self._add_obstacles()

    def _add_obstacles(self):
        self.obstacles = []
        scales = [0.3, 0.2]
        planes = [np.array([[0.4, 0.7, 1.5], [0.9, 1.0, 2.0]]),
                  np.array([[0.4, 0.5, 1.5], [0.9, 0.7, 2.0]])]

        for i, (plane, scale) in enumerate(zip(planes, scales)):
            self.obstacles.append(
                Obstacle(rng=self.rng, plane=plane,
                         scale=scale, flip_index=i))

    def stop_obstacles(self):
        # for developing
        if self.obstacles_flag:
            for obstacle in self.obstacles:
                obstacle.stop

    def get_ee_renders(self):
        """Get end-effector camera renders. y-axis is up."""
        ee_pos, ee_rot = self.robot.get_ee_pose()
        rot_matrix = p.getMatrixFromQuaternion(ee_rot)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(
            ee_pos, ee_pos + 0.1 * camera_vector, up_vector)
        # View matrix

        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=self.render_mode)

        return pb_image_to_numpy(rgb, depth, seg, self.width, self.height)

    def get_static_renders(self):
        """
        Get static camera renders.
        """
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.width, height=self.height,
            viewMatrix=self.stat_viewMat,
            projectionMatrix=self.projection_matrix,
            renderer=self.render_mode)

        return pb_image_to_numpy(rgb, depth, seg, self.width, self.height)

    def step(self):
        collided_flag = False

        if p.getContactPoints(self.robot.id, self.wall):
            print("ERROR! Robot in Collision with wall ")
            collided_flag = True

        if self.obstacles_flag:
            for obstacle in self.obstacles:
                if p.getContactPoints(self.robot.id, obstacle.id):
                    print("ERROR! Robot in Collision with obstacles.")
                    collided_flag = True
                    break
                obstacle.move()

        if self.cam_render_flag:
            if self.mode == 1:
                self.get_static_renders()
            elif self.mode == 2:
                self.get_ee_renders()

        if collided_flag:
            print("Ending Simulation")
            self.close()
            return
        p.stepSimulation()

    def get_robot(self):
        return self.robot
