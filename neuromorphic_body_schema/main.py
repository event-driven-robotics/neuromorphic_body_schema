"""
main.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description: 
This script initializes and runs a neuromorphic simulation of the iCub robot using MuJoCo. 
It integrates event-based camera, proprioception, and skin sensors, and provides visualization 
options for each sensory modality.

Modules:
- ICubEyes: Simulates event-based camera functionality.
- ICubProprioception: Simulates proprioceptive spiking events.
- ICubSkin: Simulates tactile skin events.
- DynamicGroupedSensors: Provides dynamic access to grouped sensor data.
- update_joint_positions: Updates joint positions in the MuJoCo model.
- init_POV: Configures the viewer's point of view.

Usage:
Run this script to start the simulation and visualize the sensory data.

"""


import copy
import logging
import math
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
from helpers.ed_cam import ICubEyes
from helpers.ed_prop import ICubProprioception
from helpers.ed_skin import ICubSkin
from helpers.helpers import MODEL_PATH, DynamicGroupedSensors, init_POV
from helpers.ik_solver import Ik_solver
from helpers.robot_controller import check_joints, update_joint_positions
from mujoco import viewer

# from helpers.ik_solver_fede import qpos_from_site_pose

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

VISUALIZE_CAMERA_FEED = False
VISUALIZE_ED_CAMERA_FEED = False
VISUALIZE_SKIN = False
VISUALIZE_PROPRIOCEPTION_FEED = False


def reset(keyframe, data, model):
    """
    Resets the simulation to the given keyframe.

    Args:
        keyframe (np.ndarray): Joint positions to reset to (should match data.qpos shape).
        data (mujoco.MjData): The MuJoCo data object.
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        None
    """
    assert keyframe.shape == data.qpos.shape, "Keyframe shape does not match qpos shape."
    data.qpos[:] = keyframe
    mujoco.mj_forward(model, data)


def ik_caculation(ik_solver, target_pos, target_ori, joint_names):
    """
    Runs IK and returns a dictionary mapping joint names to solved positions.

    Args:
        ik_solver: An instance of the IK solver.
        target_pos (np.ndarray): Target end-effector position.
        target_ori (np.ndarray): Target end-effector orientation.
        joint_names (list): List of joint names in the same order as the solver output.

    Returns:
        dict or None: {joint_name: position, ...} if successful, None otherwise.
    """

    try:
        q_arm = ik_solver.ik_step(target_pos, target_ori)
        joint_pose = dict(zip(joint_names, q_arm))
        logging.info(f"Solution found: {joint_pose}")
        return joint_pose
    except ValueError as e:
        logging.error(f"IK failed: {e}")
        return None


if __name__ == '__main__':
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    r_eye_camera_name = 'r_eye_camera'
    l_eye_camera_name = 'l_eye_camera'

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    # set model to 0.0 start position
    data.qpos.fill(0.0)
    # define a start position for the model
    joint_init_pos = {
        'r_shoulder_roll': 0.6,
        'r_shoulder_pitch': -0.5,
        'r_shoulder_yaw': 0.0,
        'r_elbow': 1.1,
        'l_shoulder_roll': 0.6,
        'l_shoulder_pitch': -0.5,
        'l_shoulder_yaw': 0.0,
        'l_elbow': 1.1,
    }
    # let's set the initial joint positions and actuator controls
    for joint_name, position in joint_init_pos.items():
        try:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.joint(joint_id).qpos[0] = position
            data.actuator(joint_name).ctrl[0] = position
        except ValueError:
            logging.warning(f"Joint {joint_name} not found in the model.")
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec

    # prepare the mapping from skin to body parts
    names_list = model.names.decode('utf-8').split('\x00')
    sensor_info = [x for x in names_list if "taxel" in x]

    # Extract base names and group sensor addresses by base names
    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r'_\d+$', '', name)
        grouped_sensors[base_name].append(adr)

    if DEBUG:
        for key, value in grouped_sensors.items():
            print(key, len(value))

    dynamic_grouped_sensors = DynamicGroupedSensors(data, grouped_sensors)

    joint_dict_prop = {
        'r_shoulder_roll': {
            'position_max_freq': 1000,  # Hz
            'velocity_max_freq': 1000,
            'load_max_freq': 1000,
            'limits_max_freq': 1000,
        },
        'l_shoulder_roll': {
            'position_max_freq': 1000,
            'velocity_max_freq': 1000,
            'load_max_freq': 1000,
            'limits_max_freq': 1000,
        },
        # 'r_pinky': {
        #     'position_max_freq': 1000,  # Hz
        #     'velocity_max_freq': 1000,
        #     'load_max_freq': 1000,
        #     'limits_max_freq': 1000,
        # },
        # 'l_pinky': {
        #     'position_max_freq': 1000,
        #     'velocity_max_freq': 1000,
        #     'load_max_freq': 1000,
        #     'limits_max_freq': 1000,
        # },
    }

    # # Define parameters for the sine wave
    # frequencies = [1.0, 1.0, 1.0, 1.0, 1.0]
    # min_max_pos = np.zeros((len(joints), 2))
    # for i, joint in enumerate(joints):
    #     min_max_pos[i] = model.actuator(joint).ctrlrange
    #     # to ensure we move close to the contact position
    #     if 'pinky' in joint:
    #         min_max_pos[i][0] = min_max_pos[i][1]*0.98

    ############################
    ### Start the simulation ###
    ############################

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Disable the left and right panels
        # viewer.options.gui_left = False
        # viewer.options.gui_right = False

        init_POV(viewer)

        sim_time = data.time

        skin_object = ICubSkin(
            sim_time, dynamic_grouped_sensors, show_skin=VISUALIZE_SKIN, DEBUG=DEBUG)
        r_eye_camera_object = ICubEyes(sim_time, model, data, r_eye_camera_name,
                                       show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        l_eye_camera_object = ICubEyes(sim_time, model, data, l_eye_camera_name,
                                       show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        proprioception_object = ICubProprioception(
            model, joint_dict_prop, show_proprioception=VISUALIZE_PROPRIOCEPTION_FEED, DEBUG=DEBUG)

        # Valid kinematic links shoulde be predefined
        joint_names = ["l_shoulder_pitch", "l_shoulder_roll",
                       "l_shoulder_yaw", "l_elbow", "l_wrist_prosup"]
        end_effector_name = "l_forearm"

        # IK with Quaternion seems more robust for Icub
        # should copy the data for forward knimeatics, otherwise the ik will update the model directly
        data_copy = copy.deepcopy(data)
        ik_solver = Ik_solver(model, data_copy, joint_names,
                              end_effector_name, "quat")

        # Sequential Reaching task

        target_pos = [[-0.09736548, -0.20864483, 0.94718375],
                      [-0.13067764, -0.25348467,  1.12211061]]
        target_ori = [[-0.35741511, 0.26772824, 0.02986113, 0.89425071],
                      [-0.18286161, -0.0885009,   0.46619002, 0.8610436]]

        caculated, reached, finished = False, False, False

        count = 0
        while viewer.is_running():
            # print(sim_time)
            mujoco.mj_step(model, data)  # Step the simulation
            viewer.sync()
            # sim_time_ns = data.time*1E9  # ns
            # new_joint_pos = {}
            # for (min_max, frequency, joint) in zip(min_max_pos, frequencies, joints):
            #     joint_position = min_max[0] + (min_max[1] - min_max[0]) * 0.5 * (
            #         1 + math.sin(2 * math.pi * frequency * data.time))
            #     new_joint_pos[joint] = joint_position
            #     # Update joint positions
            #     if DEBUG:
            #         print(joint, joint_position)
            # update_joint_positions(data, {joint: joint_position})

            r_eye_cam_events = r_eye_camera_object.update_camera(
                data.time*1E9)  # expects ns
            l_eye_cam_events = l_eye_camera_object.update_camera(
                data.time*1E9)  # expects ns

            skin_events = skin_object.update_skin(data.time*1E9)  # expects ns

            proprioception_events = proprioception_object.update_proprioception(
                time=data.time, data=data)  # expects seconds

            if count >= len(target_pos):
                finished = True
            if not finished:
                if not caculated:
                    q_arm = ik_caculation(
                        ik_solver, target_pos[count], target_ori[count], joint_names)
                    caculated = True
                    if not q_arm:
                        finished = True
                        logging.info(
                            f"Solution not found, task terminated at {count}th goal")
                        continue

                 # seems like the mujoco can not achieve the joints in one loop, so keep checking and control the joints
                if caculated and not check_joints(data, q_arm):
                    # currently PD controller for the joints
                    update_joint_positions(data, q_arm)
                else:
                    logging.info("Goal reached")
                    caculated = False
                    count += 1

            pass
