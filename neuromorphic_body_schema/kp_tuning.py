"""
   This  code is meant to create a training loop for optimizing the mujoco model of icub for solving inverse kinematics.
   
   It will work by optimizing the kp values per joint in the xml file of the mujoco model.
   The training loop will use the mujoco simulator to simulate the robot and collect data for training.
    
    
    
"""


import copy
import logging
import math
import re
import threading
from collections import defaultdict

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from helpers.ed_cam import ICubEyes
from helpers.ed_prop import ICubProprioception
from helpers.ed_skin import ICubSkin
# from helpers.helpers import MODEL_PATH, DynamicGroupedSensors, init_POV
from helpers.ik_solver import Ik_solver
from helpers.robot_controller import check_joints, update_joint_positions
from mujoco import viewer

# from helpers.ik_solver_fede import qpos_from_site_pose

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


MODEL_PATH = "./neuromorphic_body_schema/models/icub_v2_full_body_kp_tuning.xml"



def init(data, model, range):
    """
    Resets the simulation to the given keyframe.

    Args:
        keyframe (np.ndarray): Joint positions to reset to (should match data.qpos shape).
        data (mujoco.MjData): The MuJoCo data object.
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        None
    """
    # let's set the initial joint positions and actuator controls
    for joint_name, position in range.items():
        try:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            rdm_joint_val = np.random.uniform(position["limit_low"], position["limit_high"])
            data.joint(joint_id).qpos[0] = rdm_joint_val
            data.actuator(joint_name).ctrl[0] = rdm_joint_val
        except IndexError:
            pass
            # logging.warning(f"Joint {joint_name} not found in the model.")


def find_ctrlrange(MODEL_PATH):
    """
    Find the ctrlrange for each joint in the model.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.

    Returns:
        dict: A dictionary mapping joint names to their ctrlrange values.
    """
    joint_limits = defaultdict(dict)
    keep_processing = True
    line_counter = 0
    with open(MODEL_PATH, 'r') as file:
        lines = file.readlines()
    
    while keep_processing:
        
        if "kp=" in lines[line_counter] :
        
            if "ctrllimited" in lines[line_counter] :
                line_counter += 1
                continue
                    
            part = lines[line_counter].split('name="')[1].split('"')[0]
            #kp = float(lines[line_counter].split('kp="')[1].split('"')[0])
            
            limit_low = float(lines[line_counter].split(str('ctrlrange="'))[1].split(' ')[0].split('"')[0])
            limit_high = float(lines[line_counter].split(str('ctrlrange="'))[1].split(' ')[1].split('"')[0])

            joint_limits[part]['limit_low'] = limit_low
            joint_limits[part]['limit_high'] = limit_high
            
            line_counter += 1

        else:
            line_counter += 1
            if line_counter >= len(lines):
                keep_processing = False
                
    return dict(joint_limits)



if __name__ == '__main__':
    
# 1. read lines xml file
    # now we can open the xml robot config file 

        
    keep_processing = True
    line_counter = 0
    joint_counter = 0 
    range_joints = find_ctrlrange(MODEL_PATH)
    with open(MODEL_PATH, 'r') as file:
        lines = file.readlines()
   
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    while keep_processing:
         with open("icub_v2_full_body_kp_tuning_after.xml", 'a') as output_file:
        
            if "kp=" in lines[line_counter] :
            
                if "ctrllimited" in lines[line_counter] :
                    output_file.write( lines[line_counter] )
                    line_counter += 1
                    continue
                # 2. start from line 0, read kp and ctrl values
                
                part = lines[line_counter].split('name="')[1].split('"')[0]
                kp = float(lines[line_counter].split('kp="')[1].split('"')[0])
                KP_ORIGINAL = kp

                # 3b. init joint randomly
                data.qpos.fill(0.0)
                init(model=model, data=data, range=range_joints)
                    # let's set the initial joint positions and actuator controls
                print("Model loaded")
                
                # 3c. init joint line 0 randomly
                target = np.random.uniform(
                    range_joints[part]['limit_low'], range_joints[part]['limit_high'])

                # 4. evolve for n_steps
                count = 0
                SIM_DURATION = 100 # seconds
                NUM_REPEATS = 10
                n_steps = int(SIM_DURATION / model.opt.timestep)
                # This is to accumulate datas for later plotting
                # position_data = np.zeros((len(range_joints), NUM_REPEATS, n_steps)) # repetitions, nb_steps
                rep=0
                finished = False
                while not finished or rep>1e4 or rep< NUM_REPEATS:
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, part)
                    
                    logging.info(f"Simulation repeat {rep + 1}/{NUM_REPEATS}, part: {part}, kp_bias: {-model.actuator_biasprm[joint_id][1]},  kp_gain:{model.actuator_gainprm[joint_id][0]}")
                    delta = np.zeros((n_steps, 1))  # Initialize delta history
                    # simulate
                    for i in range(n_steps):
                        pass
                        mujoco.mj_step(model, data)  # Step the simulation
                        # # viewer.sync()
                        # position_data[rep, :, i] = data.qpos[:8]  # Store joint positions
                        data.actuator(part).ctrl[0] = target
                        
                        # 5. trace delta
                        delta[i] = data.qpos[joint_id] - target

                    # 6. based on delta history change kp
                    sign_change = np.any(np.sign(delta[:-1]) != np.sign(delta[1:]))
                    if sign_change:
                        kp_sign_change = 0.1*np.max(np.abs(delta))
                        kp_drift_change = 0.0
                    elif kp_drift_change> 1e-8:
                        kp_sign_change = 0.0
                        kp_drift_change = 0.2*delta[-1]
                    else:
                        logging.info(f"Optimization errors: kp_sign_change: {kp_sign_change}, kp_drift_change: {kp_drift_change}")
                        finished = True
                                        
                    kp = kp + kp_drift_change - kp_sign_change
                    
                    model.actuator_biasprm[joint_id][1] = -kp
                    model.actuator_gainprm[joint_id][0] = kp
                    rep+=1

                    # 7. repeat for new kp values
                
                    # 8. do same from line 1 to end of file
                    
                new_line = lines[line_counter].split('kp="')[0] + str('kp="') + str(kp) +  lines[line_counter].split(str(KP_ORIGINAL))[1]
                output_file.write(new_line)
                line_counter+=1
                    
            else:
                output_file.write( lines[line_counter] ) 
                line_counter += 1


         if line_counter >= len(lines):
            keep_processing = False






    # # Load the MuJoCo model and create a simulation
    # model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    # data = mujoco.MjData(model)
 
    # # set model to 0.0 start position
    # data.qpos.fill(0.0)
    # # define a start position for the model
    # joint_init_pos = {
    #     'r_shoulder_roll': 0.6,
    #     'r_shoulder_pitch': -0.5,
    #     'r_shoulder_yaw': 0.0,
    #     'r_elbow': 1.1,
    #     'l_shoulder_roll': 0.6,
    #     'l_shoulder_pitch': -0.5,
    #     'l_shoulder_yaw': 0.0,
    #     'l_elbow': 1.1,
    # }

    # reset(model)

    # print("Model loaded")

    # # Set the time step duration to 0.001 seconds (1 milliseconds)
    # model.opt.timestep = 0.001  # sec

    # ############################
    # ### Start the simulation ###
    # ############################

    # sim_time = data.time

    # # Valid kinematic links shoulde be predefined
    # joint_names = ["l_shoulder_pitch", "l_shoulder_roll",
    #                 "l_shoulder_yaw", "l_elbow", "l_wrist_prosup"]

    # # randomized final position to reach for single joint
    # target_pos = []

    # count = 0
    # SIM_DURATION = 10 # seconds
    # NUM_REPEATS = 8
    # n_steps = int(SIM_DURATION / model.opt.timestep)
    # # This is to accumulate datas for later plotting
    # position_data = np.zeros((8, 8, n_steps)) # 8 repetitions, 8 joints, nb_steps
   
    # for rep in range(NUM_REPEATS):
    #     # print(sim_time)
    #     print("here")
    #     if NUM_REPEATS > 1:
    #         logging.info(f"Resetting simulation for repeat {rep + 1}/{NUM_REPEATS}")
    #         reset(data, model)
    #     # simulate
    #     for i in range(n_steps):
    #         mujoco.mj_step(model, data)  # Step the simulation
    #         # viewer.sync()
    #         position_data[rep, :, i] = data.qpos[:8]  # Store joint positions

    #         update_joint_positions(data, target_pos)

    #         pass






#  with open(mujoco_model_out, 'w') as file:
#         file.write(lines)
#     pass

#     # write report to txt file
#     with open(f'./report_including_taxels.txt', 'w') as file:
#         file.write(f'Part name: Nb of taxels\n')
#         for part_to_add, taxels_to_add in zip(parts_to_add, taxel_ids_to_add):
#             file.write(f'{part_to_add}: {len(taxels_to_add)}\n')




















