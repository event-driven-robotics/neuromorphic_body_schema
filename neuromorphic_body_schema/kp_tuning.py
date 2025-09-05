"""
   This  code is meant to create a training loop for optimizing the mujoco model of icub for solving inverse kinematics.
   
   It will work by optimizing the kp values per joint in the xml file of the mujoco model.
   The training loop will use the mujoco simulator to simulate the robot and collect data for training.
    
    
    
"""


import logging
from collections import defaultdict
import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt


DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


MODEL_PATH = "./neuromorphic_body_schema/models/icub_v2_full_body_kp_tuning_after_with_tuning.xml"



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





def find_kp(model, data, kp_original):
    
    kp = kp_original
    SIM_DURATION = 100 # seconds
    NUM_REPEATS = 100 # find at least some horrible conditions to test the kp tuning on
    kp_drift_change = 0.0
    kp_sign_change = 0.0

    data.qpos.fill(0.0)
    print("Model loaded")
    
    # 4. evolve for n_steps
    n_steps = int(SIM_DURATION / model.opt.timestep)
    # This is to accumulate datas for later plotting
    # position_data = np.zeros((len(range_joints), NUM_REPEATS, n_steps)) # repetitions, nb_steps
    rep=0
    finished = False
    while rep>1e4 or rep< NUM_REPEATS:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, part)
        
        if joint_id >68:
            return kp_original
        else:
            init(model=model, data=data, range=range_joints)
            target = np.random.uniform(
            range_joints[part]['limit_low'], range_joints[part]['limit_high'])
        
            
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
                kp_sign_change = 0.5*np.max(np.abs(delta))
                kp_drift_change = 0.0
            elif kp_drift_change> 1e-8:
                kp_sign_change = 0.0
                kp_drift_change = 0.2*delta[-1]
            else:
                kp_sign_change = 0.0
                kp_drift_change = 0.0
                logging.info(f"Optimization errors: kp_sign_change: {kp_sign_change}, kp_drift_change: {kp_drift_change}")
                                
            kp = kp + kp_drift_change - kp_sign_change
            
            model.actuator_biasprm[joint_id][1] = -kp
            model.actuator_gainprm[joint_id][0] = kp
            rep+=1
                        
    return kp




if __name__ == '__main__':
    
# 1. read lines xml file
    # now we can open the xml robot config file 
        
    keep_processing = True
    line_counter = 0
    joint_counter = 0 
    range_joints = find_ctrlrange(MODEL_PATH)
    with open(MODEL_PATH, 'r') as file:
        lines = file.readlines()

    while keep_processing:
         with open("icub_v2_full_body_kp_tuning_after_with_tuning_20250904.xml", 'a') as output_file:
             
            model = mujoco.MjModel.from_xml_path(MODEL_PATH)
            data = mujoco.MjData(model)
        
            if "kp=" in lines[line_counter] :
            
                if "ctrllimited" in lines[line_counter] :
                    #output_file.write( lines[line_counter] )
                    line_counter += 1
                    continue
                # 2. start from line 0, read kp and ctrl values
                
                part = lines[line_counter].split('name="')[1].split('"')[0]
                kp = float(lines[line_counter].split('kp="')[1].split('"')[0])
                KP_ORIGINAL = kp

                kp = find_kp(model, data,KP_ORIGINAL)
                    
                new_line = lines[line_counter].split('kp="')[0] + str('kp="') + str(kp) +  lines[line_counter].split(str(KP_ORIGINAL))[1]
                output_file.write(new_line)
                line_counter+=1
                    
            else:
                output_file.write( lines[line_counter] ) 
                line_counter += 1


         if line_counter >= len(lines):
            keep_processing = False




