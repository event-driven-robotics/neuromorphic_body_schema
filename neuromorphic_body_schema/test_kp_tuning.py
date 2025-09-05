"""
    This is a script to test the kp values obtained from the kp_tuning.py file.
    
    It will run multiple simulations of the joints. It will check that the encoder value and the joint value to be reached are the same along the path.
    It will also check that no overshootin/undershooting happen in the integration of the component.
    
"""




import logging
from collections import defaultdict
import os
import mujoco
from mujoco import viewer
import numpy as np
import matplotlib.pyplot as plt
from kp_tuning import init, find_ctrlrange
from helpers.helpers import init_POV
from time import sleep

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


MODEL_PATH = "./neuromorphic_body_schema/models/icub_v2_full_body_kp_tuning_after_with_tuning.xml"


if __name__ == '__main__':
    
# 1. read lines xml file
    # now we can open the xml robot config file 
        
    keep_processing = True
    line_counter = 0
    joint_counter = 0 
    SIM_DURATION = 1.0
    range_joints = find_ctrlrange(MODEL_PATH)
    
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    print("Model loaded")
    
    with open(MODEL_PATH, 'r') as file:
        lines = file.readlines()

    while keep_processing:
        
        if "kp=" in lines[line_counter] :
            if "ctrllimited" in lines[line_counter] :
                line_counter += 1
                continue
            
            part = lines[line_counter].split('name="')[1].split('"')[0]            
            data.qpos.fill(0.0)
            n_steps = 1000 # int(SIM_DURATION / model.opt.timestep)

            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, part)
            
            position_data = []
            actuator_data = []
            
            for j in range(5):
            
                init(model=model, data=data, range=range_joints)
                position_data_temp = []
                actuator_data_temp = []
                target = np.random.uniform(
                range_joints[part]['limit_low'], range_joints[part]['limit_high'])
                
                # KP_BIAS = -model.actuator_biasprm[joint_id][1]
                # KP_GAIN = model.actuator_gainprm[joint_id][0]
                
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    init_POV(viewer)
                    
                    while viewer.is_running():
                        sleep(1.0)
                        for i in range(n_steps):
                            mujoco.mj_step(model, data)  # Step the simulation
                            viewer.sync()
                            position_data_temp.append(data.qpos[joint_id])  # Store joint positions
                            actuator_data_temp.append(data.actuator(part).ctrl[0])
                            data.actuator(part).ctrl[0] = target
                            
                        viewer.close()        
                # # data.qpos[joint_id]
                position_data.append(position_data_temp)
                actuator_data.append(actuator_data_temp)
                logging.info(f"Simulation {j+1}/5 for part {part} completed.")
                mujoco.mj_resetData(model, data) 
            
            colors = plt.cm.jet(np.linspace(0,1,5)) #this gets the colormap as an array of colours
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [5, 2]})
            fig.suptitle(f"Joint: {part}")
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Joint position (rad)")
            for i in range(5):
                ax1.plot(position_data[i], label=f"Run {i+1}", c=colors[i])
                ax1.plot(actuator_data[i], label=f"Actuator {i+1}", c=colors[i], linestyle='--')
            ax1.set_ylim(range_joints[part]['limit_low'], range_joints[part]['limit_high'])
            ax1.grid()
            ax1.legend()
            
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Error (rad)")
            for i in range(5):
                error = np.array(actuator_data[i]) - np.array(position_data[i])
                ax2.plot(error, label=f"Error Run {i+1}", c=colors[i])
            ax2.legend()
            ax2.grid()
            # plt.savefig(f"./results/kp_tuning_test_3/{part}_position.png")   
            plt.close(fig)  
                   
            line_counter+=1
                
        else:
            line_counter += 1


        if line_counter >= len(lines):
            keep_processing = False

    
        
        
        
        
        
        
        
        
        





