"""
    This is a code to test changes in the icub xml file leading to a complete model where the body parts do not trepass each other. 
    
    Since the body parts which could trepass each other, for the model's geometry, are basically just the hands, I will focus on creating the model for the hands. If further modelling 
    is going to be needed, that is a problem for future me (or future someone else).
    
    Miriam Barborini, 2025-09-05 

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


MODEL_PATH = "./neuromorphic_body_schema/models/icub_v2_full_body_kp_tuning_contact_tests.xml"


def main():

    # Load the MuJoCo model and create a simulation
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    # set model to 0.0 start position
    data.qpos.fill(0.0)
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec


    with mujoco.viewer.launch_passive(model, data) as viewer:

        init_POV(viewer)

        sim_time = data.time

        while viewer.is_running():
            mujoco.mj_step(model, data)  # Step the simulation
            viewer.sync()

            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

 
 
 
 
 
 
 
 
 
 
 
 
 
    
    
if __name__ == "__main__":
    main() 
    