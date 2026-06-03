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

# Add parent directory to path for direct script execution
import sys
from pathlib import Path

# Get the project root directory (parent of neuromorphic_body_schema)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import copy
import logging
import math
import re
import threading
from collections import defaultdict
from neuromorphic_body_schema.helpers.ed_cam import ICubEyes
import mujoco
import numpy as np

from mujoco import viewer

MODEL_PATH = "neuromorphic_body_schema/models/scene.xml"

DEBUG = False  # use to visualize the triangles
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# MuJoCo is a C-extension module and some symbols are not visible to static analyzers.
# Resolve once via getattr so runtime behavior stays identical while Pylance can type-check calls.
MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_step = getattr(mujoco, "mj_step")

CAMERA_MODE = "frame_based"  # "event_driven" or "frame_based"
CAM_TO_USE = "all"  # "left", "right", or "all"
VISUALIZE_CAMERA_FEED = True
# setting this to true also activates the event-based camera if not already activated, since we need it to show the feed
if CAMERA_MODE == "event_driven":
    VISUALIZE_ED_CAMERA_FEED = True
else:
    VISUALIZE_ED_CAMERA_FEED = False
    
if __name__ == "__main__":
    #############################
    ### setting everything up ###
    #############################

    viewer_closed_event = threading.Event()

    # Load the MuJoCo model and create a simulation
    model = MjModel.from_xml_path(MODEL_PATH)
    data = MjData(model)

    with viewer.launch_passive(model, data) as sim_viewer:
        sim_time = data.time
        
        eye_camera_object = ICubEyes(sim_time, model, data, eye=CAM_TO_USE, camera_mode=CAMERA_MODE, show_raw_feed=VISUALIZE_CAMERA_FEED, show_ed_feed=VISUALIZE_ED_CAMERA_FEED, DEBUG=DEBUG)
        
        while sim_viewer.is_running():
            # print(sim_time)
            mj_step(model, data)  # Step the simulation
            sim_viewer.sync()
            
            eye_cam_events = eye_camera_object.update_camera(
                data.time * 1e9
            )  # expects ns; returns dict {"left": events, "right": events}
