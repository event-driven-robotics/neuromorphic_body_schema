"""
MuJoCo Skin Patch Sweep Test

This script launches a MuJoCo simulation, injects synthetic tactile values directly into mujoco.data.sensordata for a specified skin patch, and sweeps through all taxels with a ramp-and-hold pattern. The ICubSkin visualization is used to cross-check the active taxel.

Usage:
- Set PATCH_NAME to the desired skin patch (e.g., 'r_forearm').
- Run the script. It will sweep through all taxels, ramping up, holding, and ramping down the value for each.
- Observe the skin visualization for correctness.

"""
import sys
from pathlib import Path

# Get the project root directory (parent of neuromorphic_body_schema)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import re
import threading
import time
from collections import defaultdict

import mujoco
import numpy as np
from mujoco import viewer

from neuromorphic_body_schema.helpers.ed_skin import ICubSkin
from neuromorphic_body_schema.helpers.helpers import (KEY_MAPPING, MODEL_PATH,
                                                      DynamicGroupedSensors,
                                                      init_POV)

# possible skin patches: 
# ["right_leg_upper", "left_leg_upper", "right_leg_lower", "left_leg_lower", "right_hand", "torso", "right_forearm_V2", "left_forearm_V2", "right_arm", "left_arm", "left_hand"]

PATCH_NAME = 'right_hand'  # Change to desired patch
RAMP_STEPS = 100
HOLD_STEPS = 50
RAMP_MAX = 10.0
SKIN_MODE = 'frame_based'  # or 'frame_based'
VISUALIZE_SKIN_FEED = True
VISUALIZE_ED_SKIN_FEED = False

# MuJoCo is a C-extension module and some symbols are not visible to static analyzers.
# Resolve once via getattr so runtime behavior stays identical while Pylance can type-check calls.
MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mj_step = getattr(mujoco, "mj_step")


def update_sim():
    # print(sim_time)
    mj_step(model, data)  # Step the simulation
    sim_viewer.sync()


if __name__ == "__main__":
    viewer_closed_event = threading.Event()

    # Load the MuJoCo model and create a simulation
    model = MjModel.from_xml_path(MODEL_PATH)
    data = MjData(model)
    # set model to 0.0 start position
    data.qpos.fill(0.0)
    # define a start position for the model
    joint_init_pos = {
        "r_shoulder_roll": 0.6,
        "r_shoulder_pitch": -0.5,
        "r_shoulder_yaw": 0.0,
        "r_elbow": 1.1,
        "l_shoulder_roll": 0.6,
        "l_shoulder_pitch": -0.5,
        "l_shoulder_yaw": 0.0,
        "l_elbow": 1.1,
    }
    # let's set the initial joint positions and actuator controls
    for joint_name, position in joint_init_pos.items():
        try:
            joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
            data.joint(joint_id).qpos[0] = position
            data.actuator(joint_name).ctrl[0] = position
        except ValueError:
            print(f"Joint {joint_name} not found in the model.")
    print("Model loaded")

    # Set the time step duration to 0.001 seconds (1 milliseconds)
    model.opt.timestep = 0.001  # sec

    # Prepare sensor grouping
    names_list = model.names.decode("utf-8").split("\x00")
    sensor_info = [x for x in names_list if "taxel" in x]
    grouped_sensors = defaultdict(list)
    for adr, name in enumerate(sensor_info):
        base_name = re.sub(r"_\d+$", "", name)
        grouped_sensors[base_name].append(adr)

    dynamic_grouped_sensors = DynamicGroupedSensors(data, grouped_sensors)

    patch_key = KEY_MAPPING[PATCH_NAME]
    if isinstance(patch_key, list):
        patch_addrs = []
        for key in patch_key:
            patch_addrs.extend(grouped_sensors[key])
    else:
        patch_addrs = grouped_sensors[patch_key]

    # Launch MuJoCo viewer and ICubSkin
    with viewer.launch_passive(model, data) as sim_viewer:
        init_POV(sim_viewer)
        sim_time = data.time
        skin_object = ICubSkin(
            sim_time,
            dynamic_grouped_sensors,
            skin=PATCH_NAME,
            skin_mode=SKIN_MODE,
            show_raw_feed=VISUALIZE_SKIN_FEED,
            show_ed_feed=VISUALIZE_ED_SKIN_FEED,
            DEBUG=False
        )
        # Sweep state machine
        sweep_states = []
        for taxel_idx, addr in enumerate(patch_addrs):
            sweep_states.append({
                'addr': addr,
                'phase': 'ramp_up',
                'step': 0
            })

        current_taxel = 0
        while sim_viewer.is_running() and current_taxel < len(sweep_states):
            state = sweep_states[current_taxel]
            addr = state['addr']
            phase = state['phase']
            step = state['step']
            print(f"Sweeping taxel {current_taxel+1}/{len(patch_addrs)}, phase: {phase}, step: {step}")
            if phase == 'ramp_up':
                taxel_val = step * RAMP_MAX / RAMP_STEPS
                data.sensordata[addr] = taxel_val
                skin_object.update_skin(data.time * 1e9)
                update_sim()
                state['step'] += 1
                if state['step'] >= RAMP_STEPS:
                    state['phase'] = 'hold'
                    state['step'] = 0
            elif phase == 'hold':
                taxel_val = RAMP_MAX
                data.sensordata[addr] = taxel_val
                skin_object.update_skin(data.time * 1e9)
                update_sim()
                state['step'] += 1
                if state['step'] >= HOLD_STEPS:
                    state['phase'] = 'ramp_down'
                    state['step'] = 0
            elif phase == 'ramp_down':
                taxel_val = RAMP_MAX - step * RAMP_MAX / RAMP_STEPS
                data.sensordata[addr] = taxel_val
                skin_object.update_skin(data.time * 1e9)
                update_sim()
                state['step'] += 1
                if state['step'] >= RAMP_STEPS:
                    state['phase'] = 'reset'
                    state['step'] = 0
            elif phase == 'reset':
                data.sensordata[addr] = 0.0
                skin_object.update_skin(data.time * 1e9)
                update_sim()
                current_taxel += 1
            # Optionally print progress
            # print(f"Taxel {current_taxel}, phase {phase}, step {step}")

        print("Sweep complete. Cross-check GUI for active taxel.")
