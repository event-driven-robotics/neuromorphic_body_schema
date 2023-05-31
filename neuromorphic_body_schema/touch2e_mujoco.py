import mujoco
from mujoco import viewer
# import cv2
import threading
import logging
import numpy as np
# import time

full_body = True
DEBUG = False

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def detect_contact(data, taxelDict):
    ''' Loop to monitor contact information'''
    while True:
        # TODO check if only ID is sufficient or ID to name is needed
        sensordata = data.sensordata
        # only output info when non-zero contact detected
        if np.sum(sensordata) > 0.0:
            taxels = np.where(sensordata != 0.0)[0]  # get non-zero taxels IDs
            # taxelDict.update(
            #     {data.sensor(taxel).name: sensordata[taxel] for taxel in taxels})
            for taxel in taxels:
                taxelDict[data.sensor(taxel).name].append(data.time, sensordata[taxel])

            if DEBUG:
                print('End of contact detection for this cycle.\n\n')


taxelDict = {}
if full_body:
    # torso
    for patchID in range(3):
        # example taxel numbers
        for taxelID in range(12):
            # taxelDict[("TORSO", "TORSO", patchID, taxelID)] = 0.0  # old
            # time and value
            taxelDict[f"TORSO__TORSO__{patchID}__{taxelID}"] = []

    # left arm (each fingertip is concidered a patch and all palm is a single patch)
    # add fingertips
    for patchID in range(5):
        for taxelID in range(12):
            taxelDict[f"LEFT_ARM__HAND__{patchID}__{taxelID}"] = []
    # add palm
    for taxelID in range(48):
        taxelDict[f"LEFT_ARM__HAND__{5}__{taxelID}"] = []
    # add forearm
    for patchID in range(2):
        for taxelID in range(12):
            taxelDict[f"LEFT_ARM__FOREARM__{patchID}__{taxelID}"] = []
    # add forearm
    for patchID in range(2):
        for taxelID in range(12):
            taxelDict[f"LEFT_ARM__UPPER_ARM_{patchID}__{taxelID}"] = []

    # add left leg
    for taxelID in range(4):
        taxelDict[f"LEFT_LEG__FOOT__{0}__{taxelID}"] = []
    # add upper leg
    for patchID in range(3):
        for taxelID in range(12):
            taxelDict[f"LEFT_LEG__UPPER_LEG__{patchID}__{taxelID}"] = []
    # add lower leg
    for patchID in range(3):
        for taxelID in range(12):
            taxelDict[f"LEFT_LEG__LOWER_LEG__{patchID}__{taxelID}"] = []

    # add forearm
    for patchID in range(2):
        for taxelID in range(12):
            taxelDict[f"RIGHT_ARM__FOREARM__{patchID}__{taxelID}"] = []
    # add forearm
    for patchID in range(2):
        for taxelID in range(12):
            taxelDict[f"RIGHT_ARM__UPPER_ARM_{patchID}__{taxelID}"] = []

    # add right leg
    for taxelID in range(4):
        taxelDict[f"RIGHT_LEG__FOOT__{0}__{taxelID}"] = []
    # add upper leg
    for patchID in range(3):
        for taxelID in range(12):
            taxelDict[f"RIGHT_LEG__UPPER_LEG__{patchID}__{taxelID}"] = []
    # add lower leg
    for patchID in range(3):
        for taxelID in range(12):
            taxelDict[f"RIGHT_LEG__LOWER_LEG__{patchID}__{taxelID}"] = []

# right arm
# add fingertips
for patchID in range(5):
    for taxelID in range(12):
        taxelDict[f"RIGHT_ARM__HAND__{patchID}__{taxelID}"] = []
# add palm
for taxelID in range(48):
    taxelDict[f"RIGHT_ARM__HAND__{5}__{taxelID}"] = []


print('init done')
if full_body:
    if DEBUG:
        model_path = './models/icub_mk2_right_hand_only_contact_sensor.xml'  # right hand only
    else:
        model_path = './neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub
else:
    if DEBUG:
        model_path = './models/icub_mk2_right_hand_only_contact_sensor.xml'  # right hand only
    else:
        # right hand only
        model_path = './neuromorphic_body_schema/models/icub_mk2_right_hand_only_contact_sensor.xml'


# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

threading.Thread(target=detect_contact, args=(data, taxelDict)).start()
viewer.launch(model, data)
