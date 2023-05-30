import mujoco
from mujoco import viewer
# import cv2
import threading
import logging
import numpy as np
# import time


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def detect_contact(model, data):
    ''' Loop to monitor contact information'''
    while True:
        # print("Gonna show a contact info soon my friend. Be patient!")
        # print(sensor_data)
        sensor_data = data.sensordata
        # only output info when non-zero contact detected
        if np.sum(sensor_data) > 0.0:
            taxel = np.where(sensor_data != 0.0)[0]  # get non-zero taxels IDs
            values = sensor_data[taxel]  # get non-zero taxel readings
            sensor_data = np.column_stack((taxel, values))  # here we fuse the taxel id and sensor reading in a 2d array
            print("Sensor readings:")
            print(sensor_data)
            print('\n')

        # the assignment of the sensor name to the taxelDict
        # seperate sensor name by '__', you get bodyPArt, skinPArt, patchID, taxelID
        # bodyPArt, skinPArt, patchID, taxelID = sensorName.split('__')
        # oldValue = taxelDict[bodyPArt, skinPArt, patchID, taxelID]  # read old sensor value
        # taxelDict[bodyPArt, skinPArt, patchID, taxelID] = sensorValue  # write new value to the dict

        print('End of contact detection for this cycle.\n\n')


model_path = 'neuromorphic_body_schema/models/icub_mk2_right_hand_only_contact_sensor.xml'  # right hand only
# model_path = 'neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml'  # full iCub

taxelDict = {}
# torso
for patchID in range(3):
    # example taxel numbers
    for taxelID in range(12):
        taxelDict[("TORSO", "TORSO", patchID, taxelID)] = 0.0

# left arm (each fingertip is concidered a patch and all palm is a single patch)
# add fingertips
for patchID in range(5):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "HAND", patchID, taxelID)] = 0.0
# add palm
for taxelID in range(48):
    taxelDict[("LEFT_ARM", "HAND", 5, taxelID)] = 0.0
# add forearm
for patchID in range(2):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "FOREARM", patchID, taxelID)] = 0.0
# add forearm
for patchID in range(2):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "UPPER_ARM", patchID, taxelID)] = 0.0

# right arm
# add fingertips
for patchID in range(5):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "HAND", patchID, taxelID)] = 0.0
# add palm
for taxelID in range(48):
    taxelDict[("RIGHT_ARM", "HAND", 5, taxelID)] = 0.0
# add forearm
for patchID in range(2):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "FOREARM", patchID, taxelID)] = 0.0
# add forearm
for patchID in range(2):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "UPPER_ARM", patchID, taxelID)] = 0.0

# add left leg
for taxelID in range(4):
    taxelDict[("LEFT_LEG", "FOOT", 0, taxelID)] = 0.0
# add upper leg
for patchID in range(3):
    for taxelID in range(12):
        taxelDict[("LEFT_LEG", "UPPER_LEG", patchID, taxelID)] = 0.0
# add lower leg
for patchID in range(3):
    for taxelID in range(12):
        taxelDict[("LEFT_LEG", "LOWER_LEG", patchID, taxelID)] = 0.0

# add right leg
for taxelID in range(4):
    taxelDict[("RIGHT_LEG", "FOOT", 0, taxelID)] = 0.0
# add upper leg
for patchID in range(3):
    for taxelID in range(12):
        taxelDict[("RIGHT_LEG", "UPPER_LEG", patchID, taxelID)] = 0.0
# add lower leg
for patchID in range(3):
    for taxelID in range(12):
        taxelDict[("RIGHT_LEG", "LOWER_LEG", patchID, taxelID)] = 0.0
print('init done')

# Load the MuJoCo model and create a simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

threading.Thread(target=detect_contact, args=(model, data)).start()
viewer.launch(model, data)
