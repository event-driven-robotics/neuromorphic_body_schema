"""helpers.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides helper functions, constants, and utilities for the neuromorphic body schema project.
It includes path definitions, sensor group mappings, and utility functions for the MuJoCo simulation environment.

Constants:
- MODEL_PATH: Path to the MuJoCo model XML file.
- TRIANGLE_INI_PATH: Path to the skin GUI configuration files.
- FIG_PATH: Path to the figures directory.
- SKIN_PARTS: List of skin body part identifiers.
- TRIANGLE_FILES: List of triangle configuration file names for skin visualization.
- KEY_MAPPING: Dictionary mapping triangle files to sensor names.
- TIME_WINDOW, TICK_HEIGHT, MARGIN, HEIGHT, WIDTH: Constants for proprioception visualization.

Classes:
- DynamicGroupedSensors: Provides dynamic access to grouped sensor data from a simulation.

Functions:
- init_POV: Initializes the point of view for the MuJoCo viewer.

"""

import os
from pathlib import Path

import numpy as np

# Model and configuration paths
_PACKAGE_DIR = Path(__file__).parent.parent
MODEL_PATH = str(_PACKAGE_DIR / "models" /
                 "icub_v2_full_body_contact_sensors.xml")
# Prefer the current folder name in this repository (skinGUI), but keep a
# fallback for earlier naming variants.
_triangle_ini_dir = _PACKAGE_DIR / "skinGUI"
if not _triangle_ini_dir.exists():
    _triangle_ini_dir = _PACKAGE_DIR / "skinGui"
TRIANGLE_INI_PATH = str(_triangle_ini_dir)
FIG_PATH = str(_PACKAGE_DIR / "figures")

### SKIN CONFIGURATION ###
# List of all skin-covered body parts in the iCub robot
SKIN_PARTS = [
    "r_hand",
    "r_forearm",
    "r_upper_arm",
    "torso",
    "l_hand",
    "l_forearm",
    "l_upper_arm",
    "r_upper_leg",
    "r_lower_leg",
    "l_upper_leg",
    "l_lower_leg",
]

MOJOCO_SKIN_PARTS = [
    "r_upper_leg",
    "r_lower_leg",
    "l_upper_leg",
    "l_lower_leg",
    "chest",
    "r_shoulder_3",
    "r_forearm",
    "r_hand",
    "r_hand_thumb_3",
    "r_hand_index_3",
    "r_hand_middle_3",
    "r_hand_ring_3",
    "r_hand_little_3",
    "l_shoulder_3",
    "l_forearm",
    "l_hand",
    "l_hand_thumb_3",
    "l_hand_index_3",
    "l_hand_middle_3",
    "l_hand_ring_3",
    "l_hand_little_3",
]

# Mapping from triangle configuration basename → positions file basename.
# Used to look up taxel2Repr for non-tactile channel filtering in the visualizer.
POSITIONS_FILES = {
    "right_arm":        "right_arm.txt",
    "left_arm":         "left_arm.txt",
    "torso":            "torso.txt",
    "right_forearm": "right_forearm_V2.txt",
    "left_forearm":  "left_forearm_V2.txt",
    # skinGUI uses *_V2_2 while positions files are stored as *_V2_1.
    "right_hand":  "right_hand_V2_1.txt",
    "left_hand":   "left_hand_V2_1.txt",
    "right_leg_upper":  "right_leg_upper.txt",
    "left_leg_upper":   "left_leg_upper.txt",
    "right_leg_lower":  "right_leg_lower.txt",
    "left_leg_lower":   "left_leg_lower.txt",
}

# Deterministic remap from skinGUI triangle patch IDs to taxel2Repr module IDs.
#
# Arm skinGUI files currently draw one module ID that is not represented as a
# tactile block in positions/taxel2Repr (7), while the tactile block 2 is not
# drawn directly. We remap 7 -> 2 so visualization uses the same tactile
# channels as the generated MuJoCo model and live sensor vectors.
ARM_PATCH_ID_REMAP = {
    "right_arm": {7: 2},
    "left_arm": {7: 2},
}

# Triangle configuration file names corresponding to each skin part for visualization
TRIANGLE_FILES = {
    "right_hand": "right_hand_V2_2",
    "right_forearm": "right_forearm_V2",
    "right_arm": "right_arm",
    "torso": "torso",
    "left_hand": "left_hand_V2_2",
    "left_forearm": "left_forearm_V2",
    "left_arm": "left_arm",
    "right_leg_upper": "right_leg_upper",
    "right_leg_lower": "right_leg_lower",
    "left_leg_upper": "left_leg_upper",
    "left_leg_lower":"left_leg_lower",
}

# Mapping from triangle configuration files to MuJoCo sensor names
# This dictionary links skin visualization configurations to their corresponding
# tactile sensor data in the simulation
KEY_MAPPING = {
    "right_leg_upper": "r_upper_leg_taxel",
    "left_leg_upper": "l_upper_leg_taxel",
    "right_leg_lower": "r_lower_leg_taxel",
    "left_leg_lower": "l_lower_leg_taxel",
    "right_hand_V2_2": [
        "r_palm_taxel",
        "r_hand_thumb_taxel",
        "r_hand_index_taxel",
        "r_hand_middle_taxel",
        "r_hand_ring_taxel",
        "r_hand_little_taxel",
    ],
    "torso": "torso_taxel",
    "right_forearm_V2": "r_forearm_taxel",
    "left_forearm_V2": "l_forearm_taxel",
    "right_arm": "r_upper_arm_taxel",
    "left_arm": "l_upper_arm_taxel",
    "left_hand_V2_2": [
        "l_palm_taxel",
        "l_hand_thumb_taxel",
        "l_hand_index_taxel",
        "l_hand_middle_taxel",
        "l_hand_ring_taxel",
        "l_hand_little_taxel",
    ],
}


class DynamicGroupedSensors:
    """
    Provides dynamic access to grouped sensor data from a simulation.

    This class allows accessing specific sensor groups dynamically by mapping keys to sensor data indices.

    Attributes:
        data (mujoco.MjData): The MuJoCo data object containing sensor data.
        grouped_sensors (dict): A dictionary mapping sensor group names to their corresponding indices in the sensor data.
    """

    def __init__(self, data, grouped_sensors):
        """
        Initializes the DynamicGroupedSensors class with sensor data and group mappings.

        Args:
            data (mujoco.MjData): The MuJoCo data object containing sensor data.
            grouped_sensors (dict): A dictionary mapping sensor group names to their corresponding indices in the sensor data.
        """

        self.data = data
        self.grouped_sensors = grouped_sensors

    def __getitem__(self, key):
        """
        Retrieves sensor data for a specific group.

        Args:
            key (str): The name of the sensor group to retrieve.

        Returns:
            np.array: The sensor data corresponding to the specified group.
        """

        adrs = self.grouped_sensors[key]
        return self.data.sensordata[adrs]


### VIEWER ###
def init_POV(viewer):
    """
    Initializes the point of view (POV) for the MuJoCo viewer.

    This function sets the camera's azimuth, distance, elevation, and look-at point to predefined values
    for a consistent and clear view of the simulation.

    Args:
        viewer (mujoco.viewer.Viewer): The MuJoCo viewer instance to configure.

    Returns:
        mujoco.viewer.Viewer: The configured viewer instance with the updated camera settings.
    """

    viewer.cam.azimuth = -4.5
    viewer.cam.distance = 2
    viewer.cam.elevation = -16
    viewer.cam.lookat = np.array([0, -0.25, 1])

    # Example: Only show groups 0 and 2, hide others
    viewer.opt.sitegroup[:] = 0  # Hide all groups
    viewer.opt.sitegroup[0] = 1  # Show group 0

    return viewer


### PROPRIOCEPTION VISUALIZATION CONSTANTS ###
# These constants define the parameters for visualizing proprioception events as time-series plots

TIME_WINDOW = 10  # Time window for event display in seconds
TICK_HEIGHT = 20  # Height of each neuron's event tick display in pixels
MARGIN = 5  # Vertical spacing between neuron displays in pixels
HEIGHT = 8 * (TICK_HEIGHT + MARGIN)  # Total display height (8 neurons per joint)
WIDTH = 500  # Display width in pixels
