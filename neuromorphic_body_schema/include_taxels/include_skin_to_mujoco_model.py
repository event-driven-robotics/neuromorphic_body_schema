"""
include_skin_to_mujoco_model.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This script integrates skin sensors (taxels) into a MuJoCo model of the iCub robot. It reads taxel positions from
configuration files, processes these positions, and inserts them into the MuJoCo XML model file. The main functionalities include:

1. Reading and Validating Taxel Data: Functions to read calibration data from .ini files and validate the taxel data.
2. Coordinate System Transformation: Functions to rebase and rotate the coordinate system of the taxels.
3. Integrating Taxels into MuJoCo Model: The main function `include_skin_to_mujoco_model` reads the MuJoCo model file,
   inserts the taxel positions, and defines them as touch sensors in the model.

The script is structured to handle different parts of the robot, ensuring that the taxels are correctly positioned and
oriented according to the robot's body parts. The final output is a modified MuJoCo XML model file with the integrated skin sensors.

"""

import os
from typing import Sequence, cast

import numpy as np

# ini files from [...]/icub-main/app/skinGui/conf/positions/*.ini
skin_parts = [
    "left_arm",
    "left_forearm_V2",
    "left_hand_V2_1",
    "left_leg_lower",
    "left_leg_upper",
    "torso",
    "right_arm",
    "right_forearm_V2",
    "right_hand_V2_1",
    "right_leg_lower",
    "right_leg_upper",
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

MUJOCO_MODEL = "./neuromorphic_body_schema/models/icub_v2_full_body.xml"
TAXEL_INI_PATH = "../icub-main/app/skinGui/conf/positions"
mujoco_model_out = (
    "./neuromorphic_body_schema/models/icub_v2_full_body_contact_sensors.xml"
)


def rebase_coordinate_system(taxel_pos: list[tuple[np.ndarray, np.ndarray, int]]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Rebases the coordinate system of the taxels to the center of the taxel array and rotates them using PCA.

    This function centers the taxel positions by computing their mean, then applies Principal Component Analysis (PCA)
    to find the axes with the largest variance and rotates the coordinate system accordingly.

    Args:
        taxel_pos (list): A list of tuples where each tuple contains:
            - [0]: np.ndarray - 3D position [x, y, z] of the taxel
            - [1]: np.ndarray - 3D normal vector of the taxel
            - [2]: int - Index of the taxel

    Returns:
        list: The same list structure with updated 3D positions after rebasing and rotation.
    """
    # first we want to find the center of the taxel array
    taxel_pos_excluded = taxel_pos[0][0]
    for i in range(1, len(taxel_pos)):
        taxel_pos_excluded = np.vstack((taxel_pos_excluded, taxel_pos[i][0]))
    center = np.mean(taxel_pos_excluded, axis=0)

    # now we can rebase the coordinate system
    taxel_pos_excluded[:, 0] -= center[0]
    taxel_pos_excluded[:, 1] -= center[1]
    taxel_pos_excluded[:, 2] -= center[2]

    # now we want to define the new axis system by using PCA to find the axis with the largest variance
    cov = np.cov(taxel_pos_excluded.T)
    _, eigenvectors = np.linalg.eig(cov)
    # now we want to rotate the taxels to the new coordinate system
    taxel_pos_excluded = np.dot(eigenvectors.T, taxel_pos_excluded.T).T

    # now we can update the taxel_pos array
    for i in range(len(taxel_pos)):
        taxel_pos[i][0][0] = taxel_pos_excluded[i][0]
        taxel_pos[i][0][1] = taxel_pos_excluded[i][1]
        taxel_pos[i][0][2] = taxel_pos_excluded[i][2]

    return taxel_pos


def rotate_position(
    pos: np.ndarray, offsets: Sequence[float], angle_degrees: Sequence[float]
) -> np.ndarray:
    """
    Rotates a position vector around the x, y, and z axes sequentially by specified angles.

    The rotation is applied in the order: X-axis, Y-axis, Z-axis. The position is first
    offset, then rotated around each axis using rotation matrices.

    Args:
        pos (np.ndarray): The 3D position vector to rotate [x, y, z].
        offsets (Sequence[float]): Translation offsets [x_offset, y_offset, z_offset] applied before rotation.
        angle_degrees (Sequence[float]): Rotation angles in degrees [angle_x, angle_y, angle_z] for each axis.

    Returns:
        np.ndarray: The rotated and translated 3D position vector, rounded to 12 decimal places.
    """

    # offset the point
    pos[0] += offsets[0]
    pos[1] += offsets[1]
    pos[2] += offsets[2]

    # rotate the point around x axis
    angle_radians = np.radians(angle_degrees[0])
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    pos = np.dot(rotation_matrix, pos)

    # rotate the point around y axis
    angle_radians = np.radians(angle_degrees[1])
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)],
        ]
    )
    pos = np.dot(rotation_matrix, pos)

    # rotate the point around z axis
    angle_radians = np.radians(angle_degrees[2])
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1],
        ]
    )
    pos = np.dot(rotation_matrix, pos)

    return np.round(pos, 12)


def read_calibration_data(file_path: str) -> np.ndarray:
    """
    Reads taxel calibration data from a configuration file.

    Parses a .txt file containing taxel positions and normals in the [calibration] section.

    Args:
        file_path (str): The path to the calibration data file.

    Returns:
        np.ndarray: A 2D numpy array where each row contains calibration data for one taxel
                    (typically 6 values: 3 for position, 3 for normal vector).
    """
    calibration = []
    start_found = False
    with open(file_path, "r") as file:
        for line in file:
            if start_found:
                if not line.strip():
                    continue
                pos_nrm = list(map(float, line.split()))
                calibration.append(pos_nrm)
            if "[calibration]" in line:
                start_found = True
    return np.array(calibration)


def validate_taxel_data(calibration: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Validates and extracts taxel data from the calibration array.

    Filters out invalid taxels (those with zero position and normal vectors) and structures
    the data for further processing.

    Args:
        calibration (np.ndarray): A 2D numpy array containing raw calibration data.

    Returns:
        list: A list of tuples where each tuple contains:
            - [0]: np.ndarray - 3D position vector [x, y, z] of the taxel
            - [1]: np.ndarray - 3D normal vector of the taxel
            - [2]: int - Zero-based index of the taxel
    """
    taxels = []
    size = len(calibration)
    for i in range(1, size):
        taxel_data = calibration[i]
        pos_nrm = taxel_data[:6]
        pos = pos_nrm[:3]
        nrm = pos_nrm[3:]
        if np.linalg.norm(nrm) != 0 or np.linalg.norm(pos) != 0:
            taxels.append((pos, nrm, i - 1))
    return taxels


def read_triangle_data(file_path: str) -> tuple[str, list[tuple[np.ndarray, int]]]:
    """
    Reads triangle configuration data from a skinGui configuration file.

    Parses a .ini file containing triangle/patch definitions in the [SENSORS] section.

    Args:
        file_path (str): The path to the triangle configuration file.

    Returns:
        tuple: A tuple containing:
            - config_type (str): The configuration type of the last triangle (e.g., "triangle_10pad").
            - triangles (list): A list of tuples, where each tuple contains:
                - [0]: np.ndarray - Array of triangle parameters [x, y, orientation, mirror_flag]
                - [1]: int - Triangle/patch ID
    """
    triangles = []
    start_found = False
    read_header = False
    with open(file_path, "r") as file:
        for line in file:
            if start_found and not read_header:
                if (
                    not line.strip()
                    or "# left lower" in line
                    or "rightupperarm_lower" in line
                ):
                    continue
                entry = line.split()
                config_type = entry[0]
                triangle = list(map(float, entry[1:]))
                triangles.append((np.array(triangle[1:]), int(triangle[0])))
            if read_header:
                read_header = False
                continue
            if "[SENSORS]" in line:
                start_found = True
                read_header = True
    return config_type, triangles


def include_skin_to_mujoco_model(
    mujoco_model: str, path_to_skin: str, skin_parts: list
) -> None:
    """
    Integrates tactile sensor (taxel) data into a MuJoCo robot model XML file.

    This function reads taxel positions from configuration files, processes and transforms them
    to match the robot's coordinate system, and inserts them as touch sensors into the MuJoCo
    model. It handles different body parts including limbs, torso, hands, and fingers.

    The function performs the following steps:
    1. Reads taxel calibration data from .txt files in the specified directory
    2. Validates and filters taxel data
    3. Parses the MuJoCo XML model file
    4. For each body part, transforms taxel positions to the correct coordinate frame
    5. Inserts taxel sites into the XML structure
    6. Adds touch sensor definitions for each taxel
    7. Writes the modified model to a new file
    8. Generates a report of added taxels

    Args:
        mujoco_model (str): Path to the input MuJoCo XML model file.
        path_to_skin (str): Path to the directory containing taxel configuration .txt files.
        skin_parts (list): List of skin part names (strings) to process (e.g., "left_arm", "right_hand_V2_1").

    Returns:
        None: The function writes output to files:
            - Modified MuJoCo model saved to `mujoco_model_out` path
            - Report saved to "./report_including_taxels.txt"
    """
    # read the taxel positions from the skin configuration file
    ini_files_taxels = os.listdir(path_to_skin)
    # only keep the files listed in skin_parts
    ini_files_taxels = [f for f in ini_files_taxels if f.split(".")[
        0] in skin_parts]
    all_taxels = []
    for taxels_ini in ini_files_taxels:
        calibration = read_calibration_data(f"{path_to_skin}/{taxels_ini}")
        taxels = validate_taxel_data(calibration)
        all_taxels.append(taxels)

    # now we can open the xml robot config file and start adding all the taxels
    with open(mujoco_model, "r") as file:
        lines = file.readlines()

    # TODO ensure the right order!
    finger_taxels = [
        [12e-3, 3.25e-3, -3.25e-3],
        [12e-3, 6.5e-3, 0.0],
        [12e-3, 3.25e-3, 3.25e-3],
        [13.5e-3, -1.5e-3, 6.5e-3],
        [13.5e-3, -1.5e-3, -6.5e-3],
        [15e-3, 3.25e-3, 3.25e-3],
        [15e-3, 6.5e-3, 0.0],
        [15e-3, 3.25e-3, -3.25e-3],
        [19.625e-3, 1.625e-3, -3.25e-3],
        [21.25e-3, 3.25e-3, 0.0],
        [19.625e-3, 1.625e-3, 3.25e-3],
        [24.5e-3, 0.0, 0.0],
    ]

    # now we iterate over the lines and add the taxels
    keep_processing = True
    add_finger_taxels = False
    line_counter = 0
    parts_to_add = []
    taxel_ids_to_add = []
    while keep_processing:
        print(lines[line_counter])
        if "<body name=" in lines[line_counter]:
            if any(part in lines[line_counter] for part in MOJOCO_SKIN_PARTS):
                # find the part that is being added
                part = lines[line_counter].split('name="')[1].split('"')[0]
                if part == "r_upper_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "right_leg_upper.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 7
                    parts_to_add.append(part)

                elif part == "r_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 8
                    parts_to_add.append(part)

                elif part == "l_upper_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_upper.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 9
                    parts_to_add.append(part)

                elif part == "l_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 10
                    parts_to_add.append(part)

                elif part == "chest":
                    # find the corresponding taxels
                    part = "torso"
                    pos_of_taxels = ini_files_taxels.index("torso.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 0
                    parts_to_add.append(part)

                elif part == "r_shoulder_3":
                    # find the corresponding taxels
                    part = "r_upper_arm"
                    pos_of_taxels = ini_files_taxels.index("right_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 1
                    parts_to_add.append(part)

                elif part == "r_forearm":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "right_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 2
                    parts_to_add.append(part)

                elif part == "r_hand":
                    # find the corresponding taxels
                    part = "r_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "right_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "r_hand_thumb_3":
                    # find the corresponding taxels
                    part = "r_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "r_hand_index_3":
                    # find the corresponding taxels
                    part = "r_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "r_hand_middle_3":
                    # find the corresponding taxels
                    part = "r_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "r_hand_ring_3":
                    # find the corresponding taxels
                    part = "r_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "r_hand_little_3":
                    # find the corresponding taxels
                    part = "r_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 3
                    parts_to_add.append(part)

                elif part == "l_shoulder_3":
                    # find the corresponding taxels
                    part = "l_upper_arm"
                    pos_of_taxels = ini_files_taxels.index("left_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 4
                    parts_to_add.append(part)

                elif part == "l_forearm":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 5
                    parts_to_add.append(part)

                elif part == "l_hand":
                    # find the corresponding taxels
                    part = "l_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "left_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 6
                    parts_to_add.append(part)

                elif part == "l_hand_thumb_3":
                    # find the corresponding taxels
                    part = "l_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 6
                    parts_to_add.append(part)

                elif part == "l_hand_index_3":
                    # find the corresponding taxels
                    part = "l_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 6
                    parts_to_add.append(part)

                elif part == "l_hand_middle_3":
                    # find the corresponding taxels
                    part = "l_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 6
                    parts_to_add.append(part)

                elif part == "l_hand_ring_3":
                    # find the corresponding taxels
                    part = "l_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    parts_to_add.append(part)

                elif part == "l_hand_little_3":
                    # find the corresponding taxels
                    part = "l_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = 6
                    parts_to_add.append(part)

                if add_taxels:
                    add_taxels = False
                    if add_finger_taxels:
                        while not found_spot_to_add:
                            line_counter += 1
                            if "</body" in lines[line_counter]:
                                # read the number of identations of the line above adding the taxels:
                                identation = len(
                                    lines[line_counter].split("<")[0]) + 8
                                found_spot_to_add = True
                                taxel_ids = []
                        idx = 0
                        # line_counter -= 1
                        for finger_pos in finger_taxels:
                            taxel_ids.append(idx)
                            # add the number of whitespace to the beginning of the line
                            lines.insert(
                                line_counter,
                                f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" pos="{finger_pos[0]} {finger_pos[1]} {finger_pos[2]}" rgba="0 1 0 0.0"/>\n',
                            )
                            line_counter += 1
                            idx += 1
                        add_finger_taxels = False
                    else:
                        while not found_spot_to_add:
                            line_counter += 1
                            if "<body" in lines[line_counter]:
                                # read the number of identations of the line above adding the taxels:
                                identation = len(
                                    lines[line_counter].split("<")[0]) + 4
                                found_spot_to_add = True
                                taxel_ids = []
                        # rebase the coordinate system
                        if (
                            part == "r_upper_arm"
                            or part == "r_forearm"
                            or part == "l_upper_arm"
                            or part == "l_forearm"
                            or part == "torso"
                        ):
                            taxels_to_add = rebase_coordinate_system(
                                taxels_to_add)
                        for taxel in taxels_to_add:
                            # line_counter -= 1  # we want to add the taxels before the next body
                            pos, _, idx = cast(tuple[np.ndarray, np.ndarray, int], taxel)
                            taxel_ids.append(idx)
                            # add the number of whitespace to the beginning of the line
                            if part == "r_upper_arm":
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, -32, 0],
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 0, -2]
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[-0.08, 0.0015, 0.012],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "r_forearm":
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 0, 90]
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 78, 0]
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 0, -2]
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[-0.05, 0, -0.0015],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "l_upper_arm":
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[-270, 0, 0],
                                )
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, -148, 0],
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0.08, 0.0015, 0.012],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "l_forearm":
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, 0, -90],
                                )
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, 102, 0],
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 0, -2]
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0.05, -0.001, 0],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "torso":
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 90, 0]
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[-4, 0, 0]
                                )
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0.06, 0.068],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "r_palm":
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, 0, 180],
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[90, 0, 0]
                                )
                                pos = rotate_position(
                                    pos=pos, offsets=[0, 0, 0], angle_degrees=[0, 15, 0]
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[-0.055, -0.005, 0.02],
                                    angle_degrees=[0, 0, 0],
                                )
                            if part == "l_palm":
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[-90, 0, 0],
                                )
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0, 0, 0],
                                    angle_degrees=[0, -15, 0],
                                )
                                # pos, up-down, left-right, front-back (looking from the front)
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=[0.055, -0.005, 0.02],
                                    angle_degrees=[0, 0, 0],
                                )

                            # TODO possibly use cylinder as type.
                            lines.insert(
                                line_counter,
                                f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" type="sphere" group="{group_counter}" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 0.0"/>\n',
                            )
                            line_counter += 1
                    line_counter -= 1  # go one line back after we added the last taxel
                    pass
                    taxel_ids_to_add.append(taxel_ids)

        elif "<tendon>" in lines[line_counter]:
            keep_processing = False

        line_counter += 1  # we need to keep iterating over the lines

    # now we have to define the sensors as touch sensors
    keep_processing = True
    identation = 4
    while keep_processing:
        # above that we need to add each taxel using:
        # <sensor>
        #     <touch name="name_to_display" site="name_of_taxel" />
        # </sensor>
        if "<contact>" in lines[line_counter]:
            for part_to_add, taxel_id_list in zip(parts_to_add, taxel_ids_to_add):
                lines.insert(line_counter, "\n")
                line_counter += 1
                lines.insert(
                    line_counter, f'{" "*identation}<!--{part_to_add}-->\n')
                line_counter += 1
                for id in taxel_id_list:
                    lines.insert(line_counter, f'{" "*identation}<sensor>\n')
                    line_counter += 1
                    lines.insert(
                        line_counter,
                        f'{" "*(identation+4)}<touch name="{part_to_add}_{id}" site="{part_to_add}_taxel_{id}" />\n',
                    )
                    line_counter += 1
                    lines.insert(line_counter, f'{" "*identation}</sensor>\n')
                    line_counter += 1
                pass
            lines.insert(line_counter, "\n")
            keep_processing = False
            pass
        pass
        line_counter += 1
    pass

    # now we can write the new xml file
    with open(mujoco_model_out, "w") as file:
        file.writelines(lines)
    pass

    # write report to txt file
    with open(f"./report_including_taxels.txt", "w") as file:
        file.write(f"Part name: Nb of taxels\n")
        for part_to_add, taxel_id_list in zip(parts_to_add, taxel_ids_to_add):
            file.write(f"{part_to_add}: {len(taxel_id_list)}\n")


if __name__ == "__main__":
    include_skin_to_mujoco_model(MUJOCO_MODEL, TAXEL_INI_PATH, skin_parts)
    print("*******************")
    print("DONE")
    pass
