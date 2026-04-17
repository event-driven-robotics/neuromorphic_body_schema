"""
include_skin_to_mujoco_model.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025
Updated: 13.03.2026

Description:
This script integrates iCub skin sensors (taxels) into a MuJoCo model by
reading skin position files, filtering tactile channels, transforming taxel
coordinates into body frames, and emitting MuJoCo sites and touch sensors.

Current filtering policy:
1. If taxel2Repr is available, channels with taxel2Repr >= 0 are tactile,
    while -1/-2 are filtered out.
2. Calibration rows with zero position and zero normal are always treated as
    non-physical channels and filtered out.
3. If taxel2Repr is missing, filtering falls back to calibration non-zero
    checks only.

This behavior is aligned with the visualization pipeline in helpers/ed_skin.py
to keep model sensor counts and displayed tactile channels consistent.

For more information about taxel placement and data structure please see https://mesh-iit.github.io/documentation/tactile_sensors/

Reproducibility notes for the full skin/taxel alignment workflow are documented
in ``docs/SKIN_TAXEL_ALIGNMENT_REPRODUCIBILITY.md``.
"""

import sys
from pathlib import Path
# Patch sys.path BEFORE any neuromorphic_body_schema imports
def find_project_root(start_path: Path, target_dir: str = "neuromorphic_body_schema") -> Path:
    current = start_path.resolve()
    while current != current.parent:
        if (current / target_dir).is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        f"Could not find {target_dir} in any parent directory of {start_path}")

project_root = find_project_root(Path(__file__))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if __name__ == "__main__":
    print(f"[DEBUG] Project root added to sys.path: {project_root}")
    print(f"[DEBUG] sys.path: {sys.path}")

import numpy as np
from typing import Sequence, cast
import re
import os
import json

from neuromorphic_body_schema.helpers.helpers import (MOJOCO_SKIN_PARTS,
                                                      POSITIONS_FILES)
def find_project_root(start_path: Path, target_dir: str = "neuromorphic_body_schema") -> Path:
    current = start_path.resolve()
    while current != current.parent:
        if (current / target_dir).is_dir():
            return current
        current = current.parent
    raise RuntimeError(
        f"Could not find {target_dir} in any parent directory of {start_path}")


project_root = find_project_root(Path(__file__))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if __name__ == "__main__":
    print(f"[DEBUG] Project root added to sys.path: {project_root}")
    print(f"[DEBUG] sys.path: {sys.path}")

from neuromorphic_body_schema.helpers.helpers import (MOJOCO_SKIN_PARTS,
                                                      POSITIONS_FILES)

# Semantic group constants for taxel site assignment
GROUP_TORSO = 1
GROUP_RIGHT_ARM_HAND = 2
GROUP_LEFT_ARM_HAND = 3
GROUP_RIGHT_LEG = 4
GROUP_LEFT_LEG = 5

# Always resolve resource paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

MUJOCO_MODEL = str(PROJECT_ROOT / "models/icub_v2_full_body.xml")
TAXEL_INI_PATH = str(SCRIPT_DIR / "positions")
MUJOCO_MODEL_OUT = str(
    PROJECT_ROOT / "models/icub_v2_full_body_contact_sensors.xml")
OPTIMIZATION_REPORT_JSON = str(
    SCRIPT_DIR / "taxel_alignment_optimization_report.json")


def load_optimized_deltas(report_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load optimized delta transforms from the optimizer JSON report.

    Returns a mapping:
        part_name -> (delta_angles_deg[3], delta_offsets_m[3])
    """
    if not os.path.exists(report_path):
        return {}

    with open(report_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    deltas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not isinstance(raw, list):
        return deltas

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        part = entry.get("part")
        optimized = entry.get("optimized")
        if not isinstance(part, str) or not isinstance(optimized, dict):
            continue

        angle_values = optimized.get("delta_angles_deg")
        offset_values = optimized.get("delta_offsets_m")
        if angle_values is None or offset_values is None:
            continue

        angles = np.array(angle_values, dtype=float)
        offsets = np.array(offset_values, dtype=float)
        if angles.shape != (3,) or offsets.shape != (3,):
            continue

        deltas[part] = (angles, offsets)

    return deltas


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


def read_taxel2repr_data(file_path: str) -> list[int]:
    """Read taxel2Repr mapping values from a positions file.

    Values have the following semantics in iCub skin files:
    - ``>= 0`` tactile channels,
    - ``-1`` unused channels,
    - ``-2`` thermal/non-tactile channels.
    """
    text = ""
    with open(file_path, "r") as file:
        text = file.read()

    key_idx = text.find("taxel2Repr")
    if key_idx < 0:
        return []

    start_idx = text.find("(", key_idx)
    if start_idx < 0:
        return []

    depth = 0
    end_idx = -1
    for idx in range(start_idx, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx < 0:
        return []

    block = text[start_idx + 1:end_idx]
    return [int(v) for v in re.findall(r"-?\d+", block)]


def validate_taxel_data(
    calibration: np.ndarray, taxel2repr: list[int] | None = None
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Validates and extracts taxel data from the calibration array.

    Channels are kept if they are tactile according to taxel2Repr (when
    provided) and have non-zero calibration content.

    Filtering rules:
    1. If ``taxel2repr[i] < 0`` (unused or thermal), the channel is skipped.
    2. Rows with both zero position and zero normal are skipped.
    3. If taxel2Repr is shorter than calibration, remaining rows are decided
       by calibration non-zero check.

    Args:
        calibration (np.ndarray): A 2D numpy array containing raw calibration data.
        taxel2repr (list[int] | None): Optional taxel2Repr mapping values.

    Returns:
        list: A list of tuples where each tuple contains:
            - [0]: np.ndarray - 3D position vector [x, y, z] of the taxel
            - [1]: np.ndarray - 3D normal vector of the taxel
            - [2]: int - Zero-based index of the taxel
    """
    taxels: list[tuple[np.ndarray, np.ndarray, int]] = []
    size = len(calibration)
    for i in range(size):
        if taxel2repr is not None and i < len(taxel2repr):
            # Keep only tactile channels, skip unused/thermal channels.
            if taxel2repr[i] < 0:
                continue
        taxel_data = calibration[i]
        pos_nrm = taxel_data[:6]
        pos = pos_nrm[:3]
        nrm = pos_nrm[3:]
        if np.linalg.norm(nrm) != 0 or np.linalg.norm(pos) != 0:
            taxels.append((pos, nrm, len(taxels)))
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
    mujoco_model: str,
    path_to_skin: str,
    skin_parts: dict[str, str],
    optimized_deltas: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """
    Integrates tactile sensor (taxel) data into a MuJoCo robot model XML file.

    This function reads taxel positions from configuration files, processes and transforms them
    to match the robot's coordinate system, and inserts them as touch sensors into the MuJoCo
    model. It handles different body parts including limbs, torso, hands, and fingers.

    The function performs the following steps:
    1. Reads taxel calibration data from .txt files in the specified directory
    2. Reads taxel2Repr mapping and reports tactile/unused/thermal counts
    3. Validates and filters taxel data using taxel2Repr + calibration checks
    4. Parses the MuJoCo XML model file
    5. For each body part, transforms taxel positions to the correct coordinate frame
    6. Inserts taxel sites into the XML structure
    7. Adds touch sensor definitions for each taxel
    8. Writes the modified model to a new file
    9. Generates a report of added taxels

    Args:
        mujoco_model (str): Path to the input MuJoCo XML model file.
        path_to_skin (str): Path to the directory containing taxel configuration .txt files.
        skin_parts (dict[str, str]): Dictionary mapping skin part names to their corresponding configuration files (e.g., {"left_arm": "left_arm.ini"}).
        optimized_deltas (dict | None): Optional per-part transform deltas from optimization,
            mapping ``part -> (delta_angles_deg, delta_offsets_m)``. If provided,
            each taxel gets this final delta after the hand-tuned transform chain.

    Returns:
        None: The function writes output to files:
            - Modified MuJoCo model saved to `MUJOCO_MODEL_OUT` path
            - Report saved to "./report_including_taxels.txt"
    """
    # read the taxel positions from the skin configuration file
    if optimized_deltas is None:
        optimized_deltas = {}

    # ini_files_taxels = os.listdir(path_to_skin)
    # only keep the files listed in skin_parts
    ini_files_taxels = list(skin_parts.values())
    all_taxels = []
    for taxels_ini in ini_files_taxels:
        file_path = os.path.join(path_to_skin, taxels_ini)
        if not os.path.isfile(file_path):
            print(f"[ERROR] Calibration file not found: {file_path}")
            continue
        calibration = read_calibration_data(file_path)
        taxel2repr = read_taxel2repr_data(file_path)
        if taxel2repr:
            tactile = len([v for v in taxel2repr if v >= 0])
            unused = len([v for v in taxel2repr if v == -1])
            thermal = len([v for v in taxel2repr if v == -2])
            print(
                f"[{taxels_ini}] taxel2Repr: tactile={tactile}, unused={unused}, thermal={thermal}, total={len(taxel2repr)}"
            )
        else:
            print(
                f"[{taxels_ini}] taxel2Repr not found. Falling back to non-zero calibration filtering only."
            )

        taxels = validate_taxel_data(
            calibration, taxel2repr if taxel2repr else None)
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
                    group_counter = GROUP_RIGHT_LEG
                    parts_to_add.append(part)

                elif part == "r_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "right_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_LEG
                    parts_to_add.append(part)

                elif part == "l_upper_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_upper.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_LEG
                    parts_to_add.append(part)

                elif part == "l_lower_leg":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_leg_lower.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_LEG
                    parts_to_add.append(part)

                elif part == "chest":
                    # find the corresponding taxels
                    part = "torso"
                    pos_of_taxels = ini_files_taxels.index("torso.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_TORSO
                    parts_to_add.append(part)

                elif part == "r_shoulder_3":
                    # find the corresponding taxels
                    part = "r_upper_arm"
                    pos_of_taxels = ini_files_taxels.index(
                        "right_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_forearm":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "right_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand":
                    # find the corresponding taxels
                    part = "r_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "right_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand_thumb_3":
                    # find the corresponding taxels
                    part = "r_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand_index_3":
                    # find the corresponding taxels
                    part = "r_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand_middle_3":
                    # find the corresponding taxels
                    part = "r_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand_ring_3":
                    # find the corresponding taxels
                    part = "r_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "r_hand_little_3":
                    # find the corresponding taxels
                    part = "r_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_RIGHT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_shoulder_3":
                    # find the corresponding taxels
                    part = "l_upper_arm"
                    pos_of_taxels = ini_files_taxels.index("left_arm.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_forearm":
                    # find the corresponding taxels
                    pos_of_taxels = ini_files_taxels.index(
                        "left_forearm_V2.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand":
                    # find the corresponding taxels
                    part = "l_palm"
                    pos_of_taxels = ini_files_taxels.index(
                        "left_hand_V2_1.txt")
                    taxels_to_add = all_taxels[pos_of_taxels]
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand_thumb_3":
                    # find the corresponding taxels
                    part = "l_hand_thumb"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand_index_3":
                    # find the corresponding taxels
                    part = "l_hand_index"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand_middle_3":
                    # find the corresponding taxels
                    part = "l_hand_middle"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand_ring_3":
                    # find the corresponding taxels
                    part = "l_hand_ring"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
                    parts_to_add.append(part)

                elif part == "l_hand_little_3":
                    # find the corresponding taxels
                    part = "l_hand_little"
                    add_finger_taxels = True
                    found_spot_to_add = False
                    add_taxels = True
                    group_counter = GROUP_LEFT_ARM_HAND
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
                                f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" type="sphere" group="{group_counter}" pos="{finger_pos[0]} {finger_pos[1]} {finger_pos[2]}" rgba="0 1 0 0.0"/>\n',
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
                            pos, _, idx = cast(
                                tuple[np.ndarray, np.ndarray, int], taxel)
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

                            # Apply optional optimized per-part refinement as final step.
                            if part in optimized_deltas:
                                delta_angles, delta_offsets = optimized_deltas[part]
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=delta_offsets.tolist(),
                                    angle_degrees=delta_angles.tolist(),
                                )

                            lines.insert(
                                line_counter,
                                f'{" "*identation}<site name="{part}_taxel_{idx}" size="0.005" type="sphere" group="{group_counter}" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 0.5"/>\n',
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

    with open(MUJOCO_MODEL_OUT, "w") as file:
        file.writelines(lines)

    # write report to txt file (also relative to script)
    report_path = str(SCRIPT_DIR / "report_including_taxels.txt")
    with open(report_path, "w") as file:
        file.write(f"Part name: Nb of taxels\n")
        for part_to_add, taxel_id_list in zip(parts_to_add, taxel_ids_to_add):
            file.write(f"{part_to_add}: {len(taxel_id_list)}\n")


if __name__ == "__main__":
    optimized = load_optimized_deltas(OPTIMIZATION_REPORT_JSON)
    if optimized:
        print(
            f"Loaded optimized deltas for {len(optimized)} parts from {OPTIMIZATION_REPORT_JSON}"
        )
    else:
        print(
            "No optimization report found or no valid deltas loaded. Using hand-tuned transforms only."
        )

    include_skin_to_mujoco_model(
        MUJOCO_MODEL,
        TAXEL_INI_PATH,
        POSITIONS_FILES,
        optimized_deltas=optimized,
    )
    print("*******************")
    print("DONE")
    pass
