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
import argparse
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
import xml.etree.ElementTree as ET

from neuromorphic_body_schema.helpers.helpers import POSITIONS_FILES

# Semantic group constants for taxel site assignment
GROUP_TORSO = 1
GROUP_RIGHT_ARM_HAND = 2
GROUP_LEFT_ARM_HAND = 3
GROUP_RIGHT_LEG = 4
GROUP_LEFT_LEG = 5

PART_TO_MODEL_BODY = {
    "r_upper_leg": "r_upper_leg",
    "r_lower_leg": "r_lower_leg",
    "l_upper_leg": "l_upper_leg",
    "l_lower_leg": "l_lower_leg",
    "r_upper_arm": "r_shoulder_3",
    "r_forearm": "r_forearm",
    "l_upper_arm": "l_shoulder_3",
    "l_forearm": "l_forearm",
    "torso": "chest",
    "r_palm": "r_hand",
    "l_palm": "l_hand",
    "r_foot": "r_ankle_2",
    "l_foot": "l_ankle_2",
}

PART_TO_GROUP = {
    "r_upper_leg": GROUP_RIGHT_LEG,
    "r_lower_leg": GROUP_RIGHT_LEG,
    "r_foot": GROUP_RIGHT_LEG,
    "l_upper_leg": GROUP_LEFT_LEG,
    "l_lower_leg": GROUP_LEFT_LEG,
    "l_foot": GROUP_LEFT_LEG,
    "torso": GROUP_TORSO,
    "r_upper_arm": GROUP_RIGHT_ARM_HAND,
    "r_forearm": GROUP_RIGHT_ARM_HAND,
    "r_palm": GROUP_RIGHT_ARM_HAND,
    "l_upper_arm": GROUP_LEFT_ARM_HAND,
    "l_forearm": GROUP_LEFT_ARM_HAND,
    "l_palm": GROUP_LEFT_ARM_HAND,
}

FINGER_BODY_TO_PART_GROUP = {
    "r_hand_thumb_3": ("r_hand_thumb", GROUP_RIGHT_ARM_HAND),
    "r_hand_index_3": ("r_hand_index", GROUP_RIGHT_ARM_HAND),
    "r_hand_middle_3": ("r_hand_middle", GROUP_RIGHT_ARM_HAND),
    "r_hand_ring_3": ("r_hand_ring", GROUP_RIGHT_ARM_HAND),
    "r_hand_little_3": ("r_hand_little", GROUP_RIGHT_ARM_HAND),
    "l_hand_thumb_3": ("l_hand_thumb", GROUP_LEFT_ARM_HAND),
    "l_hand_index_3": ("l_hand_index", GROUP_LEFT_ARM_HAND),
    "l_hand_middle_3": ("l_hand_middle", GROUP_LEFT_ARM_HAND),
    "l_hand_ring_3": ("l_hand_ring", GROUP_LEFT_ARM_HAND),
    "l_hand_little_3": ("l_hand_little", GROUP_LEFT_ARM_HAND),
}

ICUB_HEAD_COLOR = [1, 1, 1, 1]  # white
ICUB_SKIN_COLOR = [0.57647, 0.1176, 0.7882, 1]  # purple
ICUB_METAL_PARTS_COLOR = [0.62, 0.64, 0.67, 1]  # plain steel-like metallic gray
ICUB_BLACK_PARTS_COLOR = [0.08, 0.08, 0.08, 1.0]  # near-black to retain lighting detail

ICUB_MATERIAL_METAL = "icub_mat_metal_steel"
ICUB_MATERIAL_HEAD = "icub_mat_head_white_plastic"
ICUB_MATERIAL_SKIN_FABRIC = "icub_mat_skin_fabric"
ICUB_MATERIAL_SKIN_PLASTIC = "icub_mat_skin_plastic"
ICUB_MATERIAL_BLACK = "icub_mat_black_plastic"
ICUB_MATERIAL_COL_INVISIBLE = "icub_mat_collision_invisible"

HEAD_BODY_NAMES = {"head", "eyes_tilt_frame", "l_eye", "r_eye"}
PALM_BODY_NAMES = {"r_hand", "l_hand"}
FINGERTIP_BODY_NAMES = set(FINGER_BODY_TO_PART_GROUP.keys())
# Feet are skin-colored in appearance but are smooth plastic, not fabric.
SKIN_PLASTIC_BODY_NAMES = {"l_foot", "r_foot"}
# Taxel sites for feet live in l_ankle_2/r_ankle_2, but the visible foot mesh
# sits in l_foot/r_foot — remap so the foot body gets skin color, not ankle.
_SKIN_COLOR_BODY_REMAP = {"l_ankle_2": "l_foot", "r_ankle_2": "r_foot"}

FINGERTIP_TAXELS_BY_PART = {
    "r_hand_thumb": (
        (0.012047259631282814, 0.0045707405325671416, -0.0045030484852386157),
        (0.011980076404794088, 0.0064108374585990996, -6.5361078921761812e-05),
        (0.012034364144978112, 0.0045295807490834455, 0.0045174087801547642),
        (0.013512938173992043, -0.0030444420712387081, 0.0069348935724929863),
        (0.013586045888444928, -0.0030630631815760411, -0.0069280814164190017),
        (0.015033272352705802, 0.0044889268531540065, 0.0044771416030458732),
        (0.014449999695987746, 0.0063740000881873527, -0.00076800000466428515),
        (0.014886088506634539, 0.0043423223493625259, -0.0046367377630143732),
        (0.020401619351201664, 0.0025225597409109378, -0.0046364468937028242),
        (0.022105896211514976, 0.0041787195446876691, 0.000211990955292283),
        (0.020113768827948346, 0.002187599351018729, 0.0049283838556601085),
        (0.023941254030355255, -0.00027467491604982446, 0.00044836720013818855),
    ),
    "r_hand_index": (
        (0.01261799970532732, 0.0045670001495719441, -0.0045880002363938716),
        (0.011999742708112134, 0.0064731513297238957, -2.0935087344821617e-06),
        (0.012030268367907371, 0.004691586044355424, 0.004392109919447139),
        (0.012438966899463486, -0.0023114686960903231, 0.0068332893164206234),
        (0.013254007291606709, -0.0015865388292310087, -0.006716231428981108),
        (0.014285000008871515, 0.0045349996832873595, 0.0045300000325872673),
        (0.014985431840989759, 0.0063058290275956287, -3.9878429495038567e-07),
        (0.015122697105630415, 0.0043304415650389436, -0.0045481501511354504),
        (0.0203272514727816, 0.0022744400136935311, -0.0047647468511684155),
        (0.022911279492062377, 0.0030207840753381896, -0.00067252108607222805),
        (0.020162961366099009, 0.0023662329836872239, 0.0048590598575416104),
        (0.023920225139641132, -8.963295719191652e-05, -1.2769220969348219e-05),
    ),
    "r_hand_middle": (
        (0.012003930282241762, 0.0044974936167050967, -0.0045726296388056645),
        (0.011999233129475251, 0.0064522796179769873, -4.7720480334224554e-06),
        (0.010866999638505143, 0.004567000301927744, 0.0046270001795416403),
        (0.013199999834757253, -0.0030539999483501705, 0.0069590003410847524),
        (0.013614797219085473, -0.0030325279491851872, -0.0069410808449808636),
        (0.014461999790953983, 0.0045400001732966598, 0.0045090002619391986),
        (0.014027999607961967, 0.006419999966585919, -7.9999950053455209e-06),
        (0.014272999917523477, 0.0045260002386216117, -0.0045389997308839612),
        (0.020162420420601103, 0.0023608366238434277, -0.0048612631960328993),
        (0.022132843770068293, 0.0041351812848059505, -0.00013318212109208222),
        (0.02008996246766586, 0.0023015144392088062, 0.0048749820800322967),
        (0.023932148329995212, -8.1744644567379094e-05, 9.1659289881146334e-05),
    ),
    "r_hand_ring": (
        (0.012033739936541657, 0.0046645391179591681, -0.0044384720282379022),
        (0.011999224954208829, 0.0064351579502537424, 7.3817362104854675e-07),
        (0.011633000338530443, 0.0045960001546406053, 0.0045839998027610617),
        (0.012380114775990885, -0.0022503616410477741, 0.0068228215190210646),
        (0.013241580290697226, -0.001601772824185064, -0.0067171954052858054),
        (0.01508517116884907, 0.0045805962762284821, 0.0043309561635766517),
        (0.014263000317824185, 0.0064079999584395154, -7.9999989584311011e-06),
        (0.01425900025148601, 0.0045269996721492801, -0.0045390000858672955),
        (0.020557639204820007, 0.0020972543974862603, -0.0048078161986659217),
        (0.022132590876620343, 0.0041366307442462858, -0.0001326219037250312),
        (0.020149704272438982, 0.0023178170738270623, 0.0048838477505339139),
        (0.023930889956992604, -8.6692346628422338e-05, -2.0375636135457997e-05),
    ),
    "r_hand_little": (
        (0.012047259519499859, 0.0045707402802689298, -0.0045030485493019815),
        (0.011980076390079156, 0.0064108373906846608, -6.5361113208360546e-05),
        (0.012034364187797146, 0.0045295803369279379, 0.0045174089779360634),
        (0.012502656013738515, -0.0020643403376467443, 0.0068005403740342895),
        (0.01254430946108411, -0.0014715375403533046, -0.0067209834538370771),
        (0.015033272392618782, 0.0044889263965399223, 0.0044771417375375801),
        (0.014449999688142924, 0.0063739999189872309, -0.0007680000520756382),
        (0.01488608865327672, 0.0043423224691066329, -0.0046367376256124977),
        (0.020233584478308971, 0.0023048994043386268, -0.0048279664903975858),
        (0.022402436385330293, 0.0039163396807836148, 0.00016420362330991228),
        (0.020110712219259655, 0.0021756234282909468, 0.0049349311442722253),
        (0.02393851835977796, -0.00022528801173371331, -0.00039383984965300713),
    ),
    "l_hand_thumb": (
        (0.011941711038461948, 0.0041219383423117522, -0.0049913325333065866),
        (0.011998628423693756, 0.0064552537493152325, -5.1013771749121566e-06),
        (0.012109542898699744, 0.0045003961916796533, 0.0045304338234903431),
        (0.013511174706647736, -0.0014850383532294437, 0.0066513794397634676),
        (0.012374496949496452, -0.002319260504858105, -0.0068337479846089589),
        (0.015098345720439395, 0.0043725840813513264, 0.0043995513478357808),
        (0.014995848474371638, 0.0063645607935590024, -1.5440991496557194e-05),
        (0.013394000023885782, 0.0040059996733262165, -0.0050980001479671135),
        (0.019866565833230775, 0.0025701041173444403, -0.0049791825440654155),
        (0.022091615642475752, 0.0040699006950340047, -0.00013618690870325235),
        (0.02009561407804742, 0.0028075818074804523, 0.0045765450728875385),
        (0.023774856976853141, -7.6620452316416426e-05, 1.2106145436662288e-05),
    ),
    "l_hand_index": (
        (0.01201674737067321, 0.0044581373138431026, -0.0045488045064105047),
        (0.011995039412701424, 0.0063384423799347596, 1.3726109228760081e-06),
        (0.011980058167469346, 0.0044584471091887456, 0.004558726487827681),
        (0.013505665197097101, -0.0014845673670705508, 0.0066238133835700222),
        (0.012842634661722379, -0.0029608112648142303, -0.0069264072583120513),
        (0.01496509546274934, 0.0045338194827460043, 0.0045346141821692929),
        (0.014992213934348729, 0.0062464215100797218, 2.1544301341553784e-06),
        (0.015016479977938223, 0.0044388478894379566, -0.0045280674667905838),
        (0.020371976251569501, 0.0026656800340138379, -0.0044259089568925688),
        (0.021839815091784214, 0.00407153469615993, 0.00014880707796580182),
        (0.020273118201790265, 0.0019088950167688314, 0.0050202957456135146),
        (0.023814984769137473, -0.0001248604886951086, 0.00014011494564089891),
    ),
    "l_hand_middle": (
        (0.010576000468163836, 0.0036360001984907684, -0.0054050001421984068),
        (0.012142348243972978, 0.0061619402606434455, 0.00027170694511837118),
        (0.012057844806801846, 0.0044641823277328857, 0.0045438236859403226),
        (0.012644954982889919, -0.0027452106171071107, 0.0068804843730580351),
        (0.013825472175714294, -0.0016066475589449946, -0.0066834628348548999),
        (0.014342191030063362, 0.004490069708920099, 0.0044174003034660641),
        (0.014953268888791857, 0.0059313641730706964, -8.9369205359749989e-07),
        (0.014602285107289663, 0.004992903795625751, -0.0036971292780375318),
        (0.020388819336650814, 0.0026309988152529405, -0.0044347080108323046),
        (0.021798514791300634, 0.0042028928293132142, 9.8724311417717255e-05),
        (0.020122479286285907, 0.0024716672621931898, 0.0046054915873400678),
        (0.0239720733833702, -5.1926365196606778e-05, -0.00014418133836254074),
    ),
    "l_hand_ring": (
        (0.01046799961086536, 0.0043919997524777604, -0.0050470002407799093),
        (0.012005985995785971, 0.0065730763805567256, 2.7646517184924213e-06),
        (0.012115055199039628, 0.004466687748902156, 0.0046589088081414425),
        (0.01350425612974313, -0.0014689782412059932, 0.0067274827469156013),
        (0.013132000439065388, -0.0030689999902330884, -0.0069589999832612177),
        (0.01510363876673021, 0.004345961059078648, 0.0045191088497514792),
        (0.01498601872835803, 0.0063293181679864056, -6.4572960044911778e-06),
        (0.015086879111369446, 0.0046097486233352735, -0.0044800639791458576),
        (0.019911117993382167, 0.0026574671347161061, -0.005093526810451738),
        (0.022216575851313237, 0.0040831604729219765, 0.00024103783343606684),
        (0.020245199604236378, 0.0020051237734332659, 0.0048277063465591036),
        (0.02383286667573915, 9.1152615179709851e-06, 0.00015556676798521686),
    ),
    "l_hand_little": (
        (0.01201275191111916, 0.0046087584781239642, -0.0044764233811945796),
        (0.011986502547053151, 0.0063180153538259986, -1.3286270885298507e-05),
        (0.011998040329846808, 0.0043992962493249214, 0.0046287005216705352),
        (0.013503786932177589, -0.0014818807168613993, 0.0066166945347963723),
        (0.012875765126979848, -0.0029070549424707508, -0.0069216019939993666),
        (0.014611366208655671, 0.0037258814222157471, 0.0051937792893333263),
        (0.014970176658023721, 0.0060978952207759911, -2.9356723950960182e-05),
        (0.013048999830303887, 0.0039360003338732193, -0.0052109997963859451),
        (0.020148941053633306, 0.0019550246454600045, -0.0051227142779061655),
        (0.022189513663898611, 0.0040557288344130905, 0.00024761373377650889),
        (0.019947259207003937, 0.0025239106380210729, 0.005104386600974441),
        (0.023815084895939437, -0.0001249059215562222, 6.674192825322021e-05),
    ),
}

# Always resolve resource paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

_MODEL_INPUT_PATH = PROJECT_ROOT / "models/icub_v2_full_body_improved.xml"
MUJOCO_MODEL = str(_MODEL_INPUT_PATH)
TAXEL_INI_PATH = str(SCRIPT_DIR / "positions")
MUJOCO_MODEL_OUT = str(
    _MODEL_INPUT_PATH.with_name(
        f"{_MODEL_INPUT_PATH.stem}_contact_sensors{_MODEL_INPUT_PATH.suffix}"
    )
)
OPTIMIZATION_REPORT_JSON = str(
    SCRIPT_DIR / "taxel_alignment_optimization_report.json")

REPORT_REQUIRED_PARTS = {
    "r_upper_leg",
    "r_lower_leg",
    "l_upper_leg",
    "l_lower_leg",
    "r_upper_arm",
    "r_forearm",
    "l_upper_arm",
    "l_forearm",
    "torso",
    "r_palm",
    "l_palm",
    "r_foot",
    "l_foot",
}
REPORT_REQUIRED_ENTRY_KEYS = {
    "part",
    "model_body_name",
    "sensor_group",
    "position_file",
    "rebase",
    "include_to_model",
    "manual_steps",
    "optimized",
}


def _parse_report_json(report_path: str) -> list[dict[str, object]]:
    """Read the optimization report and return raw entry dictionaries."""

    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Optimization report not found at {report_path}. Run optimize_taxel_alignment.py first."
        )

    with open(report_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, list):
        raise ValueError(
            f"Optimization report at {report_path} must be a JSON list of entries."
        )

    entries: list[dict[str, object]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Optimization report entry #{idx} in {report_path} is not an object."
            )
        entries.append(cast(dict[str, object], entry))
    return entries


def validate_optimization_report(report_path: str) -> list[dict[str, object]]:
    """Validate optimization report schema and completeness for strict mode."""

    entries = _parse_report_json(report_path)

    seen_parts: set[str] = set()
    unknown_parts: set[str] = set()
    missing_required_keys: list[str] = []
    invalid_entries: list[str] = []

    for idx, entry in enumerate(entries):
        part = entry.get("part")
        if not isinstance(part, str):
            invalid_entries.append(
                f"entry #{idx} has missing or non-string 'part'"
            )
            continue

        if part in seen_parts:
            invalid_entries.append(f"duplicate part entry: {part}")
            continue
        seen_parts.add(part)

        if part not in REPORT_REQUIRED_PARTS:
            unknown_parts.add(part)

        entry_keys = set(entry.keys())
        missing_keys = sorted(REPORT_REQUIRED_ENTRY_KEYS - entry_keys)
        if missing_keys:
            missing_required_keys.append(
                f"{part}: missing keys {', '.join(missing_keys)}"
            )
            continue

        model_body_name = entry.get("model_body_name")
        sensor_group = entry.get("sensor_group")
        position_file = entry.get("position_file")
        manual_steps = entry.get("manual_steps")
        optimized = entry.get("optimized")

        if not isinstance(model_body_name, str):
            invalid_entries.append(f"{part}: model_body_name must be a string")
        if not isinstance(sensor_group, int):
            invalid_entries.append(f"{part}: sensor_group must be an int")
        if not isinstance(position_file, str):
            invalid_entries.append(f"{part}: position_file must be a string")
        if not isinstance(entry.get("rebase"), bool):
            invalid_entries.append(f"{part}: rebase must be a bool")
        if not isinstance(entry.get("include_to_model"), bool):
            invalid_entries.append(f"{part}: include_to_model must be a bool")
        if not isinstance(manual_steps, list):
            invalid_entries.append(f"{part}: manual_steps must be a list")
        if not isinstance(optimized, dict):
            invalid_entries.append(f"{part}: optimized must be an object")
        else:
            delta_angles = optimized.get("delta_angles_deg")
            delta_offsets = optimized.get("delta_offsets_m")
            if not (
                isinstance(delta_angles, list)
                and len(delta_angles) == 3
                and all(isinstance(value, (int, float)) for value in delta_angles)
            ):
                invalid_entries.append(
                    f"{part}: optimized.delta_angles_deg must be a length-3 numeric list"
                )
            if not (
                isinstance(delta_offsets, list)
                and len(delta_offsets) == 3
                and all(isinstance(value, (int, float)) for value in delta_offsets)
            ):
                invalid_entries.append(
                    f"{part}: optimized.delta_offsets_m must be a length-3 numeric list"
                )

        if isinstance(position_file, str):
            position_path = SCRIPT_DIR / "positions" / position_file
            if not position_path.exists():
                invalid_entries.append(
                    f"{part}: referenced position file does not exist: {position_path}"
                )

        if isinstance(manual_steps, list):
            for step_idx, step in enumerate(manual_steps):
                if not isinstance(step, dict):
                    invalid_entries.append(
                        f"{part}: manual_steps[{step_idx}] must be an object"
                    )
                    continue
                offsets = step.get("offsets_m")
                angles = step.get("angles_deg")
                if not (
                    isinstance(offsets, list)
                    and len(offsets) == 3
                    and all(isinstance(value, (int, float)) for value in offsets)
                ):
                    invalid_entries.append(
                        f"{part}: manual_steps[{step_idx}].offsets_m must be a length-3 numeric list"
                    )
                if not (
                    isinstance(angles, list)
                    and len(angles) == 3
                    and all(isinstance(value, (int, float)) for value in angles)
                ):
                    invalid_entries.append(
                        f"{part}: manual_steps[{step_idx}].angles_deg must be a length-3 numeric list"
                    )

    missing_parts = sorted(REPORT_REQUIRED_PARTS - seen_parts)

    problems: list[str] = []
    if unknown_parts:
        problems.append(f"unknown parts present: {', '.join(sorted(unknown_parts))}")
    if missing_parts:
        problems.append(f"missing required parts: {', '.join(missing_parts)}")
    problems.extend(missing_required_keys)
    problems.extend(invalid_entries)

    if problems:
        raise ValueError(
            "Strict optimization report validation failed:\n- " + "\n- ".join(problems)
        )

    return entries


def load_optimized_deltas(report_path: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Load optimized delta transforms from the optimizer JSON report.

    Returns a mapping:
        part_name -> (delta_angles_deg[3], delta_offsets_m[3], delta_quaternion_wxyz[4] | None)
    """
    deltas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    if not os.path.exists(report_path):
        return deltas

    for entry in _parse_report_json(report_path):
        part = entry.get("part")
        optimized = entry.get("optimized")
        if not isinstance(part, str) or not isinstance(optimized, dict):
            continue

        angle_values = optimized.get("delta_angles_deg")
        offset_values = optimized.get("delta_offsets_m")
        quat_values = optimized.get("delta_quaternion_wxyz")
        if angle_values is None or offset_values is None:
            continue

        angles = np.array(angle_values, dtype=float)
        offsets = np.array(offset_values, dtype=float)
        if angles.shape != (3,) or offsets.shape != (3,):
            continue

        quat: np.ndarray | None = None
        if quat_values is not None:
            quat_arr = np.array(quat_values, dtype=float)
            if quat_arr.shape == (4,):
                quat = quat_arr

        deltas[part] = (angles, offsets, quat)

    return deltas


def load_report_part_metadata(report_path: str) -> dict[str, dict[str, object]]:
    """Load part metadata from optimizer report keyed by part name."""

    if not os.path.exists(report_path):
        return {}

    out: dict[str, dict[str, object]] = {}
    for entry in _parse_report_json(report_path):
        part = entry.get("part")
        if not isinstance(part, str):
            continue

        model_body_name = entry.get("model_body_name")
        sensor_group = entry.get("sensor_group")
        position_file = entry.get("position_file")
        rebase = bool(entry.get("rebase", False))
        include_to_model = bool(entry.get("include_to_model", entry.get("inlcude_to_model", True)))
        manual_steps_raw = entry.get("manual_steps", [])

        manual_steps: list[tuple[list[float], list[float]]] = []
        if isinstance(manual_steps_raw, list):
            for step in manual_steps_raw:
                if not isinstance(step, dict):
                    continue
                offsets = step.get("offsets_m")
                angles = step.get("angles_deg")
                if isinstance(offsets, list) and isinstance(angles, list) and len(offsets) == 3 and len(angles) == 3:
                    manual_steps.append((
                        [float(offsets[0]), float(offsets[1]), float(offsets[2])],
                        [float(angles[0]), float(angles[1]), float(angles[2])],
                    ))

        if not isinstance(model_body_name, str) or not isinstance(position_file, str) or not isinstance(sensor_group, int):
            continue

        out[part] = {
            "model_body_name": model_body_name,
            "sensor_group": int(sensor_group),
            "position_file": position_file,
            "rebase": rebase,
            "include_to_model": include_to_model,
            "manual_steps": manual_steps,
        }

    return out



def load_skin_mesh_names(report_path: str) -> set[str]:
    """Return mesh asset names (without .stl) belonging to skin-covered parts."""
    names: set[str] = set()
    if not os.path.exists(report_path):
        return names
    for entry in _parse_report_json(report_path):
        mesh_files = entry.get("mesh_files", [])
        if not isinstance(mesh_files, list):
            continue
        for f in mesh_files:
            if isinstance(f, str):
                names.add(f.removesuffix(".stl"))
    return names


def parse_args() -> argparse.Namespace:
    """Parse script arguments for report validation behavior."""

    parser = argparse.ArgumentParser(
        description="Insert skin taxels into the MuJoCo model."
    )
    parser.add_argument(
        "--strict-report",
        action="store_true",
        help="Require the optimization report to pass strict schema and completeness validation before insertion.",
    )
    parser.add_argument(
        "--report-path",
        default=OPTIMIZATION_REPORT_JSON,
        help="Path to the optimization report JSON file.",
    )
    parser.add_argument(
        "--style-mode",
        choices=["material", "rgba"],
        default="material",
        help="Visual styling mode for geoms: material (default) or legacy rgba.",
    )
    parser.add_argument(
        "--taxel-site-size",
        type=float,
        default=0.005,
        help="MuJoCo site sphere radius in meters for inserted taxels (default: 0.005).",
    )
    parser.add_argument(
        "--apply-anchor-body-transform",
        action="store_true",
        help="Experimentally apply the anchor body's local pos/quat to inserted taxels in addition to the optimizer transform chain.",
    )
    parser.add_argument(
        "--use-frozen-report-points",
        action="store_true",
        help="Experimentally precompute final report taxel coordinates once per part and insert those coordinates verbatim.",
    )
    return parser.parse_args()


def load_body_local_transforms(model_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load body-local pos/quaternion transforms from a MuJoCo XML file."""

    tree = ET.parse(model_path)
    root = tree.getroot()

    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in root.iter("body"):
        name = body.attrib.get("name")
        if not name:
            continue
        pos = np.array(
            [float(v) for v in body.attrib.get("pos", "0 0 0").split()],
            dtype=float,
        )
        quat = np.array(
            [float(v) for v in body.attrib.get("quat", "1 0 0 0").split()],
            dtype=float,
        )
        transforms[name] = (pos, quat)
    return transforms


def load_frozen_report_points(
    report_path: str,
    path_to_skin: str,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Precompute final per-part taxel coordinates from the report once."""

    frozen: dict[str, list[tuple[int, np.ndarray]]] = {}
    if not os.path.exists(report_path):
        return frozen

    for entry in _parse_report_json(report_path):
        part = entry.get("part")
        position_file = entry.get("position_file")
        manual_steps_raw = entry.get("manual_steps", [])
        optimized = entry.get("optimized")
        include_to_model = bool(entry.get("include_to_model", entry.get("inlcude_to_model", True)))
        if not include_to_model:
            continue
        if not isinstance(part, str) or not isinstance(position_file, str) or not isinstance(optimized, dict):
            continue

        file_path = os.path.join(path_to_skin, position_file)
        if not os.path.isfile(file_path):
            continue

        calibration = read_calibration_data(file_path)
        taxel2repr = read_taxel2repr_data(file_path)
        taxels = validate_taxel_data(calibration, taxel2repr if taxel2repr else None)
        if bool(entry.get("rebase", False)):
            taxels = rebase_coordinate_system(taxels)

        manual_steps: list[tuple[list[float], list[float]]] = []
        if isinstance(manual_steps_raw, list):
            for step in manual_steps_raw:
                if not isinstance(step, dict):
                    continue
                offsets = step.get("offsets_m")
                angles = step.get("angles_deg")
                if isinstance(offsets, list) and isinstance(angles, list) and len(offsets) == 3 and len(angles) == 3:
                    manual_steps.append((
                        [float(offsets[0]), float(offsets[1]), float(offsets[2])],
                        [float(angles[0]), float(angles[1]), float(angles[2])],
                    ))

        delta_offsets = np.array(optimized.get("delta_offsets_m", [0.0, 0.0, 0.0]), dtype=float)
        quat_values = optimized.get("delta_quaternion_wxyz")
        angle_values = optimized.get("delta_angles_deg", [0.0, 0.0, 0.0])
        delta_quat = None
        if isinstance(quat_values, list) and len(quat_values) == 4:
            delta_quat = np.array(quat_values, dtype=float)
        delta_angles = np.array(angle_values, dtype=float)

        part_points: list[tuple[int, np.ndarray]] = []
        for pos_raw, _, idx in taxels:
            pos = np.array(pos_raw, dtype=float)
            for offsets_step, angles_step in manual_steps:
                pos = rotate_position(pos=pos, offsets=offsets_step, angle_degrees=angles_step)
            if delta_quat is not None:
                pos = apply_quaternion_transform(pos=pos, offsets=delta_offsets.tolist(), quat_wxyz=delta_quat.tolist())
            else:
                pos = rotate_position(pos=pos, offsets=delta_offsets.tolist(), angle_degrees=delta_angles.tolist())
            part_points.append((int(idx), np.array(pos, dtype=float)))

        frozen[part] = part_points

    return frozen


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


def _normalize_quaternion_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Normalize a quaternion in wxyz convention with stable identity fallback."""

    q = np.array(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _quat_wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Convert quaternion from wxyz to xyzw convention."""

    q = _normalize_quaternion_wxyz(quat_wxyz)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def apply_quaternion_transform(
    pos: np.ndarray,
    offsets: Sequence[float],
    quat_wxyz: Sequence[float],
) -> np.ndarray:
    """Apply translation then quaternion rotation to match optimizer transform convention."""

    qx, qy, qz, qw = _quat_wxyz_to_xyzw(quat_wxyz)

    # Build the equivalent rotation matrix from xyzw components.
    rot = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )

    shifted = np.array(pos, dtype=float) + np.array(offsets, dtype=float)
    return np.round(rot @ shifted, 12)


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


def _format_rgba(color: Sequence[float]) -> str:
    """Format RGBA values for MuJoCo XML attributes."""
    return " ".join(str(value) for value in color)


def _set_geom_rgba(line: str, rgba_value: str) -> str:
    """Set or inject an rgba attribute on a single-line <geom .../> tag."""
    if "<geom" not in line:
        return line
    if 'rgba="' in line:
        return re.sub(r'rgba="[^"]*"', f'rgba="{rgba_value}"', line, count=1)
    return line.replace("/>", f' rgba="{rgba_value}"/>', 1)


def _set_geom_material(line: str, material_name: str) -> str:
    """Set or inject a material attribute on a single-line <geom .../> tag."""
    if "<geom" not in line:
        return line
    if 'material="' in line:
        return re.sub(r'material="[^"]*"', f'material="{material_name}"', line, count=1)
    return line.replace("/>", f' material="{material_name}"/>', 1)


def _remove_geom_attr(line: str, attr_name: str) -> str:
    """Remove one XML attribute from a single-line <geom .../> tag if present."""
    if "<geom" not in line:
        return line
    return re.sub(rf'\s+{attr_name}="[^"]*"', "", line, count=1)


def _set_rgba_alpha(rgba_value: str, alpha: float) -> str:
    """Return an RGBA string with the same RGB values and a replaced alpha."""
    parts = rgba_value.split()
    if len(parts) != 4:
        return rgba_value
    parts[3] = str(alpha)
    return " ".join(parts)


def apply_body_part_colors(
    lines: list[str],
    skin_body_names: set[str],
    head_body_names: set[str],
    palm_body_names: set[str],
    fingertip_body_names: set[str],
    skin_plastic_body_names: set[str],
    style_mode: str,
) -> list[str]:
    """Style geoms by enclosing body using either material- or rgba-based assignment."""
    skin_mesh_names = load_skin_mesh_names(OPTIMIZATION_REPORT_JSON)
    colored_lines = list(lines)
    body_stack: list[str] = []
    head_rgba = _format_rgba(ICUB_HEAD_COLOR)
    skin_rgba = _format_rgba(ICUB_SKIN_COLOR)
    metal_rgba = _format_rgba(ICUB_METAL_PARTS_COLOR)
    black_rgba = _format_rgba(ICUB_BLACK_PARTS_COLOR)

    # Build a name index so we can detect paired *_col_0 <-> *_vis_0 geoms.
    geom_names: set[str] = set()
    for line in colored_lines:
        if "<geom" not in line:
            continue
        name_match = re.search(r'\bname="([^"]+)"', line)
        if name_match:
            geom_names.add(name_match.group(1))

    for idx, line in enumerate(colored_lines):
        open_match = re.search(r'<body\s+name="([^"]+)"', line)
        if open_match:
            body_stack.append(open_match.group(1))

        if "<geom" in line and body_stack:
            current_body = body_stack[-1]
            name_match = re.search(r'\bname="([^"]+)"', line)
            geom_name = name_match.group(1) if name_match else None
            mesh_match = re.search(r'\bmesh="([^"]+)"', line)
            geom_mesh = mesh_match.group(1) if mesh_match else None

            material_name = ICUB_MATERIAL_METAL
            if current_body in palm_body_names or current_body in fingertip_body_names:
                rgba_value = black_rgba
                material_name = ICUB_MATERIAL_BLACK
            elif current_body in head_body_names:
                rgba_value = head_rgba
                material_name = ICUB_MATERIAL_HEAD
            elif current_body in skin_plastic_body_names:
                rgba_value = skin_rgba
                material_name = ICUB_MATERIAL_SKIN_PLASTIC
            elif current_body in skin_body_names or (geom_mesh is not None and geom_mesh in skin_mesh_names):
                rgba_value = skin_rgba
                material_name = ICUB_MATERIAL_SKIN_FABRIC
            else:
                rgba_value = metal_rgba
                material_name = ICUB_MATERIAL_METAL

            # If both visual and collision geoms exist for the same segment,
            # keep visual geoms opaque and make collision geoms transparent.
            force_invisible_col = False
            if geom_name is not None and geom_name.endswith("_col_0"):
                vis_name = f"{geom_name[:-6]}_vis_0"
                if vis_name in geom_names:
                    rgba_value = _set_rgba_alpha(rgba_value, 0.0)
                    force_invisible_col = True
            elif geom_name is not None and geom_name.endswith("_vis_0"):
                col_name = f"{geom_name[:-6]}_col_0"
                if col_name in geom_names:
                    rgba_value = _set_rgba_alpha(rgba_value, 1.0)
            else:
                rgba_value = _set_rgba_alpha(rgba_value, 1.0)

            if style_mode == "material":
                if force_invisible_col:
                    material_name = ICUB_MATERIAL_COL_INVISIBLE
                styled = _set_geom_material(line, material_name)
                styled = _remove_geom_attr(styled, "rgba")
                colored_lines[idx] = styled
            else:
                colored_lines[idx] = _set_geom_rgba(line, rgba_value)

        close_count = line.count("</body")
        for _ in range(close_count):
            if body_stack:
                body_stack.pop()

    return colored_lines


def include_skin_to_mujoco_model(
    mujoco_model: str,
    path_to_skin: str,
    skin_parts: dict[str, str],
    optimized_deltas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None = None,
    report_part_metadata: dict[str, dict[str, object]] | None = None,
    style_mode: str = "material",
    taxel_site_size: float = 0.08,
    apply_anchor_body_transform: bool = False,
    frozen_report_points: dict[str, list[tuple[int, np.ndarray]]] | None = None,
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
        report_part_metadata (dict | None): Optional metadata loaded from the
            optimization report keyed by part name. When present, manual_steps,
            rebase flags, body anchors, and sensor groups are read from report.
        taxel_site_size (float): Radius of each inserted taxel site sphere in
            meters. This affects only UI appearance, not site center position.

    Returns:
        None: The function writes output to files:
            - Modified MuJoCo model saved to `MUJOCO_MODEL_OUT` path
            - Report saved to "./report_including_taxels.txt"
    """
    # read the taxel positions from the skin configuration file
    if optimized_deltas is None:
        optimized_deltas = {}
    if report_part_metadata is None:
        report_part_metadata = {}
    if frozen_report_points is None:
        frozen_report_points = {}

    body_local_transforms = load_body_local_transforms(mujoco_model) if apply_anchor_body_transform else {}

    report_by_body: dict[str, dict[str, object]] = {}
    for part_name, metadata in report_part_metadata.items():
        body_name = metadata.get("model_body_name")
        if isinstance(body_name, str):
            report_by_body[body_name] = {"part": part_name, **metadata}

    # Derive skin-colored body names from the report (auto-load if not provided by caller).
    # Fingertip bodies are always added as they are manually defined, not in the report.
    if not report_by_body:
        _auto_metadata = load_report_part_metadata(OPTIMIZATION_REPORT_JSON)
        _report_body_names: set[str] = set()
        for _meta in _auto_metadata.values():
            _body = _meta.get("model_body_name")
            if isinstance(_body, str):
                _report_body_names.add(_body)
    else:
        _report_body_names = set(report_by_body.keys())
    skin_body_names = _report_body_names | set(FINGER_BODY_TO_PART_GROUP.keys())
    # Remap taxel-attachment bodies to their visible mesh bodies for coloring.
    skin_body_names = {_SKIN_COLOR_BODY_REMAP.get(b, b) for b in skin_body_names}

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
    taxels_by_position_file = {
        position_file: taxels
        for position_file, taxels in zip(ini_files_taxels, all_taxels)
    }

    # now we can open the xml robot config file and start adding all the taxels
    with open(mujoco_model, "r") as file:
        lines = file.readlines()

    lines = apply_body_part_colors(
        lines=lines,
        skin_body_names=skin_body_names,
        head_body_names=HEAD_BODY_NAMES,
        palm_body_names=PALM_BODY_NAMES,
        fingertip_body_names=FINGERTIP_BODY_NAMES,
        skin_plastic_body_names=SKIN_PLASTIC_BODY_NAMES,
        style_mode=style_mode,
    )

    # now we iterate over the lines and add the taxels
    keep_processing = True
    add_finger_taxels = False
    line_counter = 0
    parts_to_add = []
    taxel_ids_to_add = []
    while keep_processing and line_counter < len(lines):
        print(lines[line_counter])
        if "<body name=" in lines[line_counter]:
            body_name = lines[line_counter].split('name="')[1].split('"')[0]
            add_taxels = False
            current_manual_steps: list[tuple[list[float], list[float]]] = []
            current_rebase = False
            current_body_transform: tuple[np.ndarray, np.ndarray] | None = None
            part = body_name
            if body_name in report_by_body:
                report_cfg = report_by_body[body_name]
                include_to_model = bool(report_cfg.get("include_to_model", True))
                if not include_to_model:
                    line_counter += 1
                    continue
                part = cast(str, report_cfg["part"])
                position_file = cast(str, report_cfg["position_file"])
                if position_file not in taxels_by_position_file:
                    raise ValueError(
                        f"Report references unknown position file for {part}: {position_file}"
                    )
                taxels_to_add = taxels_by_position_file[position_file]
                found_spot_to_add = False
                add_taxels = True
                sensor_group = report_cfg.get("sensor_group")
                if not isinstance(sensor_group, int):
                    raise ValueError(
                        f"Report metadata for {part} must provide integer sensor_group"
                    )
                group_counter = int(sensor_group)
                current_rebase = bool(report_cfg.get("rebase", False))
                manual_steps_raw = report_cfg.get("manual_steps", [])
                if isinstance(manual_steps_raw, list):
                    current_manual_steps = cast(list[tuple[list[float], list[float]]], manual_steps_raw)
                current_body_transform = body_local_transforms.get(body_name)
                parts_to_add.append(part)

            finger_cfg = FINGER_BODY_TO_PART_GROUP.get(body_name)
            if not add_taxels and finger_cfg is not None:
                part, group_counter = finger_cfg
                add_finger_taxels = True
                found_spot_to_add = False
                add_taxels = True
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
                    finger_taxels = FINGERTIP_TAXELS_BY_PART[part]
                    for finger_pos in finger_taxels:
                        taxel_ids.append(idx)
                        lines.insert(
                            line_counter,
                            f'{" "*identation}<site name="{part}_taxel_{idx}" size="{taxel_site_size}" type="sphere" group="{group_counter}" pos="{finger_pos[0]} {finger_pos[1]} {finger_pos[2]}" rgba="0 1 0 1"/>\n',
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
                    if not current_manual_steps:
                        raise ValueError(
                            f"Missing report-driven manual_steps for non-finger part: {part}"
                        )
                    if part in frozen_report_points:
                        for idx, pos in frozen_report_points[part]:
                            taxel_ids.append(idx)
                            lines.insert(
                                line_counter,
                                f'{" "*identation}<site name="{part}_taxel_{idx}" size="{taxel_site_size}" type="sphere" group="{group_counter}" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 1"/>\n',
                            )
                            line_counter += 1
                        line_counter -= 1
                        taxel_ids_to_add.append(taxel_ids)
                        continue
                    # Rebase only when explicitly declared by the report.
                    if current_rebase:
                        taxels_to_add = rebase_coordinate_system(taxels_to_add)
                    for taxel in taxels_to_add:
                        pos, _, idx = cast(
                            tuple[np.ndarray, np.ndarray, int], taxel)
                        pos = np.array(pos, dtype=float)
                        taxel_ids.append(idx)
                        for offsets_step, angles_step in current_manual_steps:
                            pos = rotate_position(
                                pos=pos,
                                offsets=offsets_step,
                                angle_degrees=angles_step,
                            )

                        # Apply optional optimized per-part refinement as final step.
                        if part in optimized_deltas:
                            delta_angles, delta_offsets, delta_quat = optimized_deltas[part]
                            if delta_quat is not None:
                                pos = apply_quaternion_transform(
                                    pos=pos,
                                    offsets=delta_offsets.tolist(),
                                    quat_wxyz=delta_quat.tolist(),
                                )
                            else:
                                pos = rotate_position(
                                    pos=pos,
                                    offsets=delta_offsets.tolist(),
                                    angle_degrees=delta_angles.tolist(),
                                )

                        if current_body_transform is not None:
                            body_pos, body_quat = current_body_transform
                            pos = apply_quaternion_transform(
                                pos=pos,
                                offsets=body_pos.tolist(),
                                quat_wxyz=body_quat.tolist(),
                            )

                        lines.insert(
                            line_counter,
                            f'{" "*identation}<site name="{part}_taxel_{idx}" size="{taxel_site_size}" type="sphere" group="{group_counter}" pos="{pos[0]} {pos[1]} {pos[2]}" rgba="0 1 0 1"/>\n',
                        )
                        line_counter += 1
                line_counter -= 1  # go one line back after we added the last taxel
                pass
                taxel_ids_to_add.append(taxel_ids)

        elif "<tendon>" in lines[line_counter] or "<contact>" in lines[line_counter]:
            keep_processing = False

        line_counter += 1  # we need to keep iterating over the lines

    if line_counter >= len(lines):
        raise ValueError(
            "Reached end of model XML before finding <contact> or <tendon> while inserting taxels."
        )

    # now we have to define the sensors as touch sensors
    keep_processing = True
    identation = 4
    while keep_processing and line_counter < len(lines):
        # above that we need to add each taxel using:
        # If the model already has a <sensor> section, insert <touch> entries into it.
        if "</sensor>" in lines[line_counter]:
            for part_to_add, taxel_id_list in zip(parts_to_add, taxel_ids_to_add):
                lines.insert(line_counter, "\n")
                line_counter += 1
                lines.insert(
                    line_counter, f'{" "*identation}<!--{part_to_add}-->\n')
                line_counter += 1
                for id in taxel_id_list:
                    lines.insert(
                        line_counter,
                        f'{" "*identation}<touch name="{part_to_add}_{id}" site="{part_to_add}_taxel_{id}" />\n',
                    )
                    line_counter += 1
            lines.insert(line_counter, "\n")
            keep_processing = False

        # Fallback for older models without a <sensor> section: create one before <contact>.
        elif "<contact>" in lines[line_counter]:
            lines.insert(line_counter, f'{" "*identation}<sensor>\n')
            line_counter += 1
            sensor_child_indent = identation + 4
            for part_to_add, taxel_id_list in zip(parts_to_add, taxel_ids_to_add):
                lines.insert(line_counter, "\n")
                line_counter += 1
                lines.insert(
                    line_counter, f'{" "*sensor_child_indent}<!--{part_to_add}-->\n')
                line_counter += 1
                for id in taxel_id_list:
                    lines.insert(
                        line_counter,
                        f'{" "*sensor_child_indent}<touch name="{part_to_add}_{id}" site="{part_to_add}_taxel_{id}" />\n',
                    )
                    line_counter += 1
            lines.insert(line_counter, f'{" "*identation}</sensor>\n')
            line_counter += 1
            lines.insert(line_counter, "\n")
            keep_processing = False
        pass
        line_counter += 1

    if keep_processing:
        raise ValueError(
            "Could not find <contact> or </sensor> section in model XML to insert touch sensors."
        )
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
    args = parse_args()
    report_path = args.report_path
    if args.strict_report:
        validated_entries = validate_optimization_report(report_path)
        print(
            f"Strict optimization report validation passed for {len(validated_entries)} parts from {report_path}"
        )

    optimized = load_optimized_deltas(report_path)
    metadata = load_report_part_metadata(report_path)
    if optimized:
        print(
            f"Loaded optimized deltas for {len(optimized)} parts from {report_path}"
        )
    else:
        print(
            "No optimization report found or no valid deltas loaded. Using hand-tuned transforms only."
        )
    if metadata:
        print(f"Loaded report metadata for {len(metadata)} parts.")
    else:
        print(
            "No report metadata found. Non-finger body taxels cannot be inserted; only manual finger routing remains available."
        )

    frozen_points = load_frozen_report_points(report_path, TAXEL_INI_PATH) if args.use_frozen_report_points else {}
    if args.use_frozen_report_points:
        print(f"Loaded frozen report points for {len(frozen_points)} parts.")

    include_skin_to_mujoco_model(
        MUJOCO_MODEL,
        TAXEL_INI_PATH,
        POSITIONS_FILES,
        optimized_deltas=optimized,
        report_part_metadata=metadata,
        style_mode=args.style_mode,
        taxel_site_size=float(args.taxel_site_size),
        apply_anchor_body_transform=bool(args.apply_anchor_body_transform),
        frozen_report_points=frozen_points,
    )
    print("*******************")
    print("DONE")
