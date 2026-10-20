"""
joint_config.py

Description:
Configuration for the sample movements in sample_pose_movement.py and
sample_velocity_movement.py.

- JOINTS maps a joint number to the name of the MuJoCo actuator that drives it.
  The numbers follow the actuator order of the model, so JOINTS[i] is the
  actuator behind data.ctrl[i].
- POSES defines named target poses as {joint number: target in radians}.
  Joints not listed in a pose keep their current target.
- SEQUENCE is the order in which the demo visits the poses, as
  (pose name, seconds to reach the pose).
- VELOCITY_SEQUENCE is the demo for velocity control (sample_velocity_movement.py), as
  ({joint number: velocity in rad/s}, seconds to move at that velocity).

Directions in this model (the robot faces the world -x axis):
- r_/l_shoulder_pitch negative raises the arm forward.
- r_/l_shoulder_roll positive lifts the arm out to the side.
- neck_yaw positive turns the head to the robot's left.
- neck_pitch negative tilts the head down.
"""

JOINTS: dict[int, str] = {
    # torso
    0: "torso_pitch",
    1: "torso_roll",
    2: "torso_yaw",
    # head
    3: "neck_pitch",
    4: "neck_roll",
    5: "neck_yaw",
    6: "eyes_tilt",
    7: "eyes_version",  # tendon: both eyes pan together
    8: "eyes_vergence",  # tendon: eyes pan in opposite directions
    # shoulders
    9: "l_shoulder_pitch",
    10: "l_shoulder_roll",
    11: "l_shoulder_yaw",
    12: "r_shoulder_pitch",
    13: "r_shoulder_roll",
    14: "r_shoulder_yaw",
    # elbows
    15: "l_elbow",
    16: "r_elbow",
    # wrists
    17: "l_wrist_prosup",
    18: "l_wrist_pitch",
    19: "l_wrist_yaw",
    20: "r_wrist_prosup",
    21: "r_wrist_pitch",
    22: "r_wrist_yaw",
    # left hand (the *_distal, finger and pinky actuators drive tendons)
    23: "l_hand_finger",
    24: "l_thumb_oppose",
    25: "l_thumb_proximal",
    26: "l_thumb_distal",
    27: "l_index_proximal",
    28: "l_index_distal",
    29: "l_middle_proximal",
    30: "l_middle_distal",
    31: "l_pinky",
    # right hand
    32: "r_hand_finger",
    33: "r_thumb_oppose",
    34: "r_thumb_proximal",
    35: "r_thumb_distal",
    36: "r_index_proximal",
    37: "r_index_distal",
    38: "r_middle_proximal",
    39: "r_middle_distal",
    40: "r_pinky",
    # hips
    41: "l_hip_pitch_motor",
    42: "l_hip_roll_motor",
    43: "l_hip_yaw_motor",
    44: "r_hip_pitch_motor",
    45: "r_hip_roll_motor",
    46: "r_hip_yaw_motor",
    # knees
    47: "l_knee_motor",
    48: "r_knee_motor",
    # ankles
    49: "l_ankle_pitch_motor",
    50: "l_ankle_roll_motor",
    51: "r_ankle_pitch_motor",
    52: "r_ankle_roll_motor",
}

POSES: dict[str, dict[int, float]] = {
    # arms relaxed beside the body, head straight
    "home": {
        3: 0.0,  # neck_pitch
        5: 0.0,  # neck_yaw
        9: -0.3,  # l_shoulder_pitch
        10: 0.4,  # l_shoulder_roll
        11: 0.0,  # l_shoulder_yaw
        12: -0.3,  # r_shoulder_pitch
        13: 0.4,  # r_shoulder_roll
        14: 0.0,  # r_shoulder_yaw
        15: 0.6,  # l_elbow
        16: 0.6,  # r_elbow
    },
    # right arm forward and up
    "right_arm_forward": {
        12: -1.3,  # r_shoulder_pitch
        13: 0.4,  # r_shoulder_roll
        16: 1.4,  # r_elbow
    },
    # right arm out to the side with the forearm pointing up, ready to wave
    "right_arm_up": {
        12: -0.8,  # r_shoulder_pitch
        13: 1.5,  # r_shoulder_roll
        14: -0.6,  # r_shoulder_yaw
        16: 1.4,  # r_elbow
    },
    "wave_out": {16: 0.9},  # r_elbow
    "wave_in": {16: 1.8},  # r_elbow
    "close_right_hand": {
        32: 0.0,  # r_hand_finger
        33: 1.2,  # r_thumb_oppose
        34: 0.4,  # r_thumb_proximal
        35: 1.0,  # r_thumb_distal
        36: 1.2,  # r_index_proximal
        37: 2.0,  # r_index_distal
        38: 1.2,  # r_middle_proximal
        39: 2.0,  # r_middle_distal
        40: 3.5,  # r_pinky
    },
    "open_right_hand": {
        32: 0.0,  # r_hand_finger
        33: 0.0,  # r_thumb_oppose
        34: 0.0,  # r_thumb_proximal
        35: 0.0,  # r_thumb_distal
        36: 0.0,  # r_index_proximal
        37: 0.0,  # r_index_distal
        38: 0.0,  # r_middle_proximal
        39: 0.0,  # r_middle_distal
        40: 0.0,  # r_pinky
    },
    "look_left": {5: 0.6},  # neck_yaw
    "look_right": {5: -0.6},  # neck_yaw
    "look_down": {3: -0.4, 5: 0.0},  # neck_pitch, neck_yaw
    "look_ahead": {3: 0.0, 5: 0.0},  # neck_pitch, neck_yaw
}

SEQUENCE: list[tuple[str, float]] = [
    ("home", 2.0),
    ("right_arm_forward", 2.0),
    ("right_arm_up", 2.0),
    ("wave_out", 0.8),
    ("wave_in", 0.8),
    ("wave_out", 0.8),
    ("wave_in", 0.8),
    ("home", 2.0),
    ("close_right_hand", 1.5),
    ("open_right_hand", 1.5),
    ("look_left", 1.0),
    ("look_right", 1.5),
    ("look_down", 1.0),
    ("look_ahead", 1.0),
]

# starts from "home"; each step moves the listed joints at constant velocity
VELOCITY_SEQUENCE: list[tuple[dict[int, float], float]] = [
    ({5: 0.3}, 2.0),  # neck_yaw: turn the head left by 0.6 rad
    ({5: -0.3}, 4.0),  # neck_yaw: turn it right, past the centre
    ({5: 0.3}, 2.0),  # neck_yaw: back to the centre
    ({16: 0.5}, 1.5),  # r_elbow: bend by 0.75 rad
    ({16: -0.5}, 1.5),  # r_elbow: straighten again
    ({12: -0.4, 16: 0.3}, 2.0),  # r_shoulder_pitch and r_elbow together
    ({12: 0.4, 16: -0.3}, 2.0),  # and back
]
