"""
sensor_config.py

Description:
Configuration for the sample readouts in src/sensor.

- PROPRIOCEPTION_JOINTS maps a joint number to the joint to read. The numbers are
  the same as in src/control/joint_config.JOINTS, so a joint has one number for
  moving it and for reading it. Only joints whose actuator has the same name as
  the joint can be used: numbers 0-6 and 9-22 (torso, head, arms, wrists).
- PROPRIOCEPTION_MAX_FREQ sets the maximum firing rate of each neuron type.
- CAMERAS maps a camera number to the name of a MuJoCo camera in the model.
"""

# the joints moved by the sample movements in src/control
PROPRIOCEPTION_JOINTS: dict[int, str] = {
    5: "neck_yaw",
    12: "r_shoulder_pitch",
    13: "r_shoulder_roll",
    16: "r_elbow",
}

PROPRIOCEPTION_MAX_FREQ: dict[str, float] = {
    "position_max_freq": 1000.0,  # Hz
    "velocity_max_freq": 1000.0,
    "load_max_freq": 1000.0,
    "limits_max_freq": 1000.0,
}

CAMERAS: dict[int, str] = {
    0: "l_eye_camera",
    1: "r_eye_camera",
}
