# Movement scripts

Two sample scripts that move the robot in the running simulation:

| Script | Control | Function |
| --- | --- | --- |
| `sample_pose_movement.py` | Position: moves joints to target angles | `move_to_pose` |
| `sample_velocity_movement.py` | Velocity: moves joints at target speeds | `move_to_vel` |

Both read their moves from `joint_config.py`. Start the simulation first, see [src/README.md](../README.md#quick-start). All commands run from the repository root.

## Pose movement

Plays `SEQUENCE`: raises the right arm and waves, closes and opens the right hand, and turns the head left, right and down. Each move eases in and out from the current targets to the pose, then holds for 0.5 s. The script prints each pose as it starts (`Moving to right_arm_up`).

**Run:**
```
python src/control/sample_pose_movement.py
```
```
docker compose run --rm sim-gpu python src/control/sample_pose_movement.py
```

**Options:**
- `--loop`: repeat the sequence until Ctrl+C.
- `--pose NAME`: move to one pose only, e.g. `--pose right_arm_up`.

## Velocity movement

Moves to `home`, then plays `VELOCITY_SEQUENCE`: turns the head left and right, bends and straightens the right elbow, and moves the shoulder and elbow together. The script prints each step (`Moving neck_yaw +0.30 rad/s for 2.0 s`).

**Run:**
```
python src/control/sample_velocity_movement.py
```
```
docker compose run --rm sim-gpu python src/control/sample_velocity_movement.py
```

**Options:**
- `--loop`: repeat the sequence until Ctrl+C.

The robot's actuators are position actuators, so the velocity is given through the targets. Every 10 ms or so, each target moves on to `start + velocity × elapsed time`, and the joint follows it. A target stops at its joint limit, so a later move in the other direction starts right away. Velocity changes take effect at once, with no easing.

## Config file: `joint_config.py`

### Joint numbers

`JOINTS` gives every actuator a number. The numbers follow the actuator order of the model, so `JOINTS[i]` is the actuator behind `data.ctrl[i]`. The sensor readouts use the same numbers.

The names are **actuator** names, which are not always joint names. The leg actuators end in `_motor`, and some head and hand actuators drive a tendon (several coupled joints) instead of a single joint.

All values are in **radians**. Directions in this model (the robot faces the world −x axis):
- `*_shoulder_pitch` negative raises the arm forward.
- `*_shoulder_roll` positive lifts the arm out to the side.
- `neck_yaw` positive turns the head to the robot's left.
- `neck_pitch` negative tilts the head down.

| # | Actuator | Part | Drives | Min (rad) | Max (rad) |
| --- | --- | --- | --- | --- | --- |
| 0 | `torso_pitch` | torso | joint | -0.384 | 1.466 |
| 1 | `torso_roll` | torso | joint | -0.681 | 0.681 |
| 2 | `torso_yaw` | torso | joint | -1.030 | 1.030 |
| 3 | `neck_pitch` | head | joint | -0.698 | 0.384 |
| 4 | `neck_roll` | head | joint | -0.349 | 0.349 |
| 5 | `neck_yaw` | head | joint | -0.873 | 0.873 |
| 6 | `eyes_tilt` | head | joint | -0.524 | 0.524 |
| 7 | `eyes_version` | head | tendon | -0.785 | 0.785 |
| 8 | `eyes_vergence` | head | tendon | -0.785 | 0.785 |
| 9 | `l_shoulder_pitch` | shoulders | joint | -1.667 | 0.087 |
| 10 | `l_shoulder_roll` | shoulders | joint | 0.000 | 2.806 |
| 11 | `l_shoulder_yaw` | shoulders | joint | -0.646 | 1.745 |
| 12 | `r_shoulder_pitch` | shoulders | joint | -1.667 | 0.087 |
| 13 | `r_shoulder_roll` | shoulders | joint | 0.000 | 2.806 |
| 14 | `r_shoulder_yaw` | shoulders | joint | -0.646 | 1.745 |
| 15 | `l_elbow` | elbows | joint | 0.096 | 1.850 |
| 16 | `r_elbow` | elbows | joint | 0.096 | 1.850 |
| 17 | `l_wrist_prosup` | wrists | joint | -0.873 | 0.873 |
| 18 | `l_wrist_pitch` | wrists | joint | -1.134 | 0.175 |
| 19 | `l_wrist_yaw` | wrists | joint | -0.436 | 0.436 |
| 20 | `r_wrist_prosup` | wrists | joint | -0.873 | 0.873 |
| 21 | `r_wrist_pitch` | wrists | joint | -1.134 | 0.175 |
| 22 | `r_wrist_yaw` | wrists | joint | -0.436 | 0.436 |
| 23 | `l_hand_finger` | left hand | tendon | 0.000 | 1.047 |
| 24 | `l_thumb_oppose` | left hand | joint | 0.000 | 1.571 |
| 25 | `l_thumb_proximal` | left hand | joint | 0.000 | 1.571 |
| 26 | `l_thumb_distal` | left hand | tendon | 0.000 | 2.966 |
| 27 | `l_index_proximal` | left hand | joint | 0.000 | 1.571 |
| 28 | `l_index_distal` | left hand | tendon | 0.000 | 2.966 |
| 29 | `l_middle_proximal` | left hand | joint | 0.000 | 1.571 |
| 30 | `l_middle_distal` | left hand | tendon | 0.000 | 2.966 |
| 31 | `l_pinky` | left hand | tendon | 0.000 | 4.360 |
| 32 | `r_hand_finger` | right hand | tendon | 0.000 | 1.047 |
| 33 | `r_thumb_oppose` | right hand | joint | 0.000 | 1.571 |
| 34 | `r_thumb_proximal` | right hand | joint | 0.000 | 1.571 |
| 35 | `r_thumb_distal` | right hand | tendon | 0.000 | 2.966 |
| 36 | `r_index_proximal` | right hand | joint | 0.000 | 1.571 |
| 37 | `r_index_distal` | right hand | tendon | 0.000 | 2.966 |
| 38 | `r_middle_proximal` | right hand | joint | 0.000 | 1.571 |
| 39 | `r_middle_distal` | right hand | tendon | 0.000 | 2.966 |
| 40 | `r_pinky` | right hand | tendon | 0.000 | 4.360 |
| 41 | `l_hip_pitch_motor` | hips | joint | -0.785 | 2.339 |
| 42 | `l_hip_roll_motor` | hips | joint | -0.349 | 2.094 |
| 43 | `l_hip_yaw_motor` | hips | joint | -1.396 | 1.396 |
| 44 | `r_hip_pitch_motor` | hips | joint | -0.785 | 2.339 |
| 45 | `r_hip_roll_motor` | hips | joint | -0.349 | 2.094 |
| 46 | `r_hip_yaw_motor` | hips | joint | -1.396 | 1.396 |
| 47 | `l_knee_motor` | knees | joint | -2.164 | 0.070 |
| 48 | `r_knee_motor` | knees | joint | -2.164 | 0.070 |
| 49 | `l_ankle_pitch_motor` | ankles | joint | -0.873 | 0.349 |
| 50 | `l_ankle_roll_motor` | ankles | joint | -0.436 | 0.436 |
| 51 | `r_ankle_pitch_motor` | ankles | joint | -0.873 | 0.349 |
| 52 | `r_ankle_roll_motor` | ankles | joint | -0.436 | 0.436 |

The ranges are the `ctrlrange` values in `neuromorphic_body_schema/models/icub_v2_full_body_improved_contact_sensors.xml`. MuJoCo limits any target outside its range to the nearest end.

### Poses and `SEQUENCE` (pose movement)

A pose maps joint numbers to target angles. Joints that are not listed keep their current target, so a pose only needs the joints that should move. `SEQUENCE` lists `(pose name, seconds to reach the pose)`:
```
POSES = {
    ...
    "left_arm_up": {
        9: -0.8,   # l_shoulder_pitch
        10: 1.5,   # l_shoulder_roll
        11: -0.6,  # l_shoulder_yaw
        15: 1.4,   # l_elbow
    },
}

SEQUENCE = [
    ("home", 2.0),
    ("left_arm_up", 2.0),
    ("home", 2.0),
]
```

### `VELOCITY_SEQUENCE` (velocity movement)

Each step is `({joint number: velocity in rad/s}, seconds)`. The sequence starts from `home`:
```
VELOCITY_SEQUENCE = [
    ({5: 0.3}, 2.0),             # neck_yaw: turn the head left by 0.6 rad
    ({5: -0.3}, 4.0),            # neck_yaw: turn it right, past the centre
    ...
    ({12: -0.4, 16: 0.3}, 2.0),  # r_shoulder_pitch and r_elbow together
]
```

`tests/test_control_config.py` checks the config against the model: joint numbers, every pose target, and that `VELOCITY_SEQUENCE` stays inside the joint ranges when started from `home`.

## Using the functions in your own script

```
from src.control.sample_pose_movement import move_to_pose
from src.control.sample_velocity_movement import move_to_vel
from src.control.joint_config import POSES
from src.sim_link import SimClient

with SimClient() as client:
    move_to_pose(client, POSES["right_arm_up"], duration=2.0)
    move_to_pose(client, {16: 1.2}, duration=1.0)           # r_elbow only
    move_to_vel(client, {5: 0.3}, duration=2.0)             # neck_yaw at +0.3 rad/s
    move_to_vel(client, {12: -0.4, 16: 0.3}, duration=2.0)  # two joints at once
```

Both functions take joint numbers and work in seconds of simulation time. To set targets directly, see [Writing your own script](../README.md#writing-your-own-script).
