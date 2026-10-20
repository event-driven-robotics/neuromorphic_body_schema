# Sensor readouts

Two sample scripts that read the robot's sensors from the running simulation:

| Script | Reads |
| --- | --- |
| `sample_proprioception.py` | Joint state (position, velocity, acceleration, torque) and proprioceptive spike events |
| `sample_vision.py` | Eye camera frames and event-camera events |

Both read their settings from `sensor_config.py`. Start the simulation first, see [src/README.md](../README.md#quick-start). To see the sensors change, run a movement script at the same time, e.g. `python src/control/sample_pose_movement.py --loop`. All commands run from the repository root.

Each readout uses about 0.7 GB of memory (see [Memory](../README.md#memory)).

## Proprioception

For each joint in `PROPRIOCEPTION_JOINTS`, reads the joint state and the spike events of the event-driven proprioception (`ICubProprioception` in `neuromorphic_body_schema/helpers/ed_prop.py`).

**Run:**
```
python src/sensor/sample_proprioception.py
```
```
docker compose run --rm sim-gpu python src/sensor/sample_proprioception.py
```

**Options:**
- `--headless`: print only, no windows.

**Printed** every 0.5 s:
```
t = 12.345 s
  #  joint               position  velocity    accel.   torque  events
  5  neck_yaw               0.600     0.000      0.00    0.121    2103
 12  r_shoulder_pitch      -0.800     0.012      0.35   -4.512    1650
 ...
                              rad     rad/s   rad/s^2       Nm
```
`events` counts the spike events since the previous table.

**Windows** (without `--headless`), one per joint:
- On the left, the spike raster of the joint's 8 neurons over the last 10 s, one row each: position +/−, velocity +/−, load +/−, lower limit, upper limit.
- On the right, a table of the joint's state and its total number of spikes.

**In the code**, each loop of `main()` has both as dicts keyed by joint number, at the comment `# use joint_states and events here`:
- `joint_states`, from `read_joint_states(data)`:
  ```
  {16: {"name": "r_elbow", "position": 1.104, "velocity": 0.611,
        "acceleration": 3.2, "torque": 2.053},
   ...}
  ```
  `position`, `velocity` and `acceleration` are the joint's `qpos`, `qvel` and `qacc`. `torque` is the force of the joint's position actuator.
- `events`: a list of `(neuron_id, timestamp, polarity)` spike events per joint.

All values come from the running simulation, copied over each loop. They are the simulation's exact values, without sensor noise. The spike encoder's "load" input is the joint acceleration, not the torque.

## Vision

For each camera in `CAMERAS`, renders the frame and converts the change between frames into events with the event-camera simulator (`CameraEventSimulator` in `neuromorphic_body_schema/helpers/ed_cam.py`).

**Run:**
```
python src/sensor/sample_vision.py
```
```
docker compose run --rm sim-gpu python src/sensor/sample_vision.py
```

**Options:**
- `--headless`: print only, no windows. In Docker without a display, also select a headless renderer:
  ```
  docker compose run --rm -e MUJOCO_GL=egl sim-gpu python src/sensor/sample_vision.py --headless
  ```
  Without a GPU, use `-e MUJOCO_GL=osmesa sim`.

**Printed** every 0.5 s:
```
t =   12.345 s  0: l_eye_camera   5321 events  1: r_eye_camera   4987 events
```

**Windows** (without `--headless`), two per camera:
- the RGB frame,
- the events since the previous frame: ON events red, OFF events blue.

Frames are 320 × 240, rendered about every 30 ms of real time (`FRAME_PERIOD`). Each event is a row `(x, y, timestamp in ns, polarity)`, and the events cover the simulation time between two frames.

## Config file: `sensor_config.py`

```
PROPRIOCEPTION_JOINTS = {
    5: "neck_yaw",
    12: "r_shoulder_pitch",
    13: "r_shoulder_roll",
    16: "r_elbow",
}

CAMERAS = {
    0: "l_eye_camera",
    1: "r_eye_camera",
}
```

- **Joints:** the numbers are the same as in `src/control/joint_config.py`, so a joint has one number for moving it and for reading it. Only joints whose actuator has the same name as the joint can be read: numbers 0–6 (torso, neck, `eyes_tilt`) and 9–22 (shoulders, elbows, wrists). The hand, eye-tendon and leg actuators are named differently from their joints.
- **`PROPRIOCEPTION_MAX_FREQ`:** the maximum firing rate of each neuron type, in Hz.
- **Cameras:** any camera in the model. To add one, put a `<camera>` in the model XML (see "Vision sensors" in the main README) and add its name here.

`tests/test_sensor_config.py` checks the config against the model.

## Writing your own readout

Use `SimMirror` to get a local copy of the running simulation, see [Writing your own script](../README.md#writing-your-own-script).
