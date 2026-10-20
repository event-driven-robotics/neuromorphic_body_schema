# Scripts for the running simulation

The scripts in this folder move the robot and read its sensors while the simulation runs. `main.py` runs the simulation, and each script connects to it as a separate program. Scripts can be started, stopped and changed without restarting the simulation, and several can run at the same time.

| Script | What it does | Details |
| --- | --- | --- |
| `control/sample_pose_movement.py` | Moves the robot through poses (position control) | [control/README.md](control/README.md) |
| `control/sample_velocity_movement.py` | Moves joints at target speeds (velocity control) | [control/README.md](control/README.md) |
| `sensor/sample_proprioception.py` | Reads joint states and proprioceptive spike events | [sensor/README.md](sensor/README.md) |
| `sensor/sample_vision.py` | Reads the eye cameras and event-camera events | [sensor/README.md](sensor/README.md) |

`sim_link.py` connects the scripts to the simulation (see [How it works](#how-it-works)).

## Quick start

All commands run from the repository root, each in its own terminal.

1. Start the simulation and leave it running:
   ```
   python neuromorphic_body_schema/main.py
   ```
   It logs `Command server listening on ('127.0.0.1', 6006)`.

2. Move the robot:
   ```
   python src/control/sample_pose_movement.py --loop
   ```

3. Read its sensors:
   ```
   python src/sensor/sample_proprioception.py
   ```

Stop a script with Ctrl+C. The simulation keeps running, and the robot holds the last targets it received.

### With Docker

The same steps with Docker (build the image first, see "Running with Docker" in the main [README](../README.md)):
```
docker compose run --rm sim-gpu python neuromorphic_body_schema/main.py
docker compose run --rm sim-gpu python src/control/sample_pose_movement.py --loop
docker compose run --rm sim-gpu python src/sensor/sample_proprioception.py
```
Without a GPU, use `sim` instead of `sim-gpu`.

Each command runs in its own container. The containers use the host network, so every script reaches the simulation on `localhost:6006`, whether it runs in a container or on the host.

## Settings in `main.py`

Two flags at the top of `neuromorphic_body_schema/main.py` control the connection:
```
ENABLE_COMMAND_SERVER = True  # accept connections from these scripts
FIXED_BASE = True             # hold the floating base in place
```

**Fixed base.** The robot stands on a free joint, and it cannot keep its balance with the legs on fixed position targets: it tips over as soon as the arms move. With `FIXED_BASE = True` the base (`root_link_freejoint`) is held in place every simulation step, also when no script is connected. Set it to `False` to let the base move freely.

## Memory

The simulation uses about 1 GB of memory, and each sensor readout about 0.7 GB, because each readout loads its own copy of the model. The movement scripts need very little. If the machine runs low on memory, close other programs or run fewer readouts at once.

## How it works

`sim_link.py` has three parts:

- **`CommandServer`** runs inside `main.py`. It listens on `localhost:6006`, with a background thread for each connected script. Once per simulation step, before `mj_step`, `main.py` calls `command_server.update()`, which:
  - holds the base in place (if `FIXED_BASE`),
  - applies the latest joint targets received, with `update_joint_positions`,
  - records the current state for the scripts.
- **`SimClient`** is how a script talks to the server. Movement scripts use it to send targets and read the current state.
- **`SimMirror`** is used by the sensor readouts. It loads the same model as `main.py` and copies the running simulation's state (time, `qpos`, `qvel`, `qacc`, actuator forces) into its own `data`. The readout then reads sensors and renders cameras from that copy, so `main.py` doesn't slow down.

The scripts work in simulation time, which they read from the server. If the simulation runs slower than real time, movements slow down with it.

## Writing your own script

Moving joints with `SimClient`:
```
from src.control.joint_config import JOINTS
from src.sim_link import SimClient

with SimClient() as client:
    client.set_targets({JOINTS[16]: 1.2})   # r_elbow to 1.2 rad
    state = client.get_state()
    print(state["time"], state["position"]["r_elbow"])
```

| Method | Returns |
| --- | --- |
| `set_targets({actuator name: target})` | Sets targets, applied at the next simulation step. Returns the state. |
| `get_state()` | Simulation time, and for every actuator its target (`ctrl`) and position (`position`) |
| `get_ranges()` | `(min, max)` range of every actuator |
| `get_sim_state()` | Time and the `qpos`, `qvel`, `qacc` and `actuator_force` arrays |

`set_targets` jumps to the new target in one go. For smooth moves use `move_to_pose` or `move_to_vel`, see [control/README.md](control/README.md#using-the-functions-in-your-own-script).

Reading sensors with `SimMirror`:
```
from src.sim_link import SimClient, SimMirror

with SimClient() as client:
    mirror = SimMirror(client)
    mirror.update()   # copy the latest state into mirror.data
    print(mirror.data.time, mirror.data.joint("r_elbow").qpos[0])
```
`mirror.model` and `mirror.data` work like the `model` and `data` in `main.py`, for example with `mujoco.Renderer`.

Run your script from the repository root, or add the repository root to `sys.path` at the top of the script as the sample scripts do.
