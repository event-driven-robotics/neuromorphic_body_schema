"""
sample_proprioception.py

Description:
Reads the proprioception of the joints in sensor_config.PROPRIOCEPTION_JOINTS
from a simulation that is already running. Start main.py first, then run this
script. Each readout gives two things per joint:
- the joint state: position, velocity, acceleration and torque (read_joint_states)
- the spike events of the event-driven proprioception (ICubProprioception)
It prints both as a table, and shows one window per joint with the spike raster
and a table of that joint's state.

Usage (from the repository root):
    python src/sensor/sample_proprioception.py
    python src/sensor/sample_proprioception.py --headless    # print only, no windows

"""

# Add the project root to the path for direct script execution
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import time

import cv2
import numpy as np
from neuromorphic_body_schema.helpers.ed_prop import (
    ICubProprioception, make_proprioception_event_frame)
from neuromorphic_body_schema.helpers.helpers import HEIGHT, WIDTH
from src.sim_link import SimClient, SimMirror
from src.sensor.sensor_config import PROPRIOCEPTION_JOINTS, PROPRIOCEPTION_MAX_FREQ

READ_PERIOD = 0.01  # sec, time between readouts
PRINT_PERIOD = 0.5  # sec, time between printed tables
PANEL_WIDTH = 230  # pixels, joint-state table to the right of the spike raster


def read_joint_states(data) -> dict:
    """
    Reads the state of every joint in PROPRIOCEPTION_JOINTS.

    Args:
        data (mujoco.MjData): MuJoCo data object.

    Returns:
        dict: Dictionary mapping joint numbers (int) to a dictionary with
              "name", "position" (rad), "velocity" (rad/s),
              "acceleration" (rad/s^2) and "torque" (Nm, from the joint's actuator).
    """
    return {
        number: {
            "name": name,
            "position": float(data.joint(name).qpos[0]),
            "velocity": float(data.joint(name).qvel[0]),
            "acceleration": float(data.joint(name).qacc[0]),
            "torque": float(data.actuator(name).force[0]),
        }
        for number, name in PROPRIOCEPTION_JOINTS.items()
    }


def print_table(sim_time: float, joint_states: dict, event_counts: dict) -> None:
    print(f"\nt = {sim_time:.3f} s")
    print(f"{'#':>3}  {'joint':<18}{'position':>10}{'velocity':>10}"
          f"{'accel.':>10}{'torque':>9}{'events':>8}")
    for number, state in joint_states.items():
        print(f"{number:>3}  {state['name']:<18}{state['position']:>10.3f}"
              f"{state['velocity']:>10.3f}{state['acceleration']:>10.2f}"
              f"{state['torque']:>9.3f}{event_counts[number]:>8}")
    print(f"{'':23}{'rad':>10}{'rad/s':>10}{'rad/s^2':>10}{'Nm':>9}")


def draw_joint_window(raster: np.ndarray, number: int, state: dict, spikes: int) -> np.ndarray:
    """
    Puts a table of the joint's state to the right of its spike raster.

    Args:
        raster (np.ndarray): Spike raster from make_proprioception_event_frame.
        number (int): Joint number.
        state (dict): The joint's entry from read_joint_states.
        spikes (int): Number of spike events of the joint since the start.

    Returns:
        np.ndarray: uint8 image with the raster on the left and the table on the right.
    """
    # uint8: cv2.putText in OpenCV 5 only draws on 8-bit images
    panel = np.zeros((HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, HEIGHT), (128, 128, 128), 1)
    rows = [
        (f"#{number} {state['name']}", ""),
        ("position", f"{state['position']:.3f} rad"),
        ("velocity", f"{state['velocity']:.3f} rad/s"),
        ("accel.", f"{state['acceleration']:.2f} rad/s/s"),
        ("torque", f"{state['torque']:.3f} Nm"),
        ("spikes", f"{spikes}"),
    ]
    for i, (label, value) in enumerate(rows):
        y = 25 + i * 30
        cv2.putText(panel, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, value, (95, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # the raster is float in [0, 1]
    return np.hstack([(raster * 255).astype(np.uint8), panel])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read proprioception from a running simulation.")
    parser.add_argument("--headless", action="store_true",
                        help="print only, no spike windows")
    args = parser.parse_args()

    try:
        client = SimClient()
    except ConnectionRefusedError:
        sys.exit("No simulation found. Start main.py first.")

    with client:
        mirror = SimMirror(client)
        names = list(PROPRIOCEPTION_JOINTS.values())
        proprioception = ICubProprioception(
            mirror.model,
            {name: dict(PROPRIOCEPTION_MAX_FREQ) for name in names},
            show_proprioception=False,  # this script draws the windows itself
        )
        # start the spike clocks now instead of at sim time 0, which would
        # produce a burst of events for all the time already simulated
        for esim in proprioception.esim:
            esim.time_of_last_spike[:] = mirror.data.time

        event_counts = {number: 0 for number in PROPRIOCEPTION_JOINTS}
        spike_totals = {number: 0 for number in PROPRIOCEPTION_JOINTS}
        rasters = {number: np.zeros((HEIGHT, WIDTH, 3)) for number in PROPRIOCEPTION_JOINTS}
        last_print = time.perf_counter()
        try:
            while True:
                sim_time = mirror.update()
                # joint state: {joint number: {"name", "position", "velocity", ...}}
                joint_states = read_joint_states(mirror.data)
                # spike events: one list of (neuron_id, timestamp, polarity) per joint,
                # in the order of PROPRIOCEPTION_JOINTS
                events = dict(zip(
                    PROPRIOCEPTION_JOINTS,
                    proprioception.update_proprioception(
                        time=sim_time, data=mirror.data),  # expects seconds
                ))
                # use joint_states and events here, e.g. to record them
                for number, joint_events in events.items():
                    event_counts[number] += len(joint_events)
                    spike_totals[number] += len(joint_events)

                if not args.headless:
                    for number, joint_events in events.items():
                        rasters[number] = make_proprioception_event_frame(
                            rasters[number], sim_time, np.array(joint_events))
                        cv2.imshow(
                            f"{number}: {PROPRIOCEPTION_JOINTS[number]}",
                            draw_joint_window(rasters[number], number,
                                              joint_states[number], spike_totals[number]))
                    cv2.waitKey(1)

                if time.perf_counter() - last_print >= PRINT_PERIOD:
                    print_table(sim_time, joint_states, event_counts)
                    event_counts = {number: 0 for number in PROPRIOCEPTION_JOINTS}
                    last_print = time.perf_counter()
                time.sleep(READ_PERIOD)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
