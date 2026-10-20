"""
sample_velocity_movement.py

Description:
Moves the iCub in a simulation that is already running, by velocity control:
each move runs joints at target speeds for a given time. Start main.py first,
then run this script to play joint_config.VELOCITY_SEQUENCE, starting from the
"home" pose. Stopping the script leaves the simulation running.

For position control, see sample_pose_movement.py.

Usage (from the repository root):
    python src/control/sample_velocity_movement.py           # sequence once
    python src/control/sample_velocity_movement.py --loop    # repeat until Ctrl+C

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

from src.control.joint_config import JOINTS, POSES, VELOCITY_SEQUENCE
from src.control.sample_pose_movement import SEND_PERIOD, move_to_pose
from src.sim_link import SimClient


def move_to_vel(client: SimClient, velocities: dict, duration: float) -> None:
    """
    Moves joints at constant velocities for `duration` seconds of simulation time.

    The robot has position actuators, so the velocity is given through the targets:
    each target moves at the requested velocity (start + velocity * elapsed time),
    and the joint follows it. A target stops at its joint limit.

    Args:
        client (SimClient): Connection to the running simulation.
        velocities (dict): Dictionary mapping joint numbers (int) to velocities in rad/s (float).
        duration (float): Simulation time in seconds to move.
    """
    velocities = {JOINTS[index]: value for index, value in velocities.items()}
    ranges = client.get_ranges()
    state = client.get_state()
    start = {name: state["ctrl"][name] for name in velocities}
    start_time = state["time"]

    elapsed = 0.0
    while elapsed < duration:
        elapsed = min(state["time"] - start_time, duration)
        targets = {}
        for name, velocity in velocities.items():
            low, high = ranges[name]
            # stop at the limit, so a later move in the other direction starts right away
            targets[name] = min(max(start[name] + velocity * elapsed, low), high)
        state = client.set_targets(targets)
        time.sleep(SEND_PERIOD)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move the iCub at target velocities in a running simulation.")
    parser.add_argument("--loop", action="store_true", help="repeat until Ctrl+C")
    args = parser.parse_args()

    try:
        client = SimClient()
    except ConnectionRefusedError:
        sys.exit("No simulation found. Start main.py first.")

    with client:
        try:
            while True:
                print("Moving to home")
                move_to_pose(client, POSES["home"], 2.0)
                for velocities, duration in VELOCITY_SEQUENCE:
                    moves = ", ".join(f"{JOINTS[i]} {v:+.2f} rad/s"
                                      for i, v in velocities.items())
                    print(f"Moving {moves} for {duration} s")
                    move_to_vel(client, velocities, duration)
                if not args.loop:
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
