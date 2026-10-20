"""
sim_link.py

Description:
Connection between the running simulation (main.py) and separate scripts,
such as the movement scripts in src/control and the readouts in src/sensor, so they
can be started and stopped without restarting the simulation.

- CommandServer runs inside main.py. It accepts connections on localhost, one
  background thread per client. Once per simulation step, update() applies the
  latest targets received and records the current state.
- SimClient is used by the scripts to send actuator targets and read the state.
- SimMirror keeps a local copy of the simulation for readout scripts.

Requests other than "ranges" and "sim_state" are answered with the current state:
    {"time": sim time in s,
     "ctrl": {actuator name: current target},
     "position": {actuator name: joint position or tendon length}}

Uses multiprocessing.connection from the standard library, so no extra
dependencies are needed.

"""

import logging
import threading
from multiprocessing.connection import Client, Listener
from typing import Callable, Optional

import mujoco
from neuromorphic_body_schema.helpers.helpers import MODEL_PATH
from neuromorphic_body_schema.helpers.robot_controller import update_joint_positions

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_forward = getattr(mujoco, "mj_forward")

BASE_JOINT = "root_link_freejoint"
ADDRESS = ("localhost", 6006)
AUTHKEY = b"icub-mujoco"


def make_base_holder(model, data) -> Callable[[], None]:
    """
    Returns a function that keeps the floating base at its current pose.

    Call the returned function before every mj_step.

    Args:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.

    Returns:
        Callable[[], None]: Function that resets the base position and velocity.
    """
    joint = model.joint(BASE_JOINT)
    qpos_adr, dof_adr = joint.qposadr[0], joint.dofadr[0]
    base_qpos = data.qpos[qpos_adr:qpos_adr + 7].copy()

    def hold_base() -> None:
        data.qpos[qpos_adr:qpos_adr + 7] = base_qpos
        data.qvel[dof_adr:dof_adr + 6] = 0.0

    return hold_base


class CommandServer:
    """
    Receives actuator targets from movement scripts and applies them in the simulation loop.

    Args:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.
        fixed_base (bool): Hold the floating base in place. Default is True.
        address (tuple): (host, port) to listen on. Default is ADDRESS.
    """

    def __init__(self, model, data, fixed_base: bool = True, address: tuple = ADDRESS):
        self.data = data
        self.names = [model.actuator(i).name for i in range(model.nu)]
        self.ranges = {
            name: tuple(float(x) for x in model.actuator(name).ctrlrange)
            for name in self.names
        }
        self.hold_base = make_base_holder(model, data) if fixed_base else None
        self._lock = threading.Lock()
        self._pending = {}
        self._state = None
        self._sim_state = None
        self._record_state()

        self.address = None
        try:
            self._listener = Listener(address, authkey=AUTHKEY)
        except OSError as e:
            self._listener = None
            logging.warning(f"Command server not started on {address}: {e}")
            return
        self.address = self._listener.address
        threading.Thread(target=self._serve, daemon=True).start()
        logging.info(f"Command server listening on {self.address}")

    def update(self) -> None:
        """
        Applies the latest received targets and records the state.

        Call once per simulation step, before mj_step.
        """
        if self.hold_base is not None:
            self.hold_base()
        with self._lock:
            pending, self._pending = self._pending, {}
        # same call as in a simulation loop that moves the robot itself
        update_joint_positions(self.data, pending)
        self._record_state()

    def _record_state(self) -> None:
        state = {
            "time": float(self.data.time),
            "ctrl": {n: float(self.data.actuator(n).ctrl[0]) for n in self.names},
            # actuator length is the joint position (or tendon length) the actuator acts on
            "position": {n: float(self.data.actuator(n).length[0]) for n in self.names},
        }
        # full MuJoCo state for SimMirror
        sim_state = {
            "time": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "qacc": self.data.qacc.copy(),
            "actuator_force": self.data.actuator_force.copy(),
        }
        with self._lock:
            self._state = state
            self._sim_state = sim_state

    def _serve(self) -> None:
        """Accepts clients and answers each one in its own thread."""
        while True:
            try:
                conn = self._listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve_client, args=(conn,), daemon=True).start()

    def _serve_client(self, conn) -> None:
        logging.info("Client connected")
        with conn:
            while True:
                try:
                    command, payload = conn.recv()
                except (EOFError, OSError):
                    break
                conn.send(self._handle(command, payload))
        logging.info("Client disconnected")

    def _handle(self, command: str, payload) -> tuple:
        if command == "set":
            unknown = set(payload) - set(self.ranges)
            if unknown:
                return ("error", f"Unknown actuators: {sorted(unknown)}")
            with self._lock:
                self._pending.update(payload)
        elif command == "ranges":
            return ("ok", self.ranges)
        elif command == "sim_state":
            with self._lock:
                return ("ok", self._sim_state)
        elif command != "state":
            return ("error", f"Unknown command '{command}'")
        with self._lock:
            return ("ok", self._state)


class SimClient:
    """
    Talks to the CommandServer of a running simulation.

    Args:
        address (tuple): (host, port) of the server. Default is ADDRESS.

    Raises:
        ConnectionRefusedError: If no simulation is running with the command server enabled.
    """

    def __init__(self, address: tuple = ADDRESS):
        self._conn = Client(address, authkey=AUTHKEY)

    def _request(self, command: str, payload=None):
        self._conn.send((command, payload))
        status, result = self._conn.recv()
        if status == "error":
            raise KeyError(result)
        return result

    def set_targets(self, targets: dict) -> dict:
        """
        Sets actuator targets; they are applied at the next simulation step.

        Args:
            targets (dict): Dictionary mapping actuator names (str) to targets in radians (float).

        Returns:
            dict: The current state (see module description).
        """
        return self._request("set", targets)

    def get_state(self) -> dict:
        """Returns the current state (see module description)."""
        return self._request("state")

    def get_ranges(self) -> dict:
        """Returns a dictionary mapping actuator names to their (min, max) ctrlrange."""
        return self._request("ranges")

    def get_sim_state(self) -> dict:
        """Returns the time and the qpos, qvel, qacc and actuator_force arrays."""
        return self._request("sim_state")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SimMirror:
    """
    Local copy of the running simulation for readout scripts.

    Loads the same model and copies the state of the running simulation into its
    own `data`, so sensors can be read and cameras rendered with the usual MuJoCo
    calls, without slowing down main.py.

    Args:
        client (SimClient): Connection to the running simulation.
        model_path (str): Model to load; must be the one main.py runs. Default is MODEL_PATH.
    """

    def __init__(self, client: SimClient, model_path: str = MODEL_PATH):
        self.client = client
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.update()

    def update(self) -> float:
        """
        Copies the latest state of the running simulation into `data`.

        Returns:
            float: The simulation time in seconds.
        """
        state = self.client.get_sim_state()
        self.data.time = state["time"]
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        mj_forward(self.model, self.data)  # body and camera poses for this qpos
        # mj_forward recomputes these locally; keep the values of the running simulation
        self.data.qacc[:] = state["qacc"]
        self.data.actuator_force[:] = state["actuator_force"]
        return self.data.time
