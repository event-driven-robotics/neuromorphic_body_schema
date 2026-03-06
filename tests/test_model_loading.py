"""
Test model loading and simulation initialization.
"""
import pytest
import mujoco
import numpy as np
from neuromorphic_body_schema.helpers.helpers import MODEL_PATH


def test_model_path_exists():
    """Test that the model XML file exists."""
    import os
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_model_loading(mujoco_model):
    """Test that the MuJoCo model loads successfully."""
    assert mujoco_model is not None


def test_data_initialization(mujoco_model, mujoco_data):
    """Test that simulation data initializes correctly."""
    assert mujoco_data is not None
    assert len(mujoco_data.qpos) > 0
    assert len(mujoco_data.ctrl) > 0


def test_timestep_configuration(mujoco_model, mujoco_data):
    """Test that timestep can be configured."""
    mujoco_model.opt.timestep = 0.001
    assert mujoco_model.opt.timestep == 0.001


def test_forward_kinematics(mujoco_model, mujoco_data):
    """Test basic forward kinematics."""
    mujoco_data.qpos.fill(0.0)
    mujoco.mj_forward(mujoco_model, mujoco_data)
    
    # Check that forward kinematics produces valid positions
    assert np.all(np.isfinite(mujoco_data.xpos)), "Forward kinematics produced non-finite positions"


def test_simulation_step(mujoco_model, mujoco_data):
    """Test that simulation can step forward."""
    initial_time = mujoco_data.time
    mujoco.mj_step(mujoco_model, mujoco_data)
    
    assert mujoco_data.time > initial_time, "Simulation time did not advance"

