"""
Test module imports to catch missing dependencies early.
"""
import pytest


def test_core_module_imports():
    """Test that core modules can be imported."""
    try:
        import mujoco
        import numpy as np
        from neuromorphic_body_schema.helpers.helpers import (
            MODEL_PATH, DynamicGroupedSensors, init_POV
        )
        from neuromorphic_body_schema.helpers.robot_controller import (
            check_joints, update_joint_positions
        )
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_helper_module_imports():
    """Test that helper modules can be imported."""
    try:
        from neuromorphic_body_schema.helpers import (
            helpers, robot_controller, draw_pads, ik_solver
        )
    except ImportError as e:
        pytest.fail(f"Failed to import helper modules: {e}")


def test_mujoco_availability():
    """Test that mujoco is installed and accessible."""
    import mujoco
    assert hasattr(mujoco, 'MjModel')
    assert hasattr(mujoco, 'MjData')
    assert hasattr(mujoco, 'mj_step')
