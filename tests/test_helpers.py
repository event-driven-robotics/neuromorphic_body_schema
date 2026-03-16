"""
Test helper functions and utilities.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from neuromorphic_body_schema.helpers.helpers import (FIG_PATH, MODEL_PATH,
                                                      TRIANGLE_INI_PATH,
                                                      DynamicGroupedSensors,
                                                      init_POV)


def test_model_path_is_string():
    """Test that MODEL_PATH is a valid string."""
    assert isinstance(MODEL_PATH, str)
    assert len(MODEL_PATH) > 0


def test_model_path_is_absolute():
    """Test that MODEL_PATH is an absolute path."""
    assert Path(MODEL_PATH).is_absolute()


def test_fig_path_is_absolute():
    """Test that FIG_PATH is a valid absolute path."""
    assert isinstance(FIG_PATH, str)
    assert Path(FIG_PATH).is_absolute()


def test_init_pov_function():
    """Test that init_POV can be called on a viewer."""


    # Create a minimal mock viewer object with opt.sitegroup
    import numpy as np
    class MockOpt:
        def __init__(self):
            # Simulate 5 site groups (as in MuJoCo default)
            self.sitegroup = np.ones(5, dtype=int)

    class MockViewer:
        class Camera:
            azimuth = 0
            distance = 0
            elevation = 0
            lookat = None

        def __init__(self):
            self.cam = self.Camera()
            self.opt = MockOpt()

    viewer = MockViewer()
    result = init_POV(viewer)

    assert viewer.cam.azimuth == -4.5
    assert viewer.cam.distance == 2
    assert viewer.cam.elevation == -16
    assert viewer.cam.lookat is not None


def test_dynamic_grouped_sensors_initialization(mujoco_data):
    """Test DynamicGroupedSensors initialization."""
    grouped_sensors = {"test_group": [0, 1, 2]}
    sensors = DynamicGroupedSensors(mujoco_data, grouped_sensors)

    assert sensors.data is not None
    assert sensors.grouped_sensors == grouped_sensors
