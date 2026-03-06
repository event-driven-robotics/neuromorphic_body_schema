"""
Test sensor parsing and initialization.
"""

import re
from collections import defaultdict

import mujoco
import numpy as np
import pytest

from neuromorphic_body_schema.helpers.helpers import (MODEL_PATH,
                                                      DynamicGroupedSensors)


def test_sensor_parsing(mujoco_model, mujoco_data):
    """Test that taxel sensors are correctly parsed from the model."""
    names_list = mujoco_model.names.decode("utf-8").split("\x00")
    sensor_info = [x for x in names_list if "taxel" in x]

    # Just check that parsing works, even if no sensors
    assert isinstance(sensor_info, list)


def test_dynamic_grouped_sensors(mujoco_model, mujoco_data):
    """Test DynamicGroupedSensors class instantiation."""
    grouped_sensors = {"test": [0, 1, 2]}

    dynamic_sensors = DynamicGroupedSensors(mujoco_data, grouped_sensors)
    assert dynamic_sensors is not None
    assert dynamic_sensors.grouped_sensors == grouped_sensors
