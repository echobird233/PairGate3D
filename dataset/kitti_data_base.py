"""Compatibility alias for :mod:`veloode_mot.datasets.kitti_data_base`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("veloode_mot.datasets.kitti_data_base")
