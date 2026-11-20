"""Compatibility alias for :mod:`veloode_mot.tracking.tracker_kf`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("veloode_mot.tracking.tracker_kf")
