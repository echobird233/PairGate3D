"""Compatibility alias for :mod:`veloode_mot.models.causal_motion_observer`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("veloode_mot.models.causal_motion_observer")
