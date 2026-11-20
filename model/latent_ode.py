"""Compatibility alias for :mod:`veloode_mot.models.latent_ode`."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("veloode_mot.models.latent_ode")
