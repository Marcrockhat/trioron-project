"""Roles — neuron vs. astrocyte distinction.  See spec §2.11."""
from __future__ import annotations

import enum


class CellRole(enum.IntEnum):
    NEURON = 0
    ASTROCYTE = 1
