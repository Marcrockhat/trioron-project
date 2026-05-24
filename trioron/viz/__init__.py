"""Visualization — observable growth.  See spec §7."""
from .recorder import Recorder
from .snapshot import Snapshot, capture_full, capture_delta
from .detect import detect_all, detect_from_directory
from .export import export_html
