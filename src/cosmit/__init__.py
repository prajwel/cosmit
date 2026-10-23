"""Cosmit: Cosmic-ray tools for AstroSat/UVIT."""

from .cosmit import (
    __version__,
    DEFAULT_THRESHOLD_FACTOR,
    WARN_FLAGGED_FRAME_PERCENT,
    CosmitError,
    CosmitPaths,
    process_events_list,
    process_observation,
)

__all__ = [
    "__version__",
    "DEFAULT_THRESHOLD_FACTOR",
    "WARN_FLAGGED_FRAME_PERCENT",
    "CosmitError",
    "CosmitPaths",
    "process_events_list",
    "process_observation",
]
