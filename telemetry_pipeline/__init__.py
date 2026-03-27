"""Core modules for deterministic Mu telemetry ingestion and preprocessing."""

from .derived_channels import add_derived_units
from .corner_metrics import (
    CornerDetectionConfig,
    choose_reference_lap,
    compute_corner_metrics,
    compute_lap_times,
    detect_main_corners,
)
from .io_mu_csv import load_mu_csv
from .lap_processing import (
    DEFAULT_DISTANCE_STEP_PCT,
    LapValidityConfig,
    align_laps_by_distance,
    build_lap_summary,
    get_valid_lap_ids,
)
from .persistence import (
    build_session_output_dir,
    save_corner_analysis,
    save_lap_analysis,
    save_session_bundle,
)

__all__ = [
    "add_derived_units",
    "build_session_output_dir",
    "choose_reference_lap",
    "compute_corner_metrics",
    "compute_lap_times",
    "CornerDetectionConfig",
    "DEFAULT_DISTANCE_STEP_PCT",
    "detect_main_corners",
    "LapValidityConfig",
    "align_laps_by_distance",
    "build_lap_summary",
    "get_valid_lap_ids",
    "load_mu_csv",
    "save_corner_analysis",
    "save_lap_analysis",
    "save_session_bundle",
]
