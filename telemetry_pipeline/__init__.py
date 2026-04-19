"""Core modules for deterministic Mu telemetry ingestion and preprocessing."""

from .derived_channels import add_derived_units
from .corner_metrics import (
    CornerDetectionConfig,
    apply_corner_reference,
    build_best_per_corner_reference,
    build_corner_report,
    choose_reference_lap,
    compute_corner_lap_metrics,
    compute_corner_metrics,
    compute_lap_times,
    detect_main_corners,
)
from .coaching import CoachingConfig, generate_coaching_outputs
from .coaching_pdf import generate_coaching_pdf
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
    save_config_snapshot,
    save_coaching_analysis,
    save_corner_analysis,
    save_lap_analysis,
    save_session_bundle,
)

__all__ = [
    "add_derived_units",
    "build_session_output_dir",
    "choose_reference_lap",
    "apply_corner_reference",
    "build_best_per_corner_reference",
    "build_corner_report",
    "CoachingConfig",
    "compute_corner_lap_metrics",
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
    "generate_coaching_outputs",
    "generate_coaching_pdf",
    "save_config_snapshot",
    "save_coaching_analysis",
    "save_corner_analysis",
    "save_lap_analysis",
    "save_session_bundle",
]
