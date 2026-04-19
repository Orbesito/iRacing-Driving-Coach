from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from .corner_metrics import (
    CornerDetectionConfig,
    apply_corner_reference,
    build_best_per_corner_reference,
    build_corner_report,
    choose_reference_lap,
    compute_corner_lap_metrics,
    compute_lap_times,
    detect_main_corners,
)
from .coaching import CoachingConfig, generate_coaching_outputs
from .coaching_pdf import generate_coaching_pdf
from .derived_channels import add_derived_units
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


TConfig = TypeVar("TConfig")


@dataclass
class SessionAnalysis:
    csv_path: Path
    metadata: dict[str, str]
    units: dict[str, str]
    telemetry_df: pd.DataFrame
    parse_report: dict[str, int]
    lap_summary: pd.DataFrame
    valid_lap_ids: list[int]
    aligned_laps: pd.DataFrame
    alignment_report: dict[str, object]
    lap_times: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    lap_validity: LapValidityConfig
    corner_detection: CornerDetectionConfig
    coaching: CoachingConfig
    allow_mixed_vehicle: bool
    reference_cache_enabled: bool


def _progress(message: str) -> None:
    print(f"\n... {message}", flush=True)


def _default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        lap_validity=LapValidityConfig(),
        corner_detection=CornerDetectionConfig(),
        coaching=CoachingConfig(),
        allow_mixed_vehicle=False,
        reference_cache_enabled=True,
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataclass_config(
    section_name: str,
    section_payload: dict[str, Any] | None,
    default_value: TConfig,
) -> TConfig:
    if section_payload is None:
        return default_value
    if not isinstance(section_payload, dict):
        raise ValueError(f"Config section '{section_name}' must be a JSON object.")

    field_names = {f.name for f in fields(type(default_value))}
    unknown = set(section_payload.keys()) - field_names
    if unknown:
        raise ValueError(
            f"Unknown keys in config section '{section_name}': {', '.join(sorted(unknown))}"
        )
    return type(default_value)(**section_payload)


def _load_runtime_config(
    config_path: Path | None,
    cli_allow_mixed_vehicle: bool,
    disable_reference_cache: bool,
) -> tuple[RuntimeConfig, dict[str, Any]]:
    runtime = _default_runtime_config()
    config_payload: dict[str, Any] = {}

    if config_path is not None:
        config_payload = _read_json_file(config_path)
        if not isinstance(config_payload, dict):
            raise ValueError("Config file must contain a JSON object at top level.")

        allowed_top = {"lap_validity", "corner_detection", "coaching", "comparison"}
        unknown_top = set(config_payload.keys()) - allowed_top
        if unknown_top:
            raise ValueError(
                "Unknown top-level config keys: " + ", ".join(sorted(unknown_top))
            )

        lap_validity = _build_dataclass_config(
            "lap_validity",
            config_payload.get("lap_validity"),
            runtime.lap_validity,
        )
        corner_detection = _build_dataclass_config(
            "corner_detection",
            config_payload.get("corner_detection"),
            runtime.corner_detection,
        )
        coaching_cfg = _build_dataclass_config(
            "coaching",
            config_payload.get("coaching"),
            runtime.coaching,
        )
        comparison_payload = config_payload.get("comparison")
        if comparison_payload is not None and not isinstance(comparison_payload, dict):
            raise ValueError("Config section 'comparison' must be a JSON object.")
        comparison_payload = comparison_payload or {}
        allowed_comparison = {"allow_mixed_vehicle", "reference_cache_enabled"}
        unknown_comparison = set(comparison_payload.keys()) - allowed_comparison
        if unknown_comparison:
            raise ValueError(
                "Unknown keys in config section 'comparison': "
                + ", ".join(sorted(unknown_comparison))
            )

        runtime = RuntimeConfig(
            lap_validity=lap_validity,
            corner_detection=corner_detection,
            coaching=coaching_cfg,
            allow_mixed_vehicle=bool(comparison_payload.get("allow_mixed_vehicle", False)),
            reference_cache_enabled=bool(comparison_payload.get("reference_cache_enabled", True)),
        )

    if cli_allow_mixed_vehicle:
        runtime = replace(runtime, allow_mixed_vehicle=True)
    if disable_reference_cache:
        runtime = replace(runtime, reference_cache_enabled=False)

    return runtime, config_payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _empty_corner_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    corner_definitions = pd.DataFrame(
        columns=[
            "corner_id",
            "official_turn_number",
            "corner_name",
            "corner_start_pct",
            "brake_start_pct",
            "brake_end_pct",
            "rotation_start_pct",
            "apex_pct",
            "rotation_end_pct",
            "traction_start_pct",
            "corner_end_pct",
            "reference_apex_speed_kmh",
            "detection_activity_score",
            "detection_curvature_score",
        ]
    )
    corner_metrics = pd.DataFrame()
    corner_ranking = pd.DataFrame()
    corner_report = {
        "corner_count": 0,
        "laps_in_metrics": 0,
        "trajectory_line_feasible": False,
        "trajectory_line_note": "No aligned laps available.",
        "coaching_score_formula": "",
    }
    return corner_definitions, corner_metrics, corner_ranking, corner_report


def _print_parse_report(parse_report: dict[str, int]) -> None:
    print("\nParse summary:")
    for key, value in parse_report.items():
        print(f"  {key}: {value}")


def _validate_required_channels(df: pd.DataFrame) -> None:
    required_for_next_stage = ["SessionTime", "Lap", "LapDistPct"]
    missing_required = [col for col in required_for_next_stage if col not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required channels for deterministic lap alignment: "
            + ", ".join(missing_required)
        )


def _analyze_single_session(
    csv_path: Path,
    label: str,
    output_base_dir: str | Path = "outputs",
    lap_validity_config: LapValidityConfig | None = None,
    output_dir_override: Path | None = None,
) -> SessionAnalysis:
    print(f"\n[{label}] Loading Mu CSV: {csv_path}")
    _progress(f"[{label}] Parsing Mu CSV metadata and telemetry")
    metadata, units, df, parse_report = load_mu_csv(csv_path)
    _print_parse_report(parse_report)

    _progress(f"[{label}] Applying derived channels")
    df, derived_units = add_derived_units(df)
    units.update(derived_units)
    _validate_required_channels(df)

    _progress(f"[{label}] Classifying valid laps")
    validity_config = lap_validity_config or LapValidityConfig()
    lap_summary = build_lap_summary(df, config=validity_config)
    valid_lap_ids = get_valid_lap_ids(lap_summary)

    print(f"\n[{label}] Lap validity summary:")
    print(
        f"  total_laps_detected: {len(lap_summary)}\n"
        f"  valid_laps: {len(valid_lap_ids)}\n"
        f"  valid_lap_ids: {valid_lap_ids}"
    )

    if valid_lap_ids:
        _progress(f"[{label}] Aligning valid laps by distance")
        aligned_laps, alignment_report = align_laps_by_distance(
            df,
            lap_ids=valid_lap_ids,
            distance_step_pct=DEFAULT_DISTANCE_STEP_PCT,
        )
        _progress(f"[{label}] Computing lap times for aligned laps")
        lap_times = compute_lap_times(df, valid_lap_ids)
    else:
        aligned_laps = pd.DataFrame(columns=["lap_id", "distance_pct"])
        alignment_report = {
            "requested_lap_count": 0,
            "aligned_lap_count": 0,
            "distance_step_pct": DEFAULT_DISTANCE_STEP_PCT,
            "distance_grid_points": 0,
            "alignment_channels": [],
        }
        lap_times = pd.DataFrame(columns=["lap_id", "start_time_s", "end_time_s", "lap_time_s"])

    print(f"\n[{label}] Distance alignment summary:")
    print(f"  requested_lap_count: {alignment_report['requested_lap_count']}")
    print(f"  aligned_lap_count: {alignment_report['aligned_lap_count']}")
    print(f"  distance_step_pct: {alignment_report['distance_step_pct']}")
    print(f"  distance_grid_points: {alignment_report['distance_grid_points']}")

    if output_dir_override is not None:
        out_dir = Path(output_dir_override)
    else:
        out_dir = build_session_output_dir(
            base_dir=output_base_dir,
            source_csv=csv_path,
            metadata=metadata,
        )
    _progress(f"[{label}] Saving session and lap artifacts")
    output_paths = save_session_bundle(
        out_dir,
        telemetry_df=df,
        metadata=metadata,
        units_map=units,
        parse_report=parse_report,
    )
    lap_paths = save_lap_analysis(
        out_dir / "laps",
        lap_summary_df=lap_summary,
        aligned_laps_df=aligned_laps,
        alignment_report=alignment_report,
    )

    print(f"\n[{label}] Saved core artifacts:")
    print(f"  session_output_dir: {out_dir}")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")
    for key, value in lap_paths.items():
        print(f"  {key}: {value}")

    return SessionAnalysis(
        csv_path=csv_path,
        metadata=metadata,
        units=units,
        telemetry_df=df,
        parse_report=parse_report,
        lap_summary=lap_summary,
        valid_lap_ids=valid_lap_ids,
        aligned_laps=aligned_laps,
        alignment_report=alignment_report,
        lap_times=lap_times,
        output_dir=out_dir,
    )


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _infer_expected_corner_count(metadata: dict[str, str]) -> int | None:
    """
    Deterministic track metadata lookup for official turn counts where known.
    """
    venue = _normalise_text(metadata.get("Venue", ""))
    if "miami" in venue and "gp" in venue:
        return 19
    return None


def _build_corner_detection_config(
    metadata: dict[str, str],
    base_config: CornerDetectionConfig,
) -> CornerDetectionConfig:
    expected_count = _infer_expected_corner_count(metadata)
    if expected_count is None:
        # Generic track-agnostic defaults.
        return base_config
    # When official turn count is known, lock detector to stable numbering.
    return replace(
        base_config,
        min_corner_count=expected_count,
        max_corner_count=expected_count,
        target_corner_count=expected_count,
    )


def _assert_compatible_sessions(
    driver_metadata: dict[str, str],
    reference_metadata: dict[str, str],
    allow_mixed_vehicle: bool = False,
) -> None:
    driver_venue = _normalise_text(driver_metadata.get("Venue", ""))
    reference_venue = _normalise_text(reference_metadata.get("Venue", ""))
    if driver_venue and reference_venue and driver_venue != reference_venue:
        raise ValueError(
            "Driver and reference sessions appear to be from different venues: "
            f"'{driver_metadata.get('Venue', '')}' vs '{reference_metadata.get('Venue', '')}'. "
            "For deterministic cross-session corner comparison, both inputs must be from the same track configuration."
        )

    driver_vehicle = _normalise_text(driver_metadata.get("Vehicle", ""))
    reference_vehicle = _normalise_text(reference_metadata.get("Vehicle", ""))
    if driver_vehicle and reference_vehicle and driver_vehicle != reference_vehicle:
        if allow_mixed_vehicle:
            print(
                "\nWARNING: Mixed-vehicle comparison enabled. "
                f"Driver vehicle '{driver_metadata.get('Vehicle', '')}' vs "
                f"reference vehicle '{reference_metadata.get('Vehicle', '')}'."
            )
            return
        raise ValueError(
            "Driver and reference sessions use different vehicles: "
            f"'{driver_metadata.get('Vehicle', '')}' vs '{reference_metadata.get('Vehicle', '')}'. "
            "Use '--allow-mixed-vehicle' only if you intentionally want cross-car comparison."
        )


def _best_lap_time_seconds(lap_times_df: pd.DataFrame) -> float:
    if lap_times_df.empty or "lap_time_s" not in lap_times_df.columns:
        return float("nan")
    lap_series = pd.to_numeric(lap_times_df["lap_time_s"], errors="coerce").dropna()
    if lap_series.empty:
        return float("nan")
    return float(lap_series.min())


def _compute_optimal_lap_time_seconds(corner_lap_metrics_df: pd.DataFrame) -> float:
    """
    Deterministic theoretical optimal lap from best corner_time_s per corner.
    """
    if corner_lap_metrics_df.empty:
        return float("nan")
    if {"corner_id", "corner_time_s"} - set(corner_lap_metrics_df.columns):
        return float("nan")

    working = corner_lap_metrics_df.copy()
    working["corner_time_s"] = pd.to_numeric(working["corner_time_s"], errors="coerce")
    working = working.dropna(subset=["corner_id", "corner_time_s"])
    if working.empty:
        return float("nan")
    best_per_corner = working.groupby("corner_id", as_index=False)["corner_time_s"].min()
    if best_per_corner.empty:
        return float("nan")
    return float(best_per_corner["corner_time_s"].sum())


def _format_time_seconds(value_s: float) -> str:
    if not math.isfinite(value_s):
        return "n/a"
    return f"{value_s:.3f} s"


def _as_float_or_nan(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_csv_any(path_candidates: list[Path]) -> pd.DataFrame:
    path = _first_existing_path(path_candidates)
    if path is None:
        names = ", ".join(str(p) for p in path_candidates)
        raise FileNotFoundError(f"Could not find any expected CSV artifact: {names}")
    return pd.read_csv(path)


def _load_session_analysis_from_output_dir(csv_path: Path, out_dir: Path) -> SessionAnalysis:
    metadata = _read_json_file(out_dir / "metadata.json")
    units = _read_json_file(out_dir / "units.json")
    parse_report = _read_json_file(out_dir / "parse_report.json")

    telemetry_df = _load_csv_any(
        [out_dir / "telemetry_numeric.csv.gz", out_dir / "telemetry_numeric.csv"]
    )
    lap_summary = pd.read_csv(out_dir / "laps" / "lap_summary.csv")
    aligned_laps = _load_csv_any(
        [
            out_dir / "laps" / "aligned_laps_by_distance.csv.gz",
            out_dir / "laps" / "aligned_laps_by_distance.csv",
        ]
    )
    alignment_report = _read_json_file(out_dir / "laps" / "alignment_report.json")
    valid_lap_ids = get_valid_lap_ids(lap_summary)
    lap_times = (
        compute_lap_times(telemetry_df, valid_lap_ids)
        if valid_lap_ids
        else pd.DataFrame(columns=["lap_id", "start_time_s", "end_time_s", "lap_time_s"])
    )

    return SessionAnalysis(
        csv_path=csv_path,
        metadata=metadata,
        units=units,
        telemetry_df=telemetry_df,
        parse_report=parse_report,
        lap_summary=lap_summary,
        valid_lap_ids=valid_lap_ids,
        aligned_laps=aligned_laps,
        alignment_report=alignment_report,
        lap_times=lap_times,
        output_dir=out_dir,
    )


def _get_or_build_reference_session_cached(
    reference_csv_path: Path,
    runtime_config: RuntimeConfig,
) -> SessionAnalysis:
    _progress("Resolving cached reference-session artifacts")
    reference_hash = _sha256_file(reference_csv_path)
    cache_inputs = {
        "reference_sha256": reference_hash,
        "lap_validity": asdict(runtime_config.lap_validity),
        "distance_step_pct": DEFAULT_DISTANCE_STEP_PCT,
    }
    cache_key = _json_fingerprint(cache_inputs)[:24]
    cache_dir = Path("outputs") / "_reference_cache" / cache_key
    cache_manifest = cache_dir / "cache_manifest.json"

    if cache_manifest.exists():
        try:
            _progress("Loading reference session from shared cache")
            session = _load_session_analysis_from_output_dir(reference_csv_path, cache_dir)
            print(f"\nReference cache hit: {cache_dir}")
            return session
        except Exception as exc:
            print(f"\nReference cache load failed, rebuilding cache: {exc}")

    _progress("Building reference-session cache artifacts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = _analyze_single_session(
        reference_csv_path,
        label="Reference Session (cache build)",
        output_base_dir="outputs",
        lap_validity_config=runtime_config.lap_validity,
        output_dir_override=cache_dir,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_csv": str(reference_csv_path),
        "reference_sha256": reference_hash,
        "cache_inputs": cache_inputs,
        "cache_dir": str(cache_dir),
    }
    save_config_snapshot(cache_dir, manifest, filename="cache_manifest.json")
    print(f"\nReference cache stored: {cache_dir}")
    return session


def _build_run_config_snapshot(
    mode: str,
    driver_session: SessionAnalysis,
    runtime_config: RuntimeConfig,
    config_file_path: Path | None,
    cli_args: dict[str, Any],
    reference_session: SessionAnalysis | None = None,
) -> dict[str, Any]:
    resolved_driver_corner_cfg = _build_corner_detection_config(
        driver_session.metadata,
        runtime_config.corner_detection,
    )
    resolved_reference_corner_cfg = (
        _build_corner_detection_config(
            reference_session.metadata,
            runtime_config.corner_detection,
        )
        if reference_session is not None
        else None
    )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "python_executable": sys.executable,
        "config_file": str(config_file_path) if config_file_path is not None else None,
        "cli_args": cli_args,
        "base_config": {
            "lap_validity": asdict(runtime_config.lap_validity),
            "corner_detection": asdict(runtime_config.corner_detection),
            "coaching": asdict(runtime_config.coaching),
            "comparison": {
                "allow_mixed_vehicle": runtime_config.allow_mixed_vehicle,
                "reference_cache_enabled": runtime_config.reference_cache_enabled,
            },
        },
        "resolved_config": {
            "driver_corner_detection": asdict(resolved_driver_corner_cfg),
            "reference_corner_detection": (
                asdict(resolved_reference_corner_cfg)
                if resolved_reference_corner_cfg is not None
                else None
            ),
        },
        "inputs": {
            "driver_csv": str(driver_session.csv_path),
            "reference_csv": str(reference_session.csv_path)
            if reference_session is not None
            else None,
        },
    }


def _build_pdf_session_context(
    coached_session: SessionAnalysis,
    corner_report: dict[str, object],
    reference_session: SessionAnalysis | None = None,
) -> dict[str, object]:
    coached_best_lap_s = _as_float_or_nan(corner_report.get("coached_best_lap_time_s"))
    if not math.isfinite(coached_best_lap_s):
        coached_best_lap_s = _best_lap_time_seconds(coached_session.lap_times)
    coached_optimal_lap_s = _as_float_or_nan(corner_report.get("coached_optimal_lap_time_s"))
    coached_potential_gain_s = _as_float_or_nan(
        corner_report.get("coached_improvement_potential_s")
    )

    context: dict[str, object] = {
        "coached_driver": coached_session.metadata.get("Driver", "n/a"),
        "coached_vehicle": coached_session.metadata.get("Vehicle", "n/a"),
        "venue": coached_session.metadata.get("Venue", "n/a"),
        "session_type": coached_session.metadata.get("Session", "n/a"),
        "session_date": coached_session.metadata.get("Session Date", "n/a"),
        "session_time": coached_session.metadata.get("Session Time", "n/a"),
        "sample_rate": coached_session.metadata.get("Sample Rate", "n/a"),
        "session_duration": coached_session.metadata.get("Session Duration", "n/a"),
        "valid_laps": len(coached_session.valid_lap_ids),
        "total_laps": int(len(coached_session.lap_summary)),
        "coached_best_lap_time_s": _format_time_seconds(coached_best_lap_s),
        "coached_optimal_lap_time_s": _format_time_seconds(coached_optimal_lap_s),
        "coached_improvement_potential_s": _format_time_seconds(coached_potential_gain_s),
        "detected_corner_count": corner_report.get(
            "detected_corner_count", corner_report.get("corner_count", "n/a")
        ),
        "expected_corner_count": corner_report.get("expected_corner_count", "n/a"),
    }
    if reference_session is not None:
        reference_best_lap_s = _as_float_or_nan(corner_report.get("reference_best_lap_time_s"))
        if not math.isfinite(reference_best_lap_s):
            reference_best_lap_s = _best_lap_time_seconds(reference_session.lap_times)
        reference_optimal_lap_s = _as_float_or_nan(
            corner_report.get("reference_optimal_lap_time_s")
        )
        reference_potential_gain_s = _as_float_or_nan(
            corner_report.get("reference_improvement_potential_s")
        )
        context.update(
            {
                "reference_driver": reference_session.metadata.get("Driver", "n/a"),
                "reference_vehicle": reference_session.metadata.get("Vehicle", "n/a"),
                "reference_best_lap_time_s": _format_time_seconds(reference_best_lap_s),
                "reference_optimal_lap_time_s": _format_time_seconds(reference_optimal_lap_s),
                "reference_improvement_potential_s": _format_time_seconds(reference_potential_gain_s),
            }
        )
    return context


def _run_single_session_mode(
    session: SessionAnalysis,
    runtime_config: RuntimeConfig,
) -> None:
    print("\nRunning mode: single-session coaching analysis.")
    coached_best_lap_s = _best_lap_time_seconds(session.lap_times)
    coached_optimal_lap_s = float("nan")
    coached_improvement_potential_s = float("nan")
    if session.aligned_laps.empty or session.lap_times.empty:
        corner_definitions, corner_metrics, corner_ranking, corner_report = _empty_corner_outputs()
        corner_reference = pd.DataFrame()
    else:
        detection_lap_id = choose_reference_lap(session.lap_times)
        print(f"Corner detection lap (fastest valid lap): {detection_lap_id}")
        corner_config = _build_corner_detection_config(
            session.metadata,
            runtime_config.corner_detection,
        )
        _progress("Detecting main corners and phase boundaries")
        corner_definitions = detect_main_corners(
            aligned_laps_df=session.aligned_laps,
            reference_lap_id=detection_lap_id,
            config=corner_config,
        )
        detected_count = int(len(corner_definitions))
        expected_count = _infer_expected_corner_count(session.metadata)
        if expected_count is not None:
            print(
                f"Detected corners: {detected_count} (expected official map turns: {expected_count})"
            )
        else:
            print(f"Detected corners: {detected_count}")

        _progress("Computing per-corner metrics for all valid laps")
        raw_corner_metrics = compute_corner_lap_metrics(
            aligned_laps_df=session.aligned_laps,
            corner_definitions_df=corner_definitions,
            lap_times_df=session.lap_times,
        )
        _progress("Building best-per-corner reference profile")
        corner_reference = build_best_per_corner_reference(raw_corner_metrics)
        coached_optimal_lap_s = _compute_optimal_lap_time_seconds(raw_corner_metrics)
        if math.isfinite(coached_best_lap_s) and math.isfinite(coached_optimal_lap_s):
            coached_improvement_potential_s = max(0.0, coached_best_lap_s - coached_optimal_lap_s)
        _progress("Scoring corners against best-per-corner reference")
        corner_metrics, corner_ranking = apply_corner_reference(
            corner_lap_metrics_df=raw_corner_metrics,
            corner_reference_df=corner_reference,
            comparison_label="self_best_per_corner",
            exclude_reference_rows=True,
        )
        corner_report = build_corner_report(
            aligned_laps_df=session.aligned_laps,
            corner_definitions_df=corner_definitions,
            corner_lap_metrics_df=corner_metrics,
            reference_mode="best_per_corner_same_session",
            reference_source=str(session.csv_path),
        )
        corner_report["operation_mode"] = "single_session"
        corner_report["corner_detection_lap_id"] = int(detection_lap_id)
        corner_report["reference_note"] = (
            "Per-corner benchmark uses the best corner performance found across all valid laps, "
            "not only the globally fastest lap."
        )
        corner_report["coached_best_lap_time_s"] = (
            coached_best_lap_s if math.isfinite(coached_best_lap_s) else None
        )
        corner_report["coached_optimal_lap_time_s"] = (
            coached_optimal_lap_s if math.isfinite(coached_optimal_lap_s) else None
        )
        corner_report["coached_improvement_potential_s"] = (
            coached_improvement_potential_s
            if math.isfinite(coached_improvement_potential_s)
            else None
        )
        corner_report["optimal_lap_note"] = (
            "Theoretical optimal lap = sum of best corner_time_s across valid laps."
        )
        corner_report["expected_corner_count"] = expected_count
        corner_report["detected_corner_count"] = detected_count
        corner_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

    corner_report.setdefault(
        "coached_best_lap_time_s",
        coached_best_lap_s if math.isfinite(coached_best_lap_s) else None,
    )
    corner_report.setdefault(
        "coached_optimal_lap_time_s",
        coached_optimal_lap_s if math.isfinite(coached_optimal_lap_s) else None,
    )
    corner_report.setdefault(
        "coached_improvement_potential_s",
        coached_improvement_potential_s if math.isfinite(coached_improvement_potential_s) else None,
    )

    _progress("Saving corner coaching artifacts")
    corner_paths = save_corner_analysis(
        session.output_dir / "corners",
        corner_definitions_df=corner_definitions,
        corner_lap_metrics_df=corner_metrics,
        corner_ranking_df=corner_ranking,
        corner_report=corner_report,
        corner_reference_df=corner_reference,
    )

    print("\nSaved corner artifacts:")
    for key, value in corner_paths.items():
        print(f"  {key}: {value}")

    _progress("Building deterministic coaching layer")
    corner_coaching_df, coaching_summary = generate_coaching_outputs(
        corner_lap_metrics_df=corner_metrics,
        corner_ranking_df=corner_ranking,
        corner_report=corner_report,
        mode_name="single_session",
        config=runtime_config.coaching,
    )
    coaching_summary["coached_best_lap_time_s"] = _format_time_seconds(coached_best_lap_s)
    coaching_summary["coached_optimal_lap_time_s"] = _format_time_seconds(coached_optimal_lap_s)
    coaching_summary["coached_improvement_potential_s"] = _format_time_seconds(
        coached_improvement_potential_s
    )
    coaching_paths = save_coaching_analysis(
        session.output_dir / "coaching",
        corner_coaching_df=corner_coaching_df,
        session_summary=coaching_summary,
    )
    _progress("Generating coaching PDF report")
    try:
        pdf_context = _build_pdf_session_context(
            coached_session=session,
            corner_report=corner_report,
            reference_session=None,
        )
        coaching_pdf_path = generate_coaching_pdf(
            out_path=session.output_dir / "coaching_report.pdf",
            mode_name="single_session",
            corner_coaching_df=corner_coaching_df,
            session_summary=coaching_summary,
            corner_definitions_df=corner_definitions,
            aligned_laps_df=session.aligned_laps,
            corner_reference_df=corner_reference,
            reference_aligned_laps_df=None,
            session_context=pdf_context,
        )
    except Exception as exc:
        coaching_pdf_path = None
        print(f"\nPDF report skipped: {exc}")

    print("\nSaved coaching artifacts:")
    for key, value in coaching_paths.items():
        print(f"  {key}: {value}")
    if coaching_pdf_path is not None:
        print(f"  coaching_report_pdf: {coaching_pdf_path}")

    print("\nTop coaching-priority corners:")
    top = corner_ranking.head(5)
    if top.empty:
        print("  none")
        return
    for _, row in top.iterrows():
        print(
            f"  {row['corner_name']} (rank {int(row['coaching_priority_rank'])}) "
            f"time_loss={row['mean_time_loss_s']:.4f}s "
            f"variability={row['inconsistency_time_s']:.4f}s "
            f"score={row['coaching_relevance_score']:.4f}"
        )


def _run_vs_reference_mode(
    driver_session: SessionAnalysis,
    reference_session: SessionAnalysis,
    runtime_config: RuntimeConfig,
) -> None:
    print("\nRunning mode: driver session vs faster reference session.")
    _assert_compatible_sessions(
        driver_session.metadata,
        reference_session.metadata,
        allow_mixed_vehicle=runtime_config.allow_mixed_vehicle,
    )
    coached_best_lap_s = _best_lap_time_seconds(driver_session.lap_times)
    coached_optimal_lap_s = float("nan")
    coached_improvement_potential_s = float("nan")
    reference_best_lap_s = _best_lap_time_seconds(reference_session.lap_times)
    reference_optimal_lap_s = float("nan")
    reference_improvement_potential_s = float("nan")
    coached_corner_profile_for_plots = pd.DataFrame()

    if (
        driver_session.aligned_laps.empty
        or driver_session.lap_times.empty
        or reference_session.aligned_laps.empty
        or reference_session.lap_times.empty
    ):
        driver_corner_definitions, driver_corner_metrics, driver_corner_ranking, driver_corner_report = (
            _empty_corner_outputs()
        )
        reference_corner_profile = pd.DataFrame()
    else:
        reference_detection_lap_id = choose_reference_lap(reference_session.lap_times)
        print(f"Reference corner detection lap: {reference_detection_lap_id}")
        corner_config = _build_corner_detection_config(
            reference_session.metadata,
            runtime_config.corner_detection,
        )

        # Stable corner IDs: corner definitions are detected once on reference session
        # and then reused for both sessions.
        _progress("Detecting canonical corners from reference session")
        canonical_corner_definitions = detect_main_corners(
            aligned_laps_df=reference_session.aligned_laps,
            reference_lap_id=reference_detection_lap_id,
            config=corner_config,
        )
        detected_count = int(len(canonical_corner_definitions))
        expected_count = _infer_expected_corner_count(reference_session.metadata)
        if expected_count is not None:
            print(
                f"Detected canonical corners: {detected_count} "
                f"(expected official map turns: {expected_count})"
            )
        else:
            print(f"Detected canonical corners: {detected_count}")

        _progress("Computing reference-session per-corner metrics")
        reference_raw_metrics = compute_corner_lap_metrics(
            aligned_laps_df=reference_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            lap_times_df=reference_session.lap_times,
        )
        _progress("Building reference best-per-corner profile")
        reference_corner_profile = build_best_per_corner_reference(reference_raw_metrics)
        reference_optimal_lap_s = _compute_optimal_lap_time_seconds(reference_raw_metrics)
        if math.isfinite(reference_best_lap_s) and math.isfinite(reference_optimal_lap_s):
            reference_improvement_potential_s = max(
                0.0, reference_best_lap_s - reference_optimal_lap_s
            )

        _progress("Scoring reference session against its own profile")
        reference_metrics_scored, reference_ranking = apply_corner_reference(
            corner_lap_metrics_df=reference_raw_metrics,
            corner_reference_df=reference_corner_profile,
            comparison_label="reference_self_best_per_corner",
            exclude_reference_rows=True,
        )
        reference_report = build_corner_report(
            aligned_laps_df=reference_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            corner_lap_metrics_df=reference_metrics_scored,
            reference_mode="best_per_corner_same_session",
            reference_source=str(reference_session.csv_path),
        )
        reference_report["operation_mode"] = "reference_baseline"
        reference_report["corner_detection_lap_id"] = int(reference_detection_lap_id)
        reference_report["reference_best_lap_time_s"] = (
            reference_best_lap_s if math.isfinite(reference_best_lap_s) else None
        )
        reference_report["reference_optimal_lap_time_s"] = (
            reference_optimal_lap_s if math.isfinite(reference_optimal_lap_s) else None
        )
        reference_report["reference_improvement_potential_s"] = (
            reference_improvement_potential_s
            if math.isfinite(reference_improvement_potential_s)
            else None
        )
        reference_report["optimal_lap_note"] = (
            "Theoretical optimal lap = sum of best corner_time_s across valid laps."
        )
        reference_report["expected_corner_count"] = expected_count
        reference_report["detected_corner_count"] = detected_count
        reference_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

        _progress("Saving reference-session corner artifacts")
        save_corner_analysis(
            reference_session.output_dir / "corners",
            corner_definitions_df=canonical_corner_definitions,
            corner_lap_metrics_df=reference_metrics_scored,
            corner_ranking_df=reference_ranking,
            corner_report=reference_report,
            corner_reference_df=reference_corner_profile,
        )

        _progress("Computing driver-session per-corner metrics on canonical corners")
        driver_raw_metrics = compute_corner_lap_metrics(
            aligned_laps_df=driver_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            lap_times_df=driver_session.lap_times,
        )
        _progress("Building coached-driver best-per-corner profile for plot overlays")
        coached_corner_profile_for_plots = build_best_per_corner_reference(driver_raw_metrics)
        coached_optimal_lap_s = _compute_optimal_lap_time_seconds(driver_raw_metrics)
        if math.isfinite(coached_best_lap_s) and math.isfinite(coached_optimal_lap_s):
            coached_improvement_potential_s = max(
                0.0, coached_best_lap_s - coached_optimal_lap_s
            )
        _progress("Scoring driver session vs external reference profile")
        driver_corner_metrics, driver_corner_ranking = apply_corner_reference(
            corner_lap_metrics_df=driver_raw_metrics,
            corner_reference_df=reference_corner_profile,
            comparison_label="vs_external_reference_driver",
            exclude_reference_rows=False,
        )
        driver_corner_definitions = canonical_corner_definitions
        driver_corner_report = build_corner_report(
            aligned_laps_df=driver_session.aligned_laps,
            corner_definitions_df=driver_corner_definitions,
            corner_lap_metrics_df=driver_corner_metrics,
            reference_mode="external_session_best_per_corner",
            reference_source=str(reference_session.csv_path),
        )
        driver_corner_report["operation_mode"] = "vs_reference_session"
        driver_corner_report["corner_detection_lap_id_reference"] = int(reference_detection_lap_id)
        driver_corner_report["stable_corner_id_note"] = (
            "Corner IDs and boundaries are defined from the reference session and reused "
            "without re-detection on the driver session."
        )
        driver_corner_report["driver_csv"] = str(driver_session.csv_path)
        driver_corner_report["reference_csv"] = str(reference_session.csv_path)
        driver_corner_report["reference_session_artifacts_dir"] = str(reference_session.output_dir)
        driver_corner_report["coached_best_lap_time_s"] = (
            coached_best_lap_s if math.isfinite(coached_best_lap_s) else None
        )
        driver_corner_report["coached_optimal_lap_time_s"] = (
            coached_optimal_lap_s if math.isfinite(coached_optimal_lap_s) else None
        )
        driver_corner_report["coached_improvement_potential_s"] = (
            coached_improvement_potential_s
            if math.isfinite(coached_improvement_potential_s)
            else None
        )
        driver_corner_report["reference_best_lap_time_s"] = (
            reference_best_lap_s if math.isfinite(reference_best_lap_s) else None
        )
        driver_corner_report["reference_optimal_lap_time_s"] = (
            reference_optimal_lap_s if math.isfinite(reference_optimal_lap_s) else None
        )
        driver_corner_report["reference_improvement_potential_s"] = (
            reference_improvement_potential_s
            if math.isfinite(reference_improvement_potential_s)
            else None
        )
        driver_corner_report["optimal_lap_note"] = (
            "Theoretical optimal lap = sum of best corner_time_s across valid laps."
        )
        driver_corner_report["expected_corner_count"] = expected_count
        driver_corner_report["detected_corner_count"] = detected_count
        driver_corner_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

    driver_corner_report.setdefault(
        "coached_best_lap_time_s",
        coached_best_lap_s if math.isfinite(coached_best_lap_s) else None,
    )
    driver_corner_report.setdefault(
        "coached_optimal_lap_time_s",
        coached_optimal_lap_s if math.isfinite(coached_optimal_lap_s) else None,
    )
    driver_corner_report.setdefault(
        "coached_improvement_potential_s",
        coached_improvement_potential_s
        if math.isfinite(coached_improvement_potential_s)
        else None,
    )
    driver_corner_report.setdefault(
        "reference_best_lap_time_s",
        reference_best_lap_s if math.isfinite(reference_best_lap_s) else None,
    )
    driver_corner_report.setdefault(
        "reference_optimal_lap_time_s",
        reference_optimal_lap_s if math.isfinite(reference_optimal_lap_s) else None,
    )
    driver_corner_report.setdefault(
        "reference_improvement_potential_s",
        reference_improvement_potential_s
        if math.isfinite(reference_improvement_potential_s)
        else None,
    )

    _progress("Saving driver-vs-reference corner artifacts")
    driver_corner_paths = save_corner_analysis(
        driver_session.output_dir / "corners_vs_reference",
        corner_definitions_df=driver_corner_definitions,
        corner_lap_metrics_df=driver_corner_metrics,
        corner_ranking_df=driver_corner_ranking,
        corner_report=driver_corner_report,
        corner_reference_df=reference_corner_profile,
    )

    print("\nSaved driver-vs-reference corner artifacts:")
    for key, value in driver_corner_paths.items():
        print(f"  {key}: {value}")

    _progress("Building deterministic coaching layer (driver vs reference)")
    corner_coaching_df, coaching_summary = generate_coaching_outputs(
        corner_lap_metrics_df=driver_corner_metrics,
        corner_ranking_df=driver_corner_ranking,
        corner_report=driver_corner_report,
        mode_name="vs_reference_session",
        config=runtime_config.coaching,
    )
    coaching_summary["coached_best_lap_time_s"] = _format_time_seconds(coached_best_lap_s)
    coaching_summary["coached_optimal_lap_time_s"] = _format_time_seconds(coached_optimal_lap_s)
    coaching_summary["coached_improvement_potential_s"] = _format_time_seconds(
        coached_improvement_potential_s
    )
    coaching_summary["reference_best_lap_time_s"] = _format_time_seconds(reference_best_lap_s)
    coaching_summary["reference_optimal_lap_time_s"] = _format_time_seconds(
        reference_optimal_lap_s
    )
    coaching_summary["reference_improvement_potential_s"] = _format_time_seconds(
        reference_improvement_potential_s
    )
    coaching_paths = save_coaching_analysis(
        driver_session.output_dir / "coaching_vs_reference",
        corner_coaching_df=corner_coaching_df,
        session_summary=coaching_summary,
    )
    _progress("Generating coaching PDF report (driver vs reference)")
    try:
        pdf_context = _build_pdf_session_context(
            coached_session=driver_session,
            corner_report=driver_corner_report,
            reference_session=reference_session,
        )
        coaching_pdf_path = generate_coaching_pdf(
            out_path=driver_session.output_dir / "coaching_report.pdf",
            mode_name="vs_reference_session",
            corner_coaching_df=corner_coaching_df,
            session_summary=coaching_summary,
            corner_definitions_df=driver_corner_definitions,
            aligned_laps_df=driver_session.aligned_laps,
            corner_reference_df=reference_corner_profile,
            coached_corner_reference_df=coached_corner_profile_for_plots,
            reference_aligned_laps_df=reference_session.aligned_laps,
            session_context=pdf_context,
        )
    except Exception as exc:
        coaching_pdf_path = None
        print(f"\nPDF report skipped: {exc}")

    print("\nSaved coaching artifacts:")
    for key, value in coaching_paths.items():
        print(f"  {key}: {value}")
    if coaching_pdf_path is not None:
        print(f"  coaching_report_pdf: {coaching_pdf_path}")

    print("\nTop coaching-priority corners vs reference:")
    top = driver_corner_ranking.head(5)
    if top.empty:
        print("  none")
        return
    for _, row in top.iterrows():
        print(
            f"  {row['corner_name']} (rank {int(row['coaching_priority_rank'])}) "
            f"time_loss={row['mean_time_loss_s']:.4f}s "
            f"variability={row['inconsistency_time_s']:.4f}s "
            f"score={row['coaching_relevance_score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Mu telemetry analysis for single-session coaching or driver-vs-reference comparison."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the driver's (or self) Mu-exported CSV file.",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default=None,
        help="Optional faster-reference Mu CSV file. If provided, driver-vs-reference mode is used.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file with lap_validity, corner_detection, coaching, and comparison settings.",
    )
    parser.add_argument(
        "--allow-mixed-vehicle",
        action="store_true",
        help="Allow vs-reference comparison across different vehicles (off by default).",
    )
    parser.add_argument(
        "--disable-reference-cache",
        action="store_true",
        help="Disable shared reference-session cache and always build reference artifacts in run-local output.",
    )
    args = parser.parse_args()

    driver_csv_path = Path(args.csv_file)
    if not driver_csv_path.exists():
        raise FileNotFoundError(f"File not found: {driver_csv_path}")

    reference_csv_path = Path(args.reference_csv) if args.reference_csv else None
    if reference_csv_path is not None and not reference_csv_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_csv_path}")

    config_path = Path(args.config) if args.config else None
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"\nPython interpreter: {sys.executable}")
    runtime_config, _ = _load_runtime_config(
        config_path=config_path,
        cli_allow_mixed_vehicle=bool(args.allow_mixed_vehicle),
        disable_reference_cache=bool(args.disable_reference_cache),
    )
    print("\nConfiguration loaded:")
    print(f"  config_file: {config_path if config_path is not None else 'defaults'}")
    print(f"  allow_mixed_vehicle: {runtime_config.allow_mixed_vehicle}")
    print(f"  reference_cache_enabled: {runtime_config.reference_cache_enabled}")

    driver_session = _analyze_single_session(
        driver_csv_path,
        label="Driver/Self Session",
        output_base_dir="outputs",
        lap_validity_config=runtime_config.lap_validity,
    )

    if reference_csv_path is None:
        snapshot = _build_run_config_snapshot(
            mode="single_session",
            driver_session=driver_session,
            runtime_config=runtime_config,
            config_file_path=config_path,
            cli_args={
                "csv_file": str(driver_csv_path),
                "reference_csv": None,
                "allow_mixed_vehicle": bool(args.allow_mixed_vehicle),
                "disable_reference_cache": bool(args.disable_reference_cache),
            },
            reference_session=None,
        )
        snapshot_path = save_config_snapshot(driver_session.output_dir, snapshot)
        print(f"\nSaved config snapshot: {snapshot_path}")
        _run_single_session_mode(driver_session, runtime_config=runtime_config)
        return

    if runtime_config.reference_cache_enabled:
        reference_session = _get_or_build_reference_session_cached(
            reference_csv_path=reference_csv_path,
            runtime_config=runtime_config,
        )
    else:
        reference_output_base = driver_session.output_dir / "comparison_reference_session"
        reference_session = _analyze_single_session(
            reference_csv_path,
            label="Reference Session",
            output_base_dir=reference_output_base,
            lap_validity_config=runtime_config.lap_validity,
        )

    snapshot = _build_run_config_snapshot(
        mode="vs_reference_session",
        driver_session=driver_session,
        runtime_config=runtime_config,
        config_file_path=config_path,
        cli_args={
            "csv_file": str(driver_csv_path),
            "reference_csv": str(reference_csv_path),
            "allow_mixed_vehicle": bool(args.allow_mixed_vehicle),
            "disable_reference_cache": bool(args.disable_reference_cache),
        },
        reference_session=reference_session,
    )
    snapshot_path = save_config_snapshot(driver_session.output_dir, snapshot)
    print(f"\nSaved config snapshot: {snapshot_path}")
    _run_vs_reference_mode(
        driver_session,
        reference_session,
        runtime_config=runtime_config,
    )


if __name__ == "__main__":
    main()
