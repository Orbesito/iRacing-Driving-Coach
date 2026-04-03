from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_matplotlib() -> tuple[Any, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    mpl_cache = Path("outputs") / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: matplotlib.\n"
            f"Interpreter in use: {sys.executable}\n"
            "Install with:\n"
            f'  "{sys.executable}" -m pip install matplotlib'
        ) from exc
    return plt, PdfPages


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _first_available_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _prepare_plot_channels(aligned_laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a plotting dataframe with normalized channel names without touching raw data.
    """
    plot_df = aligned_laps_df.copy()

    speed_col = _first_available_column(plot_df, ["Speed_kmh", "Speed"])
    if speed_col is None:
        plot_df["plot_speed_kmh"] = np.nan
    elif speed_col == "Speed_kmh":
        plot_df["plot_speed_kmh"] = _to_numeric(plot_df[speed_col])
    else:
        plot_df["plot_speed_kmh"] = _to_numeric(plot_df[speed_col]) * 3.6

    brake_col = _first_available_column(plot_df, ["BrakeRaw", "Brake"])
    throttle_col = _first_available_column(plot_df, ["ThrottleRaw", "Throttle"])
    steer_col = _first_available_column(plot_df, ["SteeringWheelAngle_deg", "SteeringWheelAngle"])
    yaw_col = _first_available_column(plot_df, ["YawRate_deg_s", "YawRate"])

    plot_df["plot_brake_pct"] = _to_numeric(plot_df[brake_col]) if brake_col else np.nan
    plot_df["plot_throttle_pct"] = _to_numeric(plot_df[throttle_col]) if throttle_col else np.nan
    if steer_col is None:
        plot_df["plot_steer_deg"] = np.nan
    elif steer_col == "SteeringWheelAngle_deg":
        plot_df["plot_steer_deg"] = _to_numeric(plot_df[steer_col])
    else:
        plot_df["plot_steer_deg"] = _to_numeric(plot_df[steer_col]) * 180.0 / np.pi

    if yaw_col is None:
        plot_df["plot_yaw_deg_s"] = np.nan
    elif yaw_col == "YawRate_deg_s":
        plot_df["plot_yaw_deg_s"] = _to_numeric(plot_df[yaw_col])
    else:
        plot_df["plot_yaw_deg_s"] = _to_numeric(plot_df[yaw_col]) * 180.0 / np.pi

    if "LatAccel" in plot_df.columns:
        plot_df["plot_lat_accel_g"] = _to_numeric(plot_df["LatAccel"])
    else:
        plot_df["plot_lat_accel_g"] = np.nan

    return plot_df


def _slice_corner_window(
    aligned_laps_df: pd.DataFrame,
    corner_start_pct: float,
    corner_end_pct: float,
) -> pd.DataFrame:
    segment = aligned_laps_df.loc[
        (aligned_laps_df["distance_pct"] >= float(corner_start_pct))
        & (aligned_laps_df["distance_pct"] <= float(corner_end_pct))
    ].copy()
    return segment.sort_values(["lap_id", "distance_pct"])


def _envelope(segment_df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = segment_df.pivot(index="distance_pct", columns="lap_id", values=value_col).sort_index()
    x = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    median = np.nanmedian(values, axis=1)
    q25 = np.nanquantile(values, 0.25, axis=1)
    q75 = np.nanquantile(values, 0.75, axis=1)
    return x, median, q25, q75


def _reference_trace(
    reference_df: pd.DataFrame | None,
    fallback_df: pd.DataFrame,
    reference_lap_id: int,
    corner_start_pct: float,
    corner_end_pct: float,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_df = reference_df if reference_df is not None else fallback_df
    segment = _slice_corner_window(source_df, corner_start_pct, corner_end_pct)
    ref = segment.loc[segment["lap_id"] == int(reference_lap_id), ["distance_pct", value_col]].copy()
    ref = ref.dropna().sort_values("distance_pct")
    if ref.empty:
        return np.array([]), np.array([])
    return ref["distance_pct"].to_numpy(dtype=float), ref[value_col].to_numpy(dtype=float)


def _phase_local_positions(corner_row: pd.Series) -> Dict[str, float]:
    start = float(corner_row["corner_start_pct"])
    return {
        "brake_start": float(corner_row["brake_start_pct"]) - start,
        "brake_end": float(corner_row["brake_end_pct"]) - start,
        "rotation_start": float(corner_row["rotation_start_pct"]) - start,
        "apex": float(corner_row["apex_pct"]) - start,
        "rotation_end": float(corner_row["rotation_end_pct"]) - start,
        "traction_start": float(corner_row["traction_start_pct"]) - start,
        "corner_end": float(corner_row["corner_end_pct"]) - start,
    }


def _plot_channel(
    ax: Any,
    x_local: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    ref_x_local: np.ndarray,
    ref_values: np.ndarray,
    label: str,
    y_label: str,
) -> None:
    if x_local.size == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.20)
        return
    ax.plot(x_local, median, color="#1f77b4", linewidth=1.8, label="Driver median")
    ax.fill_between(x_local, q25, q75, color="#1f77b4", alpha=0.20, label="Driver IQR")
    if ref_x_local.size and ref_values.size:
        ax.plot(ref_x_local, ref_values, color="#d62728", linewidth=1.5, linestyle="--", label="Reference")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Corner Distance (% lap, local)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def _draw_phase_markers(ax: Any, phase: Dict[str, float]) -> None:
    ax.axvspan(phase["brake_start"], phase["brake_end"], alpha=0.08, color="#ff7f0e")
    ax.axvspan(phase["rotation_start"], phase["rotation_end"], alpha=0.08, color="#9467bd")
    ax.axvspan(phase["traction_start"], phase["corner_end"], alpha=0.08, color="#2ca02c")
    ax.axvline(phase["apex"], color="black", linewidth=0.8, linestyle=":")


def generate_coaching_pdf(
    out_path: str | Path,
    mode_name: str,
    corner_coaching_df: pd.DataFrame,
    session_summary: Dict[str, object],
    corner_definitions_df: pd.DataFrame,
    aligned_laps_df: pd.DataFrame,
    corner_reference_df: pd.DataFrame,
    reference_aligned_laps_df: pd.DataFrame | None = None,
    max_corner_pages: int = 5,
) -> Path:
    """
    Generate a deterministic PDF coaching report with channel plots + coaching comments.
    """
    plt, PdfPages = _load_matplotlib()

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver_plot_df = _prepare_plot_channels(aligned_laps_df)
    reference_plot_df = (
        _prepare_plot_channels(reference_aligned_laps_df)
        if reference_aligned_laps_df is not None
        else None
    )

    channels = [
        ("plot_speed_kmh", "Speed", "km/h"),
        ("plot_brake_pct", "Brake", "%"),
        ("plot_throttle_pct", "Throttle", "%"),
        ("plot_steer_deg", "Steering", "deg"),
        ("plot_yaw_deg_s", "Yaw Rate", "deg/s"),
        ("plot_lat_accel_g", "Lateral Accel", "g"),
    ]

    with PdfPages(output_path) as pdf:
        summary_fig = plt.figure(figsize=(11.0, 8.5))
        summary_fig.suptitle("Driver Coaching Report", fontsize=16, fontweight="bold", y=0.97)
        lines = [
            f"Mode: {mode_name}",
            f"Main Entry Issue: {session_summary.get('main_entry_issue', 'n/a')}",
            f"Main Mid-Corner Issue: {session_summary.get('main_mid_issue', 'n/a')}",
            f"Main Exit Issue: {session_summary.get('main_exit_issue', 'n/a')}",
            f"Track Usage Assessment: {session_summary.get('track_usage_assessment', 'n/a')}",
            "",
            "Top 3 Priorities:",
        ]
        top_priorities = session_summary.get("top_3_priorities", [])
        for idx, item in enumerate(top_priorities, start=1):
            lines.append(
                f"{idx}. {item.get('corner_name', 'n/a')} [{item.get('primary_phase', 'n/a')}] - "
                f"{item.get('recommended_action', 'n/a')}"
            )
            lines.append(f"   Drill: {item.get('drill_focus', 'n/a')}")
            concise = item.get("coaching_summary_concise")
            if concise:
                lines.append(f"   Note: {concise}")
        summary_fig.text(0.04, 0.92, "\n".join(lines), va="top", fontsize=10, wrap=True)
        pdf.savefig(summary_fig, bbox_inches="tight")
        plt.close(summary_fig)

        if corner_coaching_df.empty:
            empty_fig = plt.figure(figsize=(11.0, 8.5))
            empty_fig.suptitle("No Corner Coaching Data", fontsize=14, fontweight="bold")
            empty_fig.text(0.05, 0.85, "No corners were available for PDF plotting.", fontsize=11)
            pdf.savefig(empty_fig, bbox_inches="tight")
            plt.close(empty_fig)
            return output_path

        top_corners = (
            corner_coaching_df.sort_values("coaching_priority_rank")
            .head(max_corner_pages)
            .reset_index(drop=True)
        )

        for _, coaching_row in top_corners.iterrows():
            corner_id = int(coaching_row["corner_id"])
            corner_name = str(coaching_row["corner_name"])

            corner_def = corner_definitions_df.loc[corner_definitions_df["corner_id"] == corner_id]
            if corner_def.empty:
                continue
            corner_row = corner_def.iloc[0]
            c_start = float(corner_row["corner_start_pct"])
            c_end = float(corner_row["corner_end_pct"])
            phase = _phase_local_positions(corner_row)

            driver_seg = _slice_corner_window(driver_plot_df, c_start, c_end)
            if driver_seg.empty:
                continue

            ref_lap_id = None
            ref_rows = corner_reference_df.loc[corner_reference_df["corner_id"] == corner_id]
            if not ref_rows.empty and "reference_corner_lap_id" in ref_rows.columns:
                ref_lap_id = int(ref_rows.iloc[0]["reference_corner_lap_id"])

            fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.5))
            axes = axes.flatten()
            fig.suptitle(
                f"{corner_name} Coaching (Priority {int(coaching_row['coaching_priority_rank'])})",
                fontsize=14,
                fontweight="bold",
            )

            for ax, (channel_col, title, unit) in zip(axes, channels):
                if channel_col not in driver_seg.columns:
                    ax.set_visible(False)
                    continue

                x, median, q25, q75 = _envelope(driver_seg, channel_col)
                x_local = x - c_start

                ref_x_local = np.array([])
                ref_values = np.array([])
                if ref_lap_id is not None:
                    ref_x, ref_y = _reference_trace(
                        reference_df=reference_plot_df,
                        fallback_df=driver_plot_df,
                        reference_lap_id=ref_lap_id,
                        corner_start_pct=c_start,
                        corner_end_pct=c_end,
                        value_col=channel_col,
                    )
                    ref_x_local = ref_x - c_start if ref_x.size else ref_x
                    ref_values = ref_y

                _plot_channel(
                    ax=ax,
                    x_local=x_local,
                    median=median,
                    q25=q25,
                    q75=q75,
                    ref_x_local=ref_x_local,
                    ref_values=ref_values,
                    label=title,
                    y_label=unit,
                )
                _draw_phase_markers(ax, phase)
                ax.legend(loc="best", fontsize=7)

            for idx in range(len(channels), len(axes)):
                axes[idx].set_visible(False)

            detailed_note = coaching_row.get("coaching_summary_detailed")
            if not isinstance(detailed_note, str) or not detailed_note.strip():
                detailed_note = (
                    f"Primary Phase: {coaching_row.get('primary_phase', 'n/a')}. "
                    f"Symptom: {coaching_row.get('symptom', 'n/a')}. "
                    f"Action: {coaching_row.get('recommended_action', 'n/a')}."
                )
            notes = (
                f"Primary Phase: {coaching_row.get('primary_phase', 'n/a')}\n"
                f"Symptom: {coaching_row.get('symptom', 'n/a')}\n"
                f"Likely Cause: {coaching_row.get('likely_cause', 'n/a')}\n"
                f"Action: {coaching_row.get('recommended_action', 'n/a')}\n"
                f"Drill: {coaching_row.get('drill_focus', 'n/a')}\n"
                f"Confidence: {coaching_row.get('confidence_level', 'n/a')} "
                f"({float(coaching_row.get('confidence_score', 0.0)):.2f})\n"
                f"Detailed Note: {detailed_note}"
            )
            fig.text(0.03, 0.02, notes, fontsize=9, va="bottom", wrap=True)
            fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.94])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_path
