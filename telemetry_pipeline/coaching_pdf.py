from __future__ import annotations

import os
import sys
import textwrap
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

    lap_dist_col = _first_available_column(plot_df, ["LapDist"])
    plot_df["plot_lap_dist_m"] = _to_numeric(plot_df[lap_dist_col]) if lap_dist_col else np.nan

    return plot_df


def _slice_corner_window(
    aligned_laps_df: pd.DataFrame,
    window_start_pct: float,
    window_end_pct: float,
) -> pd.DataFrame:
    segment = aligned_laps_df.loc[
        (aligned_laps_df["distance_pct"] >= float(window_start_pct))
        & (aligned_laps_df["distance_pct"] <= float(window_end_pct))
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


def _distance_mapping_m(segment_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "plot_lap_dist_m" not in segment_df.columns:
        return np.array([]), np.array([])

    pivot = (
        segment_df.pivot(index="distance_pct", columns="lap_id", values="plot_lap_dist_m")
        .sort_index()
    )
    x_pct = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    x_m = np.nanmedian(values, axis=1)
    finite = np.isfinite(x_m)
    if finite.sum() < 2:
        return np.array([]), np.array([])
    return x_pct[finite], x_m[finite]


def _pct_to_m(values_pct: np.ndarray, map_pct: np.ndarray, map_m: np.ndarray) -> np.ndarray:
    if values_pct.size == 0:
        return values_pct
    if map_pct.size < 2 or map_m.size < 2:
        return np.array([], dtype=float)
    return np.interp(values_pct, map_pct, map_m, left=np.nan, right=np.nan)


def _reference_trace(
    reference_df: pd.DataFrame | None,
    fallback_df: pd.DataFrame,
    reference_lap_id: int,
    window_start_pct: float,
    window_end_pct: float,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_df = reference_df if reference_df is not None else fallback_df
    segment = _slice_corner_window(source_df, window_start_pct, window_end_pct)
    ref_cols = ["distance_pct", value_col]
    if "plot_lap_dist_m" in segment.columns:
        ref_cols.append("plot_lap_dist_m")
    ref = segment.loc[segment["lap_id"] == int(reference_lap_id), ref_cols].copy()
    ref = ref.dropna().sort_values("distance_pct")
    if ref.empty:
        return np.array([]), np.array([])

    if "plot_lap_dist_m" in ref.columns:
        x_m = ref["plot_lap_dist_m"].to_numpy(dtype=float)
        if np.isfinite(x_m).sum() >= 2:
            return x_m, ref[value_col].to_numpy(dtype=float)

    return ref["distance_pct"].to_numpy(dtype=float), ref[value_col].to_numpy(dtype=float)


def _phase_positions(corner_row: pd.Series) -> Dict[str, float]:
    return {
        "corner_start": float(corner_row["corner_start_pct"]),
        "brake_start": float(corner_row["brake_start_pct"]),
        "brake_end": float(corner_row["brake_end_pct"]),
        "rotation_start": float(corner_row["rotation_start_pct"]),
        "apex": float(corner_row["apex_pct"]),
        "rotation_end": float(corner_row["rotation_end_pct"]),
        "traction_start": float(corner_row["traction_start_pct"]),
        "corner_end": float(corner_row["corner_end_pct"]),
    }


def _plot_window_from_phases(phase: Dict[str, float]) -> tuple[float, float]:
    """
    Build an absolute-lap plotting window with extra room on corner exit.
    """
    core_start = min(phase["corner_start"], phase["brake_start"], phase["rotation_start"])
    core_end = max(phase["corner_end"], phase["traction_start"], phase["rotation_end"])
    core_span = max(0.5, core_end - core_start)

    pre_margin = max(0.5, 0.15 * core_span)
    post_margin = max(1.2, 0.40 * core_span)

    start = max(0.0, core_start - pre_margin)
    end = min(100.0, core_end + post_margin)
    if end - start < 2.0:
        end = min(100.0, start + 2.0)
    return start, end


def _plot_channel(
    ax: Any,
    x_axis: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    ref_x_axis: np.ndarray,
    ref_values: np.ndarray,
    label: str,
    y_label: str,
    x_label: str,
    show_legend: bool,
) -> None:
    if x_axis.size == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.20)
        return
    ax.plot(x_axis, median, color="#1f77b4", linewidth=1.8, label="Driver median")
    ax.fill_between(x_axis, q25, q75, color="#1f77b4", alpha=0.20, label="Driver IQR")
    if ref_x_axis.size and ref_values.size:
        ax.plot(ref_x_axis, ref_values, color="#d62728", linewidth=1.5, linestyle="--", label="Reference")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=7)


def _draw_phase_markers(ax: Any, phase: Dict[str, float]) -> None:
    if np.isfinite(phase["brake_start"]) and np.isfinite(phase["brake_end"]):
        ax.axvspan(phase["brake_start"], phase["brake_end"], alpha=0.08, color="#ff7f0e")
    if np.isfinite(phase["rotation_start"]) and np.isfinite(phase["rotation_end"]):
        ax.axvspan(phase["rotation_start"], phase["rotation_end"], alpha=0.08, color="#9467bd")
    if np.isfinite(phase["traction_start"]) and np.isfinite(phase["corner_end"]):
        ax.axvspan(phase["traction_start"], phase["corner_end"], alpha=0.08, color="#2ca02c")
    if np.isfinite(phase["apex"]):
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
    session_context: Dict[str, object] | None = None,
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
        session_context = session_context or {}
        coached_driver = session_context.get("coached_driver", "n/a")
        coached_vehicle = session_context.get("coached_vehicle", "n/a")
        venue = session_context.get("venue", "n/a")
        session_type = session_context.get("session_type", "n/a")
        session_date = session_context.get("session_date", "n/a")
        session_time = session_context.get("session_time", "n/a")
        sample_rate = session_context.get("sample_rate", "n/a")
        session_duration = session_context.get("session_duration", "n/a")
        valid_laps = session_context.get("valid_laps", "n/a")
        total_laps = session_context.get("total_laps", "n/a")
        best_lap_s = session_context.get("coached_best_lap_time_s", "n/a")
        expected_corners = session_context.get("expected_corner_count", "n/a")
        detected_corners = session_context.get("detected_corner_count", "n/a")
        reference_driver = session_context.get("reference_driver", "n/a")
        reference_vehicle = session_context.get("reference_vehicle", "n/a")
        reference_best_lap_s = session_context.get("reference_best_lap_time_s", "n/a")

        lines = [
            f"Mode: {mode_name}",
            "",
            "Session Context:",
            f"- Circuit: {venue}",
            f"- Session Type: {session_type}",
            f"- Date/Time: {session_date} {session_time}",
            f"- Sample Rate: {sample_rate}",
            f"- Session Duration: {session_duration}",
            f"- Coached Driver: {coached_driver}",
            f"- Coached Car: {coached_vehicle}",
            f"- Laps Used: {valid_laps} valid / {total_laps} total",
            f"- Coached Best Lap: {best_lap_s}",
            f"- Corner Model: detected {detected_corners}, expected {expected_corners}",
        ]
        if mode_name == "vs_reference_session":
            lines.extend(
                [
                    f"- Reference Driver: {reference_driver}",
                    f"- Reference Car: {reference_vehicle}",
                    f"- Reference Best Lap: {reference_best_lap_s}",
                    f"- Pairing: {coached_driver} (coached) vs {reference_driver} (reference)",
                ]
            )
        lines.extend(
            [
                "",
            f"Main Entry Issue: {session_summary.get('main_entry_issue', 'n/a')}",
            f"Main Mid-Corner Issue: {session_summary.get('main_mid_issue', 'n/a')}",
            f"Main Exit Issue: {session_summary.get('main_exit_issue', 'n/a')}",
            f"Track Usage Assessment: {session_summary.get('track_usage_assessment', 'n/a')}",
            "",
            "Graph Guide:",
            "- Orange background = braking phase.",
            "- Purple background = rotation / mid-corner phase.",
            "- Green background = traction / exit phase.",
            "- Vertical dotted line = apex point.",
            "- Driver median = median trace across valid laps.",
            "- Driver IQR = interquartile range (25th to 75th percentile) across valid laps.",
            "- X axis uses lap distance in meters when LapDist is available.",
            "",
            "Top 3 Priorities:",
            ]
        )
        top_priorities = session_summary.get("top_3_priorities", [])
        for idx, item in enumerate(top_priorities, start=1):
            practice_focus = item.get("practice_focus", item.get("drill_focus", "n/a"))
            lines.append(
                f"{idx}. {item.get('corner_name', 'n/a')} [{item.get('primary_phase', 'n/a')}] - "
                f"{item.get('recommended_action', 'n/a')}"
            )
            lines.append(f"   Practice Focus: {practice_focus}")
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
            phase = _phase_positions(corner_row)
            w_start, w_end = _plot_window_from_phases(phase)

            driver_seg = _slice_corner_window(driver_plot_df, w_start, w_end)
            if driver_seg.empty:
                continue

            map_pct, map_m = _distance_mapping_m(driver_seg)
            use_meter_axis = map_pct.size >= 2
            if use_meter_axis:
                phase_plot = {
                    key: float(np.interp(np.array([value]), map_pct, map_m, left=np.nan, right=np.nan)[0])
                    for key, value in phase.items()
                }
                x_label = "Lap Distance (m)"
                x_min = float(np.interp(np.array([w_start]), map_pct, map_m, left=np.nan, right=np.nan)[0])
                x_max = float(np.interp(np.array([w_end]), map_pct, map_m, left=np.nan, right=np.nan)[0])
                if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                    x_min = float(map_m.min())
                    x_max = float(map_m.max())
            else:
                phase_plot = phase
                x_label = "Lap Distance (% total lap)"
                x_min, x_max = w_start, w_end

            ref_lap_id = None
            ref_rows = corner_reference_df.loc[corner_reference_df["corner_id"] == corner_id]
            if not ref_rows.empty and "reference_corner_lap_id" in ref_rows.columns:
                ref_lap_id = int(ref_rows.iloc[0]["reference_corner_lap_id"])

            fig = plt.figure(figsize=(15.5, 10.5))
            gs = fig.add_gridspec(
                4,
                2,
                height_ratios=[1.0, 1.0, 1.0, 0.42],
                left=0.04,
                right=0.99,
                top=0.92,
                bottom=0.05,
                wspace=0.10,
                hspace=0.32,
            )
            axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
            fig.suptitle(
                f"{corner_name} Coaching (Priority {int(coaching_row['coaching_priority_rank'])})",
                fontsize=14,
                fontweight="bold",
            )

            for ax, (channel_col, title, unit) in zip(axes, channels):
                if channel_col not in driver_seg.columns:
                    ax.set_visible(False)
                    continue

                x_pct, median, q25, q75 = _envelope(driver_seg, channel_col)
                x_axis = x_pct
                if use_meter_axis:
                    x_m = _pct_to_m(x_pct, map_pct, map_m)
                    if x_m.size and np.isfinite(x_m).sum() >= 2:
                        x_axis = x_m

                ref_x_axis = np.array([])
                ref_values = np.array([])
                if ref_lap_id is not None:
                    ref_x, ref_y = _reference_trace(
                        reference_df=reference_plot_df,
                        fallback_df=driver_plot_df,
                        reference_lap_id=ref_lap_id,
                        window_start_pct=w_start,
                        window_end_pct=w_end,
                        value_col=channel_col,
                    )
                    if use_meter_axis:
                        ref_x_numeric = np.asarray(ref_x, dtype=float)
                        # If reference x is still in percent-space, map it to meters.
                        if ref_x_numeric.size and np.nanmax(ref_x_numeric) <= 100.0:
                            ref_x_m = _pct_to_m(ref_x_numeric, map_pct, map_m)
                            if ref_x_m.size and np.isfinite(ref_x_m).sum() >= 2:
                                ref_x_axis = ref_x_m
                            else:
                                ref_x_axis = ref_x_numeric
                        else:
                            ref_x_axis = ref_x_numeric
                    else:
                        ref_x_axis = np.asarray(ref_x, dtype=float)
                    ref_values = ref_y

                _plot_channel(
                    ax=ax,
                    x_axis=x_axis,
                    median=median,
                    q25=q25,
                    q75=q75,
                    ref_x_axis=ref_x_axis,
                    ref_values=ref_values,
                    label=title,
                    y_label=unit,
                    x_label=x_label,
                    show_legend=True,
                )
                _draw_phase_markers(ax, phase_plot)
                ax.set_xlim(x_min, x_max)

            for idx in range(len(channels), len(axes)):
                axes[idx].set_visible(False)

            detailed_note = coaching_row.get("coaching_summary_detailed")
            if not isinstance(detailed_note, str) or not detailed_note.strip():
                detailed_note = (
                    f"Primary Phase: {coaching_row.get('primary_phase', 'n/a')}. "
                    f"Symptom: {coaching_row.get('symptom', 'n/a')}. "
                    f"Action: {coaching_row.get('recommended_action', 'n/a')}."
                )
            reference_label = (
                "Reference driver"
                if mode_name == "vs_reference_session"
                else "Self best-per-corner"
            )
            ref_lap_text = (
                f"{reference_label} lap {ref_lap_id}"
                if ref_lap_id is not None
                else f"{reference_label} lap n/a"
            )
            text_ax = fig.add_subplot(gs[3, :])
            text_ax.axis("off")
            wrapped_notes = textwrap.fill(f"Coaching Summary: {detailed_note}", width=220)
            wrapped_ref = textwrap.fill(f"Reference Segment Used: {ref_lap_text}", width=220)
            text_ax.text(
                0.0,
                0.98,
                wrapped_ref,
                fontsize=9,
                va="top",
                ha="left",
                transform=text_ax.transAxes,
            )
            text_ax.text(
                0.0,
                0.65,
                wrapped_notes,
                fontsize=9,
                va="top",
                ha="left",
                transform=text_ax.transAxes,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_path
