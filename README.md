# iRacing Driving Coach

A Python-based telemetry analysis and automated driver-coaching tool for iRacing.

The software processes Mu-exported telemetry, compares laps or drivers, identifies where lap time is being lost through braking, rotation and traction, and automatically generates a structured coaching report.

## What It Does

- Filters and aligns valid laps using lap-distance telemetry.
- Detects and analyses individual corners.
- Evaluates braking, rotation and traction performance.
- Identifies time loss and driver inconsistency.
- Generates evidence-based coaching feedback.
- Produces an automatic PDF coaching report.

Two analysis modes are available:

- **Single session** - Compares a driver's laps against their own best corner performance.
- **Driver vs reference** - Compares a driver against a faster reference session.

## Example Output

The report combines telemetry traces with automatically generated coaching feedback, highlighting the corners and driving phases with the greatest performance potential.

Plot guide:

- Orange background: Braking phase.
- Purple background: Rotation / mid-corner phase.
- Green background: Traction / exit phase.
- Vertical dotted line: Detected apex.
- Blue line and grey band: Coached driver median and interquartile range across valid laps.
- Red line: Coached driver's best lap or best-corner segment.
- Green dashed line: Faster reference driver in `vs-reference` mode.

### Single Session

<p align="center">
  <img src="Report_single.png" alt="Single-session coaching report example" width="90%">
</p>

### Driver vs Reference

<p align="center">
  <img src="Report_reference.png" alt="Driver versus reference coaching report example" width="90%">
</p>

Full example PDF reports are included in the repository:

- [`coaching_report_single.pdf`](coaching_report_single.pdf).
- [`coaching_report_reference.pdf`](coaching_report_reference.pdf).

## Quick Start

### Requirements

- Python 3.10+.
- Windows terminal, PowerShell or CMD.
- [Mu](https://github.com/patrickmoore/Mu/releases/tag/1.9.5.0) for converting iRacing `.ibt` telemetry to CSV.

Mu is an external prerequisite. It is used only to convert iRacing telemetry into CSV and optional MoTeC i2-compatible files. The original contribution of this project starts after conversion, with Python-based telemetry ingestion, analysis, coaching logic and reporting.

### Install Dependencies

From this project folder:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not available, use:

```powershell
py -m pip install -r requirements.txt
```

### Analyse a Single Session

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\session.csv"
```

### Compare Against a Reference Driver

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\session.csv" --vs-reference "C:\path\to\reference.csv"
```

Results are written to a session-specific folder under `outputs/`, including the generated `coaching_report.pdf`.

## Analysis Pipeline

The processing pipeline is deterministic and reproducible:

1. Parse and validate Mu-exported telemetry.
2. Preserve raw channels and create derived channels separately.
3. Select valid laps and reject out laps, partial laps and contaminated laps.
4. Align valid laps by lap distance rather than time.
5. Detect corners and split them into braking, rotation and traction phases.
6. Calculate per-corner performance metrics.
7. Rank the most relevant areas for improvement.
8. Generate deterministic coaching feedback and the final PDF report.

<details>
<summary><strong>Operating Modes</strong></summary>

## Single-Session Mode

Use this mode when analysing one driver's session without an external reference driver.

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv"
```

If `python` is not available:

```powershell
py .\TelemetryAnalyzer.py "C:\path\to\my_session.csv"
```

This mode:

- Detects corners from the driver's own session.
- Computes per-corner metrics for all valid laps.
- Builds a best-per-corner internal benchmark.
- Ranks priority corners by time loss, variability and relevance score.
- Generates coaching CSV/JSON outputs.
- Generates a coaching PDF report.

## Driver vs Reference Mode

Use this mode when comparing a coached driver against a faster reference driver.

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --vs-reference "C:\path\to\faster_driver_session.csv"
```

If `python` is not available:

```powershell
py .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --vs-reference "C:\path\to\faster_driver_session.csv"
```

This mode:

- Detects canonical corners on the reference session.
- Reuses those corner IDs and boundaries for the coached driver session.
- Builds the benchmark from the best corner segments in the reference session.
- Scores coached driver corner performance against the reference benchmark.
- Generates coaching CSV/JSON outputs.
- Generates a coaching PDF report with overlays.
- Shows coached median and coached IQR traces.
- Shows coached best-corner lap trace.
- Shows reference driver trace.

By default, this mode expects:

- The same track configuration, based on `Venue` metadata.
- The same vehicle, based on `Vehicle` metadata.

Intentional cross-car comparison can be enabled with:

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --vs-reference "C:\path\to\ref.csv" --allow-mixed-vehicle
```

</details>

<details>
<summary><strong>Advanced Configuration</strong></summary>

## Optional Runtime Flags

- `--config <path>` loads deterministic tuning parameters from JSON.
- `--allow-mixed-vehicle` allows cross-car comparison in `vs-reference` mode.
- `--disable-reference-cache` disables shared cached reference-session artifacts.

Example:

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --config ".\pipeline_config.json"
```

## Example Config File

```json
{
  "lap_validity": {
    "min_samples": 1000,
    "min_dist_span_pct": 95.0
  },
  "corner_detection": {
    "min_corner_count": 10,
    "max_corner_count": 25,
    "min_corner_spacing_m": 90.0,
    "curvature_weight": 0.35,
    "manual_apex_pct": null
  },
  "coaching": {
    "max_priority_corners": 5,
    "min_time_loss_s": 0.03,
    "min_inconsistency_s": 0.04
  },
  "comparison": {
    "allow_mixed_vehicle": false,
    "reference_cache_enabled": true
  }
}
```

The resolved runtime configuration is always saved to:

- `outputs/<session-id>/config_snapshot.json`.

## Configuration Tuning

Default values are tuned for FIA-style road and street circuits. If you analyse very different layouts, for example ovals, these are the main variables to review.

### Corner Detection

File: `telemetry_pipeline/corner_metrics.py` (`CornerDetectionConfig`).

- `min_corner_count`, `max_corner_count`, `target_corner_count`: Expected number of significant corners.
- `min_corner_spacing_m`, `min_corner_spacing_pct`: Minimum spacing between detected apexes.
- `activity_quantile_threshold`, `min_activity_score`: Sensitivity of peak acceptance.
- `curvature_weight`: Balance between dynamic signals and path geometry from `Lat`/`Lon`.
- `speed_apex_search_window_pct`: Local search window used to refine apexes to minimum speed.
- `manual_apex_pct`: Optional list of official corner apex positions in percentage lap distance.

Tuning guidance:

- Increase spacing to avoid over-splitting long-radius turns.
- Decrease spacing slightly for tight technical complexes.
- Increase activity thresholds for cleaner but fewer corner detections.
- Decrease activity thresholds if true corners are being missed.
- Increase `curvature_weight` only when geometric path data is reliable.
- Use `manual_apex_pct` only when a circuit has been validated against onboard video, MoTeC or an official map.
- When `manual_apex_pct` is provided, automatic peak selection is bypassed, but braking, rotation, traction phases and all metrics are still computed from telemetry.

### Phase Boundaries

File: `telemetry_pipeline/corner_metrics.py` (`CornerDetectionConfig`).

- `brake_threshold_pct`.
- `throttle_reapply_threshold_pct`.
- `throttle_reapply_consecutive_points`.

These control where braking, rotation and traction phases begin and end. If pedal channels are noisier or differently scaled, these are the first values to retune.

### Lap Validity and Alignment

File: `telemetry_pipeline/lap_processing.py`.

- `LapValidityConfig.min_samples`, `min_dist_span_pct`, `min_ontrack_fraction`, etc.
- `DEFAULT_DISTANCE_STEP_PCT` for alignment resolution.

Use stricter validity thresholds for cleaner datasets, or relax carefully for sparse telemetry.

### Coaching Threshold Sensitivity

File: `telemetry_pipeline/coaching.py` (`CoachingConfig`).

- `min_time_loss_s`, `min_inconsistency_s`, `min_priority_score`.
- `significant_*` thresholds for brake, rotation, traction, yaw, steer, slip and related signals.

These values change how strict the coaching engine is when labelling an issue as meaningful.

### Known-Track Official Turn Counts

File: `telemetry_pipeline/cli.py` (`_infer_expected_corner_count`).

- If fixed official turn numbering is required for additional tracks, add mappings there.
- If no mapping is defined, the detector runs in generic track-agnostic mode.

</details>

<details>
<summary><strong>Technical Methodology</strong></summary>

## Telemetry Ingestion

- Metadata, units and numeric telemetry are parsed from Mu-exported CSV files.
- Raw telemetry channels are preserved.
- Derived channels are added separately, including `Speed_kmh`, `SteeringWheelAngle_deg` and `YawRate_deg_s`.
- Mu is required for `.ibt` to `.csv` conversion, but Mu is not reimplemented by this project.

## Lap Validation and Distance Alignment

Laps are classified as valid or invalid using deterministic rules:

- Lap ID must be `>= 1`, which ignores lap 0 out/warm-up by default.
- The lap must contain enough telemetry samples, default `>= 1000`.
- Distance coverage must start at or before `1%`, end at or after `99%`, and span at least `95%`.
- `OnPitRoad` fraction must be `0` when the channel exists.
- `IsOnTrack` fraction must be at least `0.99` when the channel exists.

Comparisons are performed by `LapDistPct` distance alignment, not by time. Valid laps are interpolated onto a shared distance grid, default `0.1%`, for direct point-by-point comparison.

## Corner Detection and Phase Segmentation

Corner apexes are detected deterministically from local peaks in a corner-activity score combining:

- Speed deficit.
- Yaw-rate demand.
- Lateral acceleration.
- Steering demand.
- Path curvature when `Lat`/`Lon` is available.

Corner boundaries are defined from neighbouring apex midpoints. Each corner is then split into:

- Braking phase, from `brake_start_pct` to `brake_end_pct`.
- Rotation phase, from `rotation_start_pct` to `rotation_end_pct` around apex.
- Traction phase, from `traction_start_pct` to `corner_end_pct`.

Known tracks can use an official turn-count mapping or a validated `manual_apex_pct` map for clearer T1/T2/... reporting.

## Per-Corner Metrics

The software extracts coaching-relevant metrics for each valid lap and each detected corner. These include:

- Corner and phase time.
- Brake point, brake duration, brake peak and brake release behaviour.
- Minimum speed and apex speed loss.
- Steering demand and steering release behaviour.
- Yaw-rate and lateral-acceleration response.
- Throttle reapplication and exit acceleration.
- Wheel-speed spread when wheel-speed channels are available.
- Conservative apex-position delta when `Lat`/`Lon` channels are available.

The report also includes a deterministic theoretical optimal lap time:

- Optimal lap = sum of best `corner_time_s` across corners.
- Potential gain = best full lap time minus theoretical optimal lap time.

## Corner Ranking and Coaching Logic

Per-corner ranking supports:

- Time loss versus the deterministic reference profile.
- Variability and inconsistency across valid laps.
- A robust ranking blend using mean and median for time loss.
- A robust inconsistency blend using standard deviation and median absolute deviation.
- A combined coaching relevance score.

The coaching layer is deterministic and rule-based. It maps measured telemetry patterns to symptoms, likely causes, recommended actions and practical focus notes.

## Coaching Confidence and Priority

- `coaching_priority_rank` orders corners by intervention value, where `1` is the highest priority.
- The ranking is computed deterministically from time loss, inconsistency and coaching relevance evidence.
- In single-session mode, the reference is built from the driver's best corner segment per corner.
- In `vs-reference` mode, the reference is built from the faster reference driver session.
- `confidence` ranges from `0.00` to `1.00` and expresses evidence strength, not driver quality.

Confidence increases when:

- The signal pattern is consistent across laps.
- Multiple channels support the same diagnosis, for example speed, brake and yaw.
- Core channels needed for the diagnosis are available.

Confidence decreases when:

- Evidence is weak or contradictory.
- Behaviour varies strongly lap to lap.
- Key channels for that inference are missing.

Suggested interpretation bands:

- `>= 0.80`: High confidence, safe to prioritise immediately.
- `0.60 to 0.79`: Medium confidence, good candidate to verify with overlays.
- `< 0.60`: Low confidence, treat as a hypothesis requiring extra validation.

</details>

<details>
<summary><strong>Generated Files and Output Structure</strong></summary>

Every run writes to a session-specific folder:

- `outputs/<session-id>/...`.

Core artifacts:

- `telemetry_numeric.csv.gz`.
- `metadata.json`.
- `units.json`.
- `parse_report.json`.
- `laps/lap_summary.csv`.
- `laps/aligned_laps_by_distance.csv.gz`.
- `laps/alignment_report.json`.
- `config_snapshot.json`.
- `coaching_report.pdf`.

Single-session analysis outputs:

- `corners/corner_definitions.csv`.
- `corners/corner_lap_metrics.csv.gz`.
- `corners/corner_ranking.csv`.
- `corners/corner_reference_profile.csv`.
- `corners/corner_report.json`.
- `coaching/corner_coaching.csv`.
- `coaching/session_coaching_summary.json`.
- `coaching/coaching_report.json`.

Vs-reference additional outputs:

- `corners_vs_reference/corner_definitions.csv`.
- `corners_vs_reference/corner_lap_metrics.csv.gz`.
- `corners_vs_reference/corner_ranking.csv`.
- `corners_vs_reference/corner_reference_profile.csv`.
- `corners_vs_reference/corner_report.json`.
- `coaching_vs_reference/corner_coaching.csv`.
- `coaching_vs_reference/session_coaching_summary.json`.
- `coaching_vs_reference/coaching_report.json`.

Reference cache outputs, when enabled:

- `outputs/_reference_cache/<cache-key>/...`.

If cache is disabled:

- `outputs/<driver-session-id>/comparison_reference_session/<reference-session-id>/...`.

Additional notes:

- `<session-id>` is auto-built from metadata, including date, time, vehicle, venue, session and driver.
- Large tables are stored as `.csv.gz` to reduce disk usage without data loss.

</details>

<details>
<summary><strong>Mu CSV Format and Troubleshooting</strong></summary>

## Mu Export Format Expected

Mu CSV files must contain:

1. Metadata lines at the top.
2. One blank line.
3. Header row.
4. Units row.
5. Sampled telemetry rows.

## Troubleshooting

`ModuleNotFoundError: No module named 'pandas'`.

```powershell
python -m pip install pandas
```

If needed:

```powershell
py -m pip install pandas
```

`Missing dependency: matplotlib`.

```powershell
python -m pip install matplotlib
```

If needed:

```powershell
py -m pip install matplotlib
```

`File not found`.

- Check the CSV path.
- Use quotes around paths that contain spaces.

</details>
