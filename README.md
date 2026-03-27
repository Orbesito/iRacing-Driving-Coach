# iRacing-Driving-Coach

Deterministic telemetry ingestion for Mu-exported iRacing sessions.

## Prerequisites (Final User)

1. Windows terminal (PowerShell or CMD).
2. Python 3.10+ installed and available in terminal:
   - `python --version`
   - if `python` is not available, try `py --version`
3. Mu installed (external prerequisite) and able to export iRacing `.ibt` files to `.csv`.

## Install Python (If Needed)

1. Install Python 3.10+ on Windows.
2. During installation, enable `Add python.exe to PATH`.
3. Re-open terminal and verify:
   - `python --version`
   - or `py --version`

## Install Mu (If Needed)

1. Install Mu telemetry utility on your Windows machine.
2. Use Mu to convert iRacing `.ibt` files to `.csv`.
3. Confirm the exported CSV follows the format documented below.

## Install Dependencies

From this project folder, run one of:

```powershell
python -m pip install -r requirements.txt
```

or:

```powershell
py -m pip install -r requirements.txt
```

## Run

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\mu_exported_session.csv"
```

If `python` is not available:

```powershell
py .\TelemetryAnalyzer.py "C:\path\to\mu_exported_session.csv"
```

No extra inputs are required for lap processing or saving. The script always:

- applies derived channels,
- detects valid laps deterministically,
- ignores out/warm-up and short/partial laps,
- aligns valid laps by distance (`LapDistPct`),
- includes acceleration channels (`LatAccel`, `LongAccel`) in analysis,
- detects main corners from a deterministic multi-signal profile (speed, yaw, lateral accel, steering),
- segments each corner into braking, rotation, and traction phases,
- computes per-corner coaching metrics and ranking,
- saves outputs for next stages.

## Saved Artifacts

Every run saves artifacts in a session-specific folder:

- `outputs/<session-id>/telemetry_numeric.csv`
- `outputs/<session-id>/metadata.json`
- `outputs/<session-id>/units.json`
- `outputs/<session-id>/parse_report.json`
- `outputs/<session-id>/laps/lap_summary.csv`
- `outputs/<session-id>/laps/aligned_laps_by_distance.csv`
- `outputs/<session-id>/laps/alignment_report.json`
- `outputs/<session-id>/corners/corner_definitions.csv`
- `outputs/<session-id>/corners/corner_lap_metrics.csv`
- `outputs/<session-id>/corners/corner_ranking.csv`
- `outputs/<session-id>/corners/corner_report.json`

`<session-id>` is auto-built from metadata (date, time, vehicle, venue, session, driver), so different sessions do not overwrite each other.

## Lap Comparison Approach

- Laps are classified as valid/invalid using deterministic rules:
  - lap ID must be >= 1 (ignores lap 0 out/warm-up by default),
  - enough telemetry samples (`>= 1000`),
  - sufficient distance coverage (`start <= 1%`, `end >= 99%`, span `>= 95%`),
  - no pit-road contamination (`OnPitRoad` fraction must be `0` when channel exists),
  - lap must be mostly on track (`IsOnTrack` fraction `>= 0.99` when channel exists).
- Comparisons are done by `LapDistPct` distance alignment, not by time.
- Valid laps are interpolated onto a shared distance grid (`0.1%` step) for direct point-by-point comparison.

## Corner Coaching Metrics

- Corner apexes are detected deterministically from local peaks in a corner-activity score combining:
  - speed deficit,
  - yaw-rate demand,
  - lateral acceleration,
  - steering demand.
- Corner boundaries are defined from neighboring apex midpoints.
- Per-corner phase boundaries are represented as:
  - braking phase (`brake_start_pct` to `brake_end_pct`)
  - rotation phase (`rotation_start_pct` to `rotation_end_pct`, around apex)
  - traction phase (`traction_start_pct` to `corner_end_pct`)
- Per-corner ranking supports:
  - time loss vs reference lap
  - variability / inconsistency
  - combined coaching relevance score

## Mu Export Format Expected

The CSV should contain:

1. Metadata lines at the top.
2. One blank line.
3. Header row.
4. Units row.
5. Sampled telemetry rows.

## Troubleshooting

- `ModuleNotFoundError: No module named 'pandas'`:
  - install with the same interpreter you use to run the script:
    - `python -m pip install pandas`
    - or `py -m pip install pandas`
- `File not found`:
  - check the CSV path and quotes.

## Notes

- Mu is not replaced by this project; Mu is required to produce the CSV input.
- Raw channels are preserved. Derived channels are added separately (`Speed_kmh`, `SteeringWheelAngle_deg`, `YawRate_deg_s`).
