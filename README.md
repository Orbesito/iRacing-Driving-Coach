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

### Mode 1: Single-Session Analysis (Your Session Only)

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv"
```

If `python` is not available:

```powershell
py .\TelemetryAnalyzer.py "C:\path\to\my_session.csv"
```

In this mode, the tool:
- detects corners from your session,
- computes per-corner metrics for every valid lap,
- builds the reference per corner from the best corner segment across your valid laps,
- ranks coaching-priority corners by time loss, variability, and relevance score.

### Mode 2: Driver vs Faster Reference Driver

```powershell
python .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --reference-csv "C:\path\to\faster_driver_session.csv"
```

If `python` is not available:

```powershell
py .\TelemetryAnalyzer.py "C:\path\to\my_session.csv" --reference-csv "C:\path\to\faster_driver_session.csv"
```

In this mode, the tool:
- detects corners on the reference session,
- uses those corner IDs/boundaries as stable canonical corners for both sessions,
- builds a per-corner benchmark from the reference session best corner segments,
- compares your session against that benchmark turn by turn,
- and in PDF corner plots overlays both:
  - external reference driver best-corner trace,
  - coached driver best-corner trace (from the coached session).

The comparison mode expects both files to be from the same track configuration.

The script always:

- applies derived channels,
- detects valid laps deterministically,
- ignores out/warm-up and short/partial laps,
- aligns valid laps by distance (`LapDistPct`),
- includes acceleration channels (`LatAccel`, `LongAccel`) in analysis,
- includes additional high-value dynamics channels in aligned outputs (`BrakeRaw`, `ThrottleRaw`, wheel speeds, velocity components, `VertAccel`, `Lat`, `Lon`),
- detects corners from a deterministic multi-signal profile (speed, yaw, lateral accel, steering, and path curvature when Lat/Lon is available),
- segments each corner into braking, rotation, and traction phases,
- computes per-corner coaching metrics and ranking,
- builds a deterministic coaching interpretation layer (symptom, cause, action, drill, confidence),
- writes both concise and detailed coaching notes per priority corner,
- generates a coaching PDF report with channel overlays (speed, brake, throttle, steering, yaw rate, lateral acceleration),
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
- `outputs/<session-id>/corners/corner_reference_profile.csv`
- `outputs/<session-id>/corners/corner_report.json`
- `outputs/<session-id>/coaching/corner_coaching.csv`
- `outputs/<session-id>/coaching/session_coaching_summary.json`
- `outputs/<session-id>/coaching/coaching_report.json`
- `outputs/<session-id>/coaching/coaching_report.pdf`

When `--reference-csv` is used, additional driver comparison outputs are saved at:
- `outputs/<driver-session-id>/corners_vs_reference/corner_definitions.csv`
- `outputs/<driver-session-id>/corners_vs_reference/corner_lap_metrics.csv`
- `outputs/<driver-session-id>/corners_vs_reference/corner_ranking.csv`
- `outputs/<driver-session-id>/corners_vs_reference/corner_reference_profile.csv`
- `outputs/<driver-session-id>/corners_vs_reference/corner_report.json`
- `outputs/<driver-session-id>/coaching_vs_reference/corner_coaching.csv`
- `outputs/<driver-session-id>/coaching_vs_reference/session_coaching_summary.json`
- `outputs/<driver-session-id>/coaching_vs_reference/coaching_report.json`
- `outputs/<driver-session-id>/coaching_vs_reference/coaching_report.pdf`
- `outputs/<driver-session-id>/comparison_reference_session/<reference-session-id>/...` (reference-session ingestion/laps/corners artifacts used in the comparison run)

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
  - steering demand,
  - path curvature (when `Lat`/`Lon` is available).
- Corner boundaries are defined from neighboring apex midpoints.
- Apex candidates are filtered with a minimum physical spacing target (meters), converted to `% lap` using estimated lap length from `Lat/Lon` when available. This reduces over-splitting of one complex into multiple false corners on different track lengths.
- For known tracks with an official turn-count mapping (currently Miami GP), detection is constrained to that official corner count for clearer T1/T2/... references.
- Per-corner phase boundaries are represented as:
  - braking phase (`brake_start_pct` to `brake_end_pct`)
  - rotation phase (`rotation_start_pct` to `rotation_end_pct`, around apex)
  - traction phase (`traction_start_pct` to `corner_end_pct`)
- Raw pedal channels are used when available (`BrakeRaw`, `ThrottleRaw`) for phase detection and pedal metrics.
- Additional outputs include traction wheel-speed spread and body-slip proxy from velocity channels when available.
- A conservative positioning proxy (`apex_position_delta_m` vs reference) is included when `Lat/Lon` channels are present.
- Per-corner ranking supports:
  - time loss vs deterministic reference profile
  - variability / inconsistency
  - combined coaching relevance score
- The report also includes a deterministic theoretical optimal lap time:
  - optimal lap = sum of best `corner_time_s` across corners
  - potential gain = best full lap time minus theoretical optimal lap time

## Interpreting Confidence and Priority Score (Academic Use)

- `coaching_priority_rank` orders corners by intervention value (1 = highest priority).
- The ranking is computed deterministically from:
  - time loss versus the selected reference,
  - variability/inconsistency across valid laps,
  - coaching relevance weighting from the phase and symptom evidence.
- In single-session mode, the reference is built from the driver's best corner segment per corner.
- In vs-reference mode, the reference is built from the faster reference driver session.
- `confidence` (0.00 to 1.00) expresses evidence strength for the coaching conclusion, not driver quality.
- Confidence increases when:
  - the signal pattern is consistent across laps,
  - multiple channels support the same diagnosis (for example speed + brake + yaw),
  - core channels needed for that diagnosis are available.
- Confidence decreases when:
  - evidence is weak or contradictory,
  - behavior varies strongly lap to lap,
  - key channels for that inference are missing.
- Suggested interpretation bands:
  - `>= 0.80`: high confidence (safe to prioritise immediately),
  - `0.60 to 0.79`: medium confidence (good candidate, verify with overlays),
  - `< 0.60`: low confidence (treat as hypothesis, require extra validation).

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
- `Missing dependency: matplotlib` (PDF report):
  - install with the same interpreter you use to run the script:
    - `python -m pip install matplotlib`
    - or `py -m pip install matplotlib`
- `File not found`:
  - check the CSV path and quotes.

## Notes

- Mu is not replaced by this project; Mu is required to produce the CSV input.
- Raw channels are preserved. Derived channels are added separately (`Speed_kmh`, `SteeringWheelAngle_deg`, `YawRate_deg_s`).
