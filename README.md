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

Optional flags:

- `--preview-rows 10` to change preview size.
- `--no-derived` to skip derived channels.

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
