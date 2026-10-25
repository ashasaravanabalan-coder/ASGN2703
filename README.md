# ASGN2703 – Anomaly Detection in Industrial Equipment Sensor Data (Isolation Forest)

## Contents
| File | Description |
|---|---|
| `anomaly_detection_isolation_forest.ipynb` | Main notebook: runs the workflow step by step with explanations, visualisations and discussion (already executed, outputs included) |
| `anomaly_utils.py` | Modular helper functions (preprocessing, model, evaluation, plotting) imported by the notebook |
| `ASGN2703_Report.pdf` | PDF report: methodology, findings, recommendations |
| `industrial_sensor_data.csv` | Input dataset (3,000 records) |
| `anomaly_results.csv` | Output: all records ranked by anomaly score, with predicted flag |
| `results.json` | Metrics written by the notebook |
| `figures/` | Saved plots (PNG) |
| `requirements.txt` | Python dependencies |
| `make_report.py` | (Optional) regenerates the PDF from `results.json` and `figures/` (needs `reportlab`, `pillow`) |

## How to run
1. Python 3.9+ is required. Optionally create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter from this folder (the CSV and `anomaly_utils.py` must be in the same folder as the notebook):
   ```bash
   jupyter notebook anomaly_detection_isolation_forest.ipynb
   ```
4. Choose **Kernel → Restart & Run All**. Run time is under one minute. Figures are written to `figures/`, and `results.json` and `anomaly_results.csv` are regenerated.

Optional – rebuild the PDF: `pip install reportlab pillow` then `python make_report.py`.

Random seeds are fixed (`RANDOM_STATE = 42`), so results are reproducible.

## Key result (summary)
Isolation Forest (300 trees, contamination 5 %) flags 147 readings. Against `FaultFlag`: precision 0.41, recall 0.04, F1 0.08, ROC-AUC 0.48. A supervised Random Forest also scores ROC-AUC ≈ 0.50, so `FaultFlag` carries no signal recoverable from the sensor features in this dataset. See the report for details and recommendations.
