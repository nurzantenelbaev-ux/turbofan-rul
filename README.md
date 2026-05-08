# Predictive Maintenance on Turbofan Engines · XGBoost vs LSTM

Comparing XGBoost and LSTM for **Remaining Useful Life (RUL)** prediction on the NASA C-MAPSS dataset (FD001–FD004).

> **Course:** CSCI 447 — Group Project, Nazarbayev University.  
> **Author:** Nurzhan Tenelbayev (SEDS, Department of Computer Science).  
> **Tagged release:** `v1.0`

---

## Results

Test-set performance, averaged over five seeds (7, 13, 42, 101, 999):

| Subset | Model       | RMSE (cycles)   | NASA Score       |
| :---:  | :---        |     :---:       |   :---:          |
| FD001  | XGBoost     |    15.44        |   393.46         |
| FD001  | **LSTM**    |  **13.69 ✓**    | **365.59 ✓**     |
| FD002  | **XGBoost** | **14.27 ✓**     | **1036.52 ✓**    |
| FD002  | LSTM        |    15.08        |   1046.79        |
| FD003  | XGBoost     |    13.35        |   337.78         |
| FD003  | **LSTM**    |  **12.60 ✓**    | **301.90 ✓**     |
| FD004  | XGBoost     |    15.61        | **1150.36 ✓**    |
| FD004  | **LSTM**    |  **15.47 ✓**    |   1296.28        |

LSTM wins 3/4 subsets on RMSE; XGBoost wins 3/4 on NASA score (which penalises late predictions more heavily).

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/turbofan-rul
cd turbofan-rul

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Data

The NASA C-MAPSS dataset is **not included** in the repository. Download it with:

```bash
bash scripts/download_data.sh
```

This downloads (~5 MB) and unpacks the data into `data/raw/`.

---

## Running the Pipeline

Run these commands in order. Replace `FD001` with `FD002`, `FD003`, or `FD004` to run other subsets.

```bash
# Step 1 — Extract and engineer features
python scripts/extract_features.py --subset FD001

# Step 2 — Select the most informative features
python scripts/select_features.py --subset FD001 --method xgb_importance --k 101
python scripts/select_features.py --subset FD001 --method xgb_importance --k 70
python scripts/select_features.py --subset FD001 --method xgb_importance --k 102
python scripts/select_features.py --subset FD001 --method xgb_importance --k 220

# Step 3 — Tune XGBoost hyperparameters
# Models folder already has best parameters, so it can be skipped
python scripts/tune_xgb.py --subset FD001

# Step 4 — Tune LSTM hyperparameters
# Models folder already has best parameters, so it can be skipped
python scripts/tune_lstm.py --subset FD001

# Step 5 — Train final models (5 seeds) and evaluate on the test set
python scripts/train.py --subset FD001

# Step 6 (optional) — Feature importance analysis
python scripts/feature_importance.py --subset FD001
```

To run all four subsets at once:

```bash
for SUB in FD001 FD002 FD003 FD004; do
    python scripts/extract_features.py --subset $SUB
    python scripts/select_features.py  --subset $SUB
    python scripts/tune_xgb.py         --subset $SUB
    python scripts/tune_lstm.py        --subset $SUB
    python scripts/train.py            --subset $SUB --seeds 7 13 42 101 999
done
```

**Outputs written automatically:**

| Output | Location |
| :--- | :--- |
| Final metrics (RMSE, NASA score) | `models/results_<SUB>.csv` |
| Feature importance plots | `results/feature_importance/` |
| Hyperparameter tuning logs | `models/` |
| Trained model checkpoints | `models/` |

> Pre-tuned hyperparameters are already committed under `configs/` — you can skip Steps 3–4 and go straight to `train.py` if you want to reproduce the exact reported results without re-tuning.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The suite (28 tests) covers RMSE/NASA scoring, feature sanity checks, and data-loader correctness.

---

## Repository Layout

```
turbofan-rul/
├── src/                    # Library modules (importable, no side-effects)
│   ├── config.py           # Seeds, paths, sensor lists, hardware detection
│   ├── data_loader.py      # Loading, RUL construction, scaling, windowing
│   ├── features.py         # ~39 hand-crafted features per sensor
│   ├── models.py           # XGBoost and LSTM model definitions
│   └── evaluate.py         # RMSE and NASA scoring
├── scripts/                # Entry points — one task per script
│   ├── download_data.sh    # Fetch NASA C-MAPSS into data/raw/
│   ├── extract_features.py
│   ├── select_features.py
│   ├── tune_xgb.py
│   ├── tune_lstm.py
│   ├── train.py
│   ├── feature_importance.py
│   └── check_gpu.py        # Smoke test for CUDA
├── configs/                # Committed tuned-hyperparameter JSONs
├── data/                   # Raw files (gitignored) + processed cache
├── models/                 # Checkpoints, result CSVs/JSONs
├── results/                # Plots, importance reports, run logs
├── notebooks/              # Exploration notebooks
├── tests/                  # Unit tests
├── .gitignore
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Reproducibility

All randomness is fixed in `src/config.py`:

- Python `random`, `numpy`, and `torch` (CPU + CUDA) seeded to **42**.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- The five evaluation seeds are `[7, 13, 42, 101, 999]`.
- Train/val split is engine-level (no row leakage).

---

## Hardware Notes

The pipeline runs on CPU or CUDA GPU (auto-detected). Reference timings (i7-12700H, RTX 1050 Ti, 32 GB RAM):

| Stage (FD001)               | CPU only | With GPU |
| :---                        |  :---:   |  :---:   |
| Feature extraction          |  6 min   |  6 min   |
| XGBoost tuning              |  10 min   |  8 min   |
| LSTM tuning (4 configs)     | 20 min   |  20 min   |
| Final 5-seed training       | 30 min   |  25 min   |

---

## Citation

```bibtex
@misc{tenelbayev2026turbofan,
  title  = {Predictive Maintenance on Turbofan Engines: XGBoost vs LSTM on the NASA C-MAPSS Dataset},
  author = {Nurzhan Tenelbayev},
  year   = {2026},
  note   = {CSCI 447 group project, Nazarbayev University},
  url    = {https://github.com/<your-username>/turbofan-rul},
}
```

Dataset:

```bibtex
@inproceedings{saxena2008damage,
  title     = {Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation},
  author    = {Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  booktitle = {Int. Conf. on Prognostics and Health Management (PHM)},
  year      = {2008},
}
```

---

## License

MIT — see `LICENSE`.


