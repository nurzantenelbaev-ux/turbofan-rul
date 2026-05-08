# Contributions

This was a single-author project. Below is a per-component log of what
was done by whom; this also appears in the final report's *Author
Contributions* section.

## Author

**Nurzhan Tenelbayev** — SEDS, Department of Computer Science,
Nazarbayev University.

| Component                                                    | Contributor       |
| :---                                                         | :---              |
| Data pipeline (loading, cleaning, RUL construction)          | Nurzhan Tenelbayev |
| Condition-aware normalization for FD002 / FD004              | Nurzhan Tenelbayev |
| Feature library (≈39 features per sensor)                    | Nurzhan Tenelbayev |
| Feature-selection comparison (Pearson / MI / XGBoost gain)   | Nurzhan Tenelbayev |
| XGBoost 5-stage greedy hyperparameter tuning                 | Nurzhan Tenelbayev |
| LSTM 4-config grid + training tricks (target scaling, clipping) | Nurzhan Tenelbayev |
| Final 5-seed training + test evaluation                      | Nurzhan Tenelbayev |
| Per-subset feature-importance analysis (`feature_importance.py`) | Nurzhan Tenelbayev |
| Unit-test suite (`tests/`)                                   | Nurzhan Tenelbayev |
| Plots and figures                                            | Nurzhan Tenelbayev |
| Final report (LaTeX, IEEE template)                          | Nurzhan Tenelbayev |
| README, Makefile, `.gitignore`, environment files            | Nurzhan Tenelbayev |

## External assistance

- **Dr. Aliya Nugumanova** (CSCI 447 instructor) — project guidance,
  feedback during the proposal and presentation stages.
- **NASA Ames Research Center** — public release of the C-MAPSS dataset.
- **Anthropic's Claude** — editorial feedback on the manuscript and
  scaffolding for the project structure (Makefile, GitHub upload guide,
  `requirements.txt`, README skeleton). All technical decisions, code
  logic, and experimental results are the author's own.
