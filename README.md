# Auditing ML Models for Fairness  
_Reproducible pipelines for baseline and fairness‑aware training on the Adult Census Income dataset._

> This code accompanies the paper **“Auditing Machine‑Learning Models for Fairness: Theory and Case Study, a Reproducible Pipeline from Metrics to Mitigation.”**  
> Everything below is tested on Docker 20.10+.

---

## Repository layout

```
.
├── Biased_Models/                   # accuracy‑only benchmark
│   ├── ML_Biased.py
│   ├── audit_adult.py
│   ├── Dockerfile
│   └── requirements.txt
└── Fair_Machine_Learning_Models/    # same pipeline + ExponentiatedGradient
    ├── Fair_machine_learning.py
    ├── audit.py
    ├── Dockerfile
    └── requirements.txt
```

Each directory is completely self‑contained: its own Docker image, requirements, and outputs (although the original outputs were preserved, the .joblib files were eliminated due to their size).

---

## 1 · Baseline (accuracy‑only) run

```bash
cd Biased_Models

# build
docker build -t adult_bias_benchmark .

# run (results will be written inside audit/, models/ and predictions/)
docker run --rm -v "$(pwd)":/app adult_bias_benchmark
```

### What you get

```
Biased_Models/
└─ audit/
   ├─ metrics_<best>.csv
   ├─ parity_gaps_<best>.png
   ├─ roc_age_<best>.png
   ├─ … (other plots)
   └─ audit_provenance.json
models/
└─ model_<all_models>.joblib
predictions/
└─ preds_<all_models>.csv
```

The benchmark tunes 14 classifiers with nested Optuna TPE (5×3 CV) and audits the best performer.

---

## 2 · Fairness‑aware run (Equalised Odds constraint)

```bash
cd Fair_Machine_Learning_Models

# build
docker build -t adult_bias_fair .

# run
docker run --rm -v "$(pwd)":/app adult_bias_fair
```

### Outputs

```
Fair_Machine_Learning_Models/
└─ audit/
   ├─ metrics_XGB_fair.csv
   ├─ parity_gaps_XGB_fair.png
   ├─ roc_age_band_xgb_fair.png
   ├─ … (other plots)
   └─ audit_provenance.json
models/
└─ model_XGB_fair.joblib
predictions/
└─ preds_XGB.csv
```

This pipeline takes the best hyper‑parameters found in the baseline, wraps the model in **Fairlearn’s Exponentiated Gradient** with an Equalised‑Odds constraint (`ε = 0.005`), and re‑audits the result.

---

## 3 · Compare the two runs

* Baseline <best_model>: higher accuracy / larger DP & EO gaps  
* Fair XGB: ≈ lower accuracy /  smaller fairness gaps
---

## Data

If `adult.csv` is absent the scripts automatically download it via `fairlearn.datasets.fetch_adult()` and cache a local copy.

---

## License

MIT—see `LICENSE`.

---

## Citation

Please cite the forthcoming CORE 2025 paper (DOI to appear):

```bibtex
@inproceedings{rodriguez2025fairaudit,
  title     = {Auditing Machine‑Learning Models for Fairness:
               Theory and Case Study, a Reproducible Pipeline from Metrics to Mitigation},
  author    = {Rodríguez Rodríguez, Noé Oswaldo and Rosas Alatriste, Carolina
               and Alarcón Paredes, Antonio and Yáñez Márquez, Cornelio
               and Recio García, Juan Antonio},
  booktitle = {Proceedings of the CORE 2025 Conference},
  year      = {2025}
}
```
