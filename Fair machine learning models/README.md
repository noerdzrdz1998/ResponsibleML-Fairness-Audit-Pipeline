# Fairness Audit Pipeline

This repository walks through training an XGBoost model on the Adult Census Income data, applying an Equalized Odds constraint, and producing an audit report that balances accuracy with fairness.

## Prerequisites

- Docker installed on your machine

## Usage

1. Build the Docker image:

   ```bash
   docker build -t fairness_audit .
   ```

2. Run the debiasing and audit pipeline:

   ```bash
   docker run --rm -v "$(pwd)":/app fairness_audit
   ```

After running, you will find:

- `predictions/` – CSVs with out‑of‑fold scores
- `models/` – serialized best estimator
- `audit/` – fairness metrics and plots
