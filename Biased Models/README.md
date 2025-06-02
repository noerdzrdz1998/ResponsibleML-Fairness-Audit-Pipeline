# Adult Bias Benchmark and Audit

This repository provides a complete pipeline to train multiple classifiers on the Adult Census Income dataset, evaluate their performance and fairness metrics, and generate an audit report with detailed plots and metrics.

## Prerequisites

- Docker installed on your machine

## Usage

1. Build the Docker image:

   ```bash
   docker build -t adult_bias_audit .
   ```

2. Run the benchmark and audit pipeline:

   ```bash
   docker run --rm -v "$(pwd)":/app adult_bias_audit
   ```

After running, you will find:

- `predictions/` directory containing per-model prediction CSVs
- `models/` directory containing serialized model files
- `audit/` directory containing fairness metrics CSVs and plots
