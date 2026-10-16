# Used Car Price Prediction with Neural Networks

> Tuning an ANN across several architectures to predict used-car resale prices.

## Overview

- Individual Deep Learning final-exam project (UAS, dataset B): predict a used car's selling price from tabular features.
- Starts from a required 3-hidden-layer sigmoid ANN (512, 256, 128 neurons) and improves it step by step.

## Data / Scope

- About 4,200 car listings with 16 columns such as year, mileage, engine size, fuel, region and city; split 2,515 / 839 / 839 for train, validation and test after encoding.
- Data issues handled: wrong data types, inconsistent categories, impossible years (up to 3014), negative mileage and heavy price outliers.

## Workflow

- Cleaned the data and winsorized price outliers (IQR caps) instead of dropping them, to keep real expensive cars.
- One-hot encoding and scaling, then a baseline sigmoid ANN trained for 25 epochs.
- Tried improved versions v2 to v4 and an activation search (ReLU, ELU, tanh, LeakyReLU, sigmoid and hybrids).
- Evaluated baseline and the final model on the validation and test sets with MAE, RMSE and R².

## Results

- Baseline ANN: validation MAE 742, RMSE 1,663, R² 0.9652.
- Final hybrid sigmoid-tanh model (v4): validation MAE 489, RMSE 1,408, R² 0.9751, with the best validation MSE (0.0279 vs 0.0313 for the baseline).
- Ideas that failed are kept in the notebook (v2 and v3 did not beat the baseline), which shows the tuning process.
- Predictions of the final model sit closer to the ideal diagonal than the baseline, especially for mid-priced cars.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, Matplotlib

## Repository Structure

```
notebook/    Full notebook with code, outputs and written interpretation
figures/     Key charts exported from the notebook
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/UAS_DL_Used_Car_Price_ANN.ipynb
```

The notebook was written in Google Colab; the dataset is not included, so upload it to the Colab session (or edit the path in the first data-loading cell).

## Team

Individual project by Nadya Angelie Lislie (Deep Learning course, BINUS University).

## Notes

- Student ID numbers were removed from the notebook text.
- The raw dataset (`2B.parquet`) is course material and is not included.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
