# Copilot Global Instructions

This repository builds a Machine Learning system to forecast the next best incentive target for travel agencies based on historical transaction and performance data.

## Project Context

Database: PostgreSQL  
Domain: Airline incentive optimization  
Goal: Predict next best incentive tier or revenue target for each travel agency.

## Rules for Code Generation

- Use Python 3.11+
- Prefer pandas, numpy, scikit-learn, xgboost, lightgbm
- Use SQLAlchemy for database access
- Use modular architecture:
  - data/
  - features/
  - models/
  - evaluation/
  - api/

- Follow clean architecture principles
- Write reusable functions
- Add type hints
- Add docstrings

## Modeling Guidelines

- Perform time-based train-test split
- Avoid data leakage
- Use rolling window features
- Use feature scaling only when needed
- Use SHAP for explainability

## Evaluation

- For classification: F1, ROC-AUC, Precision-Recall
- For regression: RMSE, MAE, R2
- Log experiments

## Code Style

- Follow PEP8
- Use logging instead of print
- Use config.yaml for parameters