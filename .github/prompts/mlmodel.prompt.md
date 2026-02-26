# ML Model Prompt

Build a machine learning pipeline that predicts the next best incentive tier for a travel agency.

## Context

We have the following PostgreSQL tables:

- airlines
- routes
- transactions
- travel_agencies
- agency_performance
- incentive_programs
- incentive_tiers
- incentive_redemptions

Each agency has monthly performance and transaction history.

## Task

1. Extract historical data
2. Engineer rolling features
3. Generate label for next quarter tier
4. Train a classification model
5. Evaluate performance
6. Save model artifact
7. Provide prediction function

## Constraints

- Use time-based split
- Avoid data leakage
- Modular code
- Production-ready structure

## Expected Output

- data_loader.py
- feature_engineering.py
- train.py
- predict.py
- evaluate.py

Use scikit-learn or XGBoost.