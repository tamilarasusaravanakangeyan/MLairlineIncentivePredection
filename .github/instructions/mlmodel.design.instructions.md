# ML Model Design Instructions

Design a production-grade ML system to forecast the next best incentive target for travel agencies.

## Objective

Predict:

- Next incentive tier (classification)
OR
- Next revenue target (regression)

based on historical data from:

- transactions
- agency_performance
- incentive_redemptions
- incentive_tiers
- routes
- airlines

## System Design Requirements

### 1. Data Layer

- Extract from PostgreSQL
- Join tables efficiently
- Avoid N+1 queries
- Cache intermediate datasets

### 2. Feature Engineering

Create:

- Rolling 3-month revenue
- Rolling 6-month revenue
- Revenue growth %
- Ticket growth %
- Average commission rate
- Redemption frequency
- Loyalty usage ratio
- Airline diversity count
- Route diversity count
- Seasonality features (month, quarter)

### 3. Label Engineering

If classification:
- Label = next_quarter_tier_id

If regression:
- Label = next_quarter_revenue_target

### 4. Model Training

- Use XGBoost / LightGBM as default
- Perform time-series split
- Avoid random shuffling

### 5. Model Output

Return:
- predicted_tier
- confidence_score
- explanation values

### 6. Model Explainability

Use SHAP values to explain:
- Top features impacting prediction

### 7. Deployment

Expose via FastAPI endpoint:

POST /predict
{
  "agency_id": int,
  "as_of_date": date
}