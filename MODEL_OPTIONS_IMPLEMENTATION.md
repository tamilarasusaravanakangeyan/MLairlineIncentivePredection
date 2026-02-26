# Model Options Implementation Guide

This document explains how Option A, Option B, and Option C are implemented in this repository, including assumptions, model behavior, and output interpretation.

## Scope

The system predicts incentive outcomes for travel agencies using historical transaction and performance data from PostgreSQL.

Implemented options:

- Option A — Classification (next quarter tier)
- Option B — Regression (next quarter revenue target)
- Option C — Recommendation (program + tier + commission)

---

## 1) Shared Data and Feature Foundation

All three options use the same leakage-safe feature base generated from these tables:

- `transactions`
- `agency_performance`
- `incentive_redemptions`
- `incentive_tiers`
- `incentive_programs`
- `routes`
- `airlines`

### Key engineered features

- Rolling windows:
  - `rolling_3m_revenue`
  - `rolling_6m_revenue`
  - `rolling_3m_tickets`
- Lags:
  - `revenue_lag_1m`
  - `tickets_lag_1m`
- Growth:
  - `revenue_growth_pct`
  - `ticket_growth_pct`
- RFM-style:
  - `recency_days`
  - `frequency_txn_per_quarter`
  - `monetary_avg_revenue_per_txn`
- Incentive behavior:
  - `loyalty_redemption_ratio`
  - `avg_commission_rate`
- Diversity and seasonality:
  - `airline_diversity_count`
  - `route_diversity_count`
  - `month_num`
  - `quarter_num`

### Leakage controls

- Time-based split only (last 3 months validation).
- Feature windows use historical rows up to the current snapshot month.
- Targets are from **next quarter**, so they are not part of current features.

---

## 2) Option A — Classification

### Objective

Predict which incentive tier an agency should be assigned next quarter.

### Target

- `next_quarter_tier_id`

### Label strategy

- Future quarter revenue is computed by agency and quarter.
- Annualized revenue is mapped to tier thresholds from `incentive_tiers`:
  - `min_annual_revenue`
  - `max_annual_revenue`
- The mapped tier becomes the label.

### Model

- `XGBClassifier` (`multi:softprob`)
- Class imbalance handled via sample weights derived from class frequencies.
- Early stopping enabled.

### Evaluation metrics

- **F1-score (weighted)**
  - Combines precision and recall across classes while accounting for class imbalance.
  - Useful when some tiers are much more frequent than others.
- **ROC-AUC (OvR, weighted)**
  - Measures class separability across thresholds in multiclass setup.
  - Useful for understanding ranking quality, not only hard class assignments.
- **Classification report** (per-tier precision/recall/F1/support)
  - Helps identify weak tiers (for example, low recall for a specific tier).
  - Supports operational rules for escalation/manual review.

### Outputs

- `predicted_tier_id`
- `confidence_score` (max predicted class probability)

### How to interpret

- `predicted_tier_id`: recommended tier class for next quarter.
- `confidence_score`:
  - Near 1.0: model is very certain relative to alternatives.
  - Near 0.3–0.5: low separation between classes; use with caution.
- **How metrics help operationally**:
  - Higher weighted F1 indicates better overall assignment quality.
  - Low per-class recall indicates missed agencies for that tier (under-targeting risk).
  - Low per-class precision indicates over-targeting that tier (cost/risk of wrong incentives).

---

## 3) Option B — Regression

### Objective

Predict next quarter revenue target value for an agency.

### Target

- `next_quarter_revenue_target`

### Label strategy

- Future quarter actual revenue is assigned to each current snapshot as the regression target.

### Model

- `XGBRegressor` (`reg:squarederror`)
- Early stopping enabled.
- Trained on same leakage-safe feature set used in Option A.

### Evaluation metrics

- **RMSE**
  - Penalizes larger errors more strongly.
  - Best for monitoring risk of large target misses.
- **MAE**
  - Average absolute error in revenue units.
  - Easiest metric for business interpretation and planning bands.
- **R2**
  - Fraction of target variance explained by the model.
  - Useful for overall fit diagnostics versus simple baseline behavior.

### Outputs

- `predicted_revenue_target` (float)

### How to interpret

- This is the model’s estimate of achievable next-quarter revenue target given historical behavior.
- Use RMSE/MAE to calibrate operational confidence bounds.
  - Example: planning range = prediction ± MAE (business heuristic).
- **How metrics help operationally**:
  - Rising RMSE indicates occasional large misses and increased planning risk.
  - MAE provides a practical expected miss size per agency.
  - Negative or low R2 indicates weak explanatory signal and need for model/feature review.

---

## 4) Option C — Next Best Incentive Recommendation

### Objective

Recommend the next best incentive package:

- Program
- Tier
- Commission percentage

### Inputs used

- Option A output:
  - `predicted_tier_id`
  - `confidence_score`
- Option B output:
  - `predicted_revenue_target`
- Incentive catalog from DB:
  - `program_id`, `program_name`
  - `tier_id`, `tier_name`, `tier_level`
  - `commission_percentage`
  - revenue thresholds

### Recommendation strategy (current implementation)

1. Convert predicted quarterly revenue to annualized revenue.
2. Filter candidate tiers/programs whose annual-revenue range contains this value.
3. Score candidates using weighted components:
   - commission attractiveness (normalized)
   - tier match with Option A
   - classification confidence contribution
4. Pick highest scored candidate.

### Evaluation metrics (current and recommended)

- **Current implementation note**:
  - Option C is a decision/ranking layer over Option A + B outputs.
  - It does not yet train a dedicated recommendation model with intrinsic ML loss metrics.
- **Operational metrics to track**:
  - Top-1 acceptance rate of recommendations.
  - Revenue uplift versus current policy/baseline.
  - Commission efficiency (uplift per commission cost).
  - Tier transition success rate after recommended offer execution.

### Output fields

- `program_id`, `program_name`
- `tier_id`, `tier_name`, `tier_level`
- `commission_percentage`
- `predicted_tier_id`
- `confidence_score`
- `predicted_revenue_target`

### How to interpret

- Recommendation is a decision layer over A+B predictions, not a standalone model.
- If commission is high but tier match is weak, recommendation score balances the trade-off.
- Use this output for decision support, with business guardrails.
- **How metrics help operationally**:
  - Acceptance rate indicates recommendation usability.
  - Uplift measures business value impact.
  - Efficiency protects incentive budget while maximizing outcome.

---

## 5) Assumptions

1. **Revenue-to-tier mapping is valid**
   - Tier thresholds in `incentive_tiers` represent meaningful boundaries.
2. **Historical behavior is predictive**
   - Recent rolling and lag metrics reflect near-future outcomes.
3. **Data quality is acceptable**
   - Missing values are handled, but systematic source errors can degrade quality.
4. **Quarterly seasonality is stable**
   - Month/quarter effects are assumed to persist.
5. **Recommendation scoring weights are heuristic**
   - Current Option C weights are policy choices, not learned via optimization.

---

## 6) API Mapping

- Option A endpoint: `GET /predict/{agency_id}?as_of_date=YYYY-MM-DD`
- Option B endpoint: `GET /predict-revenue/{agency_id}?as_of_date=YYYY-MM-DD`
- Option C endpoint: `GET /recommend/{agency_id}?as_of_date=YYYY-MM-DD`

If regression artifact is missing, Option B and C endpoints return `503`.

---

## 7) Practical Interpretation Playbook

### When Option A is high-confidence

- If `confidence_score >= 0.75`, the predicted tier is usually stable enough for automated downstream rules.

### When Option A is low-confidence

- If `confidence_score < 0.55`, treat as a review candidate:
  - inspect feature drift
  - compare with historical tier transitions
  - use Option B target to sanity check aggressiveness

### For Option B target usage

- Use prediction as a central target.
- Apply business margin bands for planning and negotiation.

### For Option C usage

- Recommended for next-action planning:
  - assign program + tier + commission offer strategy
- Should be combined with business constraints (budget, campaign strategy, geography).

---

## 8) Known Limitations

- Option C ranking is heuristic (not contextual bandit or causal uplift model).
- Confidence score is classification probability, not calibrated business probability.
- No explicit uncertainty intervals for regression yet.

---

## 9) Future Enhancements

- Calibrated class probabilities (Platt/Isotonic).
- Quantile regression for revenue target ranges (P10/P50/P90).
- Learned recommendation policy (uplift/ranking model).
- MLflow tracking integration for reproducibility and model governance.

---

## 10) Documentation Updates

- 2026-02-26: Added detailed evaluation metric definitions and operational interpretation for Option A, Option B, and Option C.
- 2026-02-26: Aligned project and API README wording with this implementation guide.
