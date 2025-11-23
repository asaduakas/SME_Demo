# 🏗️ Construction Fleet Parts Intelligence Dashboard

An interactive Streamlit dashboard for managing construction fleet parts with AI-driven insights. It combines predictive maintenance, inventory optimization, and cost simulation to help fleet managers make smarter procurement and maintenance decisions.
---
## Features
### 📊 Overview

- Visualize parts failure risk and supplier lead times.

- Identify high-risk parts based on failure probability × downtime impact.

- Explore key KPIs: total parts, average failure rate, lead time, and part costs.

### 🤖 ML Predictions

- Predict if a part is due for replacement using a Random Forest classifier.

- Analyze feature importance and model performance (accuracy & ROC AUC).

- Predict for existing parts or custom input values.

### 📦 Inventory Optimization

- Calculate forecasted consumption, safety stock, and reorder points.

- Generate interactive reorder visualizations by part and category.

- Supports adjustable forecast horizons.

### 💰 Cost Simulation

- Estimate financial impact of parts failure and downtime.

- Model replacement and downtime costs by part and machine type.

- Visualize top cost-driving parts and risk cost breakdowns.

## Installation

```
git clone <repo_url>
cd fleet-parts-dashboard
pip install -r requirements.txt
streamlit run app.py
```
Dependencies include:
- streamlit
- pandas
- numpy
- scikit-learn
- plotly

## Usage

1. Launch the dashboard with: `streamlit run app.py`

2. Use the sidebar to filter machines, parts, and suppliers.

3. Explore tabs for:
- Overview
- ML Predictions
- Inventory Optimization
- Cost Simulation

4. Input custom part data in the ML Predictions tab for on-demand predictions.

## Data

- Currently uses a synthetic dataset: synthetic_construction_parts_dataset.csv.

- Replace with real maintenance records for production use.

## Notes

- The ML model is trained on synthetic data.

- All calculations are demo-oriented; parameters may require calibration for real-world operations.

## License

MIT License
