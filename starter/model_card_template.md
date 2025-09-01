# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model type: Logistic Regression (binary classification)
- Task: Predict whether a person's income exceeds $50K/yr based on census features
- Frameworks/Libraries: scikit-learn
- Feature preprocessing: OneHotEncoder for categorical variables
- Repository path:
  - Training script: starter/starter/train_model.py
  - Data processing: starter/starter/ml/data.py
  - Model utilities: starter/starter/ml/model.py

## Intended Use
- Primary: Educational use for demonstrating an ML pipeline, data processing, evaluation, and deployment readiness.
- Out of scope: High-stakes decisions. Do not use for credit, employment, housing, or other sensitive use-cases.

## Training Data
- Source: UCI Adult Census Income dataset variant included in this repo
- Path: starter/data/census_clean.csv (cleaned by data_cleaning_eda.ipynb)
- Features used:
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
  - Continuous: all other columns (e.g., age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week)
- Label: salary (<=50K vs >50K)

## Evaluation Data
- Split: 80/20 train/test with stratification on the label
- Path: Derived from the same cleaned dataset via train/test split inside the training script

## Metrics
- Overall metrics stored at: starter/model/metrics.json
  - Keys: precision, recall, fbeta
- Slice (group) metrics stored at: starter/model/slice_output.txt
  - Per-category performance across each categorical feature

_Please run the training script to generate metrics:_
- From repo root: python -m starter.starter.train_model

{
  "precision": 0.5711842667806755,
  "recall": 0.8520408163265306,
  "fbeta": 0.6839006910673151
}

## Ethical Considerations
- Sensitive attributes: race and sex are included as features and slices are reported for transparency.
- Risks: Potential for biased performance across demographic groups.
- Mitigations: Review slice metrics (starter/model/slice_output.txt) and consider rebalancing, feature review, or fairness-aware techniques if deploying.

## Caveats and Recommendations
- Small sample (example subset) may not reflect real-world distributions.
- Model choice (logistic regression) is simple and may underfit complex relationships.
- Before deployment: expand training data, perform hyperparameter tuning, calibration, and robust fairness assessment.
