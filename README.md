# Network Attack Classification System

ML-based intrusion detection system trained on the **NSL-KDD** dataset, classifying network
connections into 5 categories: `normal`, `dos`, `probe`, `r2l`, `u2r`.

## Contents
- `notebook/network_attack_classification.ipynb` — full analysis: EDA, preprocessing, training, evaluation, feature importance (already executed with outputs)
- `dashboard/app.py` — interactive Streamlit dashboard (live classifier, model comparison, confusion matrix, feature importance)
- `train_model.py` / `finalize_model.py` — training scripts
- `models/` — saved trained models (Random Forest, Decision Tree, SVM) + encoders + metrics
- `data/` — NSL-KDD dataset (KDDTrain+, KDDTest+)

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib
```

## Run the dashboard

```bash
cd dashboard
streamlit run app.py
```

## Results

| Model | Validation Accuracy (same-distribution split) | KDDTest+ Accuracy (unseen attack types) |
|---|---|---|
| Random Forest | **99.89%** | 75.25% |
| Decision Tree | 99.73% | 77.70% |
| SVM | 98.98% | 75.55% |

**Note on the two numbers:** NSL-KDD's official `KDDTest+` file deliberately includes attack
variants never seen during training, so accuracy is naturally lower there. The ~99% figure
(80/20 split of `KDDTrain+`) is the number most commonly cited in NSL-KDD papers/projects and
is a fair, honest number to put on a resume — just know the ~75% figure if asked about
generalization to unseen attacks in an interview, since that shows a deeper understanding of
the dataset than just the headline accuracy.

## Tech stack
Python, scikit-learn, pandas, NumPy, Streamlit, Plotly, matplotlib/seaborn, joblib
