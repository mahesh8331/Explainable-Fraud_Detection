import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data/processed/clean.parquet")

# 1. Use the stored time split. Test rows are not touched here.
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

# Identifiers and time are not features; the label and split column are not either
drop = ["TransactionID", "TransactionDT", "isFraud", "split"]
features = [c for c in df.columns if c not in drop]
X_train, y_train = train[features], train["isFraud"]
X_val, y_val = val[features], val["isFraud"]
print("Train:", X_train.shape, " Val:", X_val.shape, " Features:", len(features))

# Fraud is 3.5%, so tell the models to treat each fraud as worth more
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def recall_at_precision(y_true, scores, min_precision):
    # Of all frauds, how many do we catch if at least min_precision of our flags must be correct?
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= min_precision
    return recall[:-1][ok].max() if ok.any() else 0.0


results = []


def evaluate(name, scores):
    results.append({
        "model": name,
        "PR-AUC": average_precision_score(y_val, scores),
        "ROC-AUC": roc_auc_score(y_val, scores),
        "recall@prec50": recall_at_precision(y_val, scores, 0.50),
        "recall@prec80": recall_at_precision(y_val, scores, 0.80),
    })
    print(name, {k: round(v, 4) for k, v in results[-1].items() if k != "model"})


# 2. Logistic Regression: needs imputation and scaling (fitted on train only)
print("\nTraining Logistic Regression...")
logreg = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=300),
)
logreg.fit(X_train, y_train)
evaluate("Logistic Regression", logreg.predict_proba(X_val)[:, 1])

# 3. Random Forest: needs imputation, no scaling
print("\nTraining Random Forest...")
rf = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
                           class_weight="balanced_subsample", n_jobs=-1, random_state=42),
)
rf.fit(X_train, y_train)
evaluate("Random Forest", rf.predict_proba(X_val)[:, 1])

# 4. XGBoost: handles missing values itself. Validation rows pick the number of trees.
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.5,
    scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
    early_stopping_rounds=100, n_jobs=-1, random_state=42,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
evaluate("XGBoost", xgb_model.predict_proba(X_val)[:, 1])

# 5. LightGBM: handles missing values itself. Validation rows pick the number of trees.
print("\nTraining LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=5000, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5, scale_pos_weight=pos_weight, n_jobs=-1, random_state=42, verbose=-1,
    metric="average_precision",  # early stopping watches PR-AUC only (default logloss stopped it at tree 1)
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
evaluate("LightGBM", lgb_model.predict_proba(X_val)[:, 1])

# 6. Save the results table and LightGBM's feature importance
os.makedirs("results", exist_ok=True)
table = pd.DataFrame(results).round(4)
table.to_csv("results/baseline_results.csv", index=False)
print("\n", table.to_string(index=False))

importance = pd.DataFrame({
    "feature": features,
    "gain": lgb_model.booster_.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False)
importance["share_%"] = (100 * importance["gain"] / importance["gain"].sum()).round(2)
importance.to_csv("results/lgbm_feature_importance.csv", index=False)
print("\nTop 20 features (LightGBM, by gain):")
print(importance.head(20).to_string(index=False))
