# Data Preprocessing: What We Did, Why, and How to Defend It

Project PW_25_MAR_02 · IEEE-CIS Fraud Detection dataset
All numbers below were computed from the actual data files in this project.

---

## 1. The dataset in one table

| Item | Value |
|---|---|
| Source | Kaggle IEEE-CIS Fraud Detection (`train_transaction.csv` + `train_identity.csv`) |
| Transactions | 590,540 |
| Time span | 182 days (TransactionDT is seconds from an unknown start) |
| Columns after joining the two tables | 434 (394 transaction + 41 identity, minus the shared ID) |
| Fraud cases | 20,663 (3.50%), so about 1 fraud per 28 normal |
| Rows with identity info | 24.4% |
| Duplicate TransactionIDs | 0 |
| Fully duplicated rows (ignoring ID) | 3 (negligible, kept) |
| Zero or negative amounts | 0 |
| Infinite values | 0 |
| Cells that are missing | 43.8% overall |

**What the columns are**

| Group | Count | Meaning |
|---|---|---|
| V1–V339 | 339 | Anonymised engineered features from Vesta (the dataset's provider) |
| C1–C14 | 14 | Counting features (e.g. how many addresses are linked to a card) |
| D1–D15 | 15 | Time-delta features (days since some earlier event) |
| M1–M9 | 9 | Match flags (T/F: do name, address, etc. match) |
| card1–card6 | 6 | Card details |
| addr1–2, dist1–2 | 4 | Address and distance |
| id_01–id_38 | 29 kept (9 dropped) | Identity: device, browser, screen, network |
| Others | | TransactionAmt, ProductCD, email domains, DeviceType, DeviceInfo |

---

## 2. The preprocessing pipeline

```
Raw train_transaction (590,540 x 394) + train_identity (144,233 x 41)
        │
        ▼  [1] LEFT JOIN on TransactionID (keep every transaction)
Merged table (590,540 x 434)
        │
        ▼  [2] DROP columns that are >95% empty        (9 columns removed)
        │
        ▼  [3] ENCODE text columns as numbers          (29 columns)
        │
        ▼  [4] ADD time-based split column             (70% / 15% / 15%)
        │
        ▼  [5] SAVE as data/processed/clean.parquet    (590,540 x 426)
```

Code: `src/03_preprocess.py`.

---

## 3. Each step: what, why, and the alternative we considered

### Step 1: Join the two tables
- **What:** left join on `TransactionID`, so every transaction is kept even without identity info.
- **Why:** only 24.4% of transactions have identity records. An inner join would throw away three quarters of the data.
- **Side effect worth knowing:** "no identity record" is itself information. Missing identity can correlate with fraud, and tree models can learn from that.

### Step 2: Drop columns that are more than 95% missing
- **What:** removed 9 columns: `id_07, id_08, id_21, id_22, id_23, id_24, id_25, id_26, id_27`.
- **Why:** a column that is 99% empty carries almost no usable information and only adds noise and training time.
- **Why 95%?** It is a judgement call. A stricter cut (for example 50%) would remove 205 columns, many of which are still useful, because "missing" can itself be a signal. We chose a conservative cut that only removes columns that are essentially empty.
- **Honest limitation:** the threshold was computed on all rows, including validation and test. No labels were used, so this is not label leakage, but a strict pipeline would compute it on training rows only. Before the final report we should re-run it on the training split only and confirm the same 9 columns are dropped.

### Step 3: Encode text columns as numbers
- **What:** 29 text columns (e.g. `ProductCD`, `card4`, `card6`, email domains, `M1–M9`, `DeviceInfo`) are replaced by integer codes. A missing value stays missing (NaN).
- **Why:** XGBoost, LightGBM, scikit-learn models and GNNs need numbers.
- **Why not one-hot encoding?** `DeviceInfo` has 1,786 distinct values, `id_33` 260, `id_31` 130. One-hot would add thousands of columns.
- **Honest limitation:** integer codes create an arbitrary order (code 5 > code 3 means nothing). Tree models cope with this, linear models do not. For Logistic Regression this encoding is weak, which is acceptable because it is only a baseline. For the GNN we will revisit this (for example, learned embeddings or frequency encoding).
- **Leakage check:** the codes are built from the text values only, never from the fraud label, so there is no target leakage.

### Step 4: Time-based split
- **What:** transactions sorted by `TransactionDT`; the first 70% are train (to about day 121), the next 15% validation (to about day 152), the last 15% test.
- **Why:** a bank predicts the future from the past. A random split lets the model peek at the future and gives inflated scores. The literature review also showed the fraud rate drifts over time (2.5% in the first 30 days, about 4% in months 2–4, about 3.4% later), so a time split is the honest test.
- **Result:**

| Split | Rows | Fraud rate |
|---|---|---|
| Train | 413,378 | 3.52% |
| Validation | 88,581 | 3.43% |
| Test | 88,581 | 3.48% |

- **Why a stored column?** Every model (baselines and GNN) uses exactly the same rows, so the comparison is fair and reproducible.
- **Rule:** test data is used once, for final reporting. All tuning and early stopping use validation.

### Step 5: Save as Parquet
- **Why:** much faster to load than CSV, and the types are preserved.

---

## 4. How the remaining issues are handled (and where)

| Issue | How handled | Where |
|---|---|---|
| **Missing values** | Left as NaN in the saved file. XGBoost and LightGBM learn how to treat NaN natively. Logistic Regression and Random Forest use median imputation, fitted on training data only, inside a pipeline. | Model training (`04_baselines.py`) |
| **Class imbalance (3.5% fraud)** | No resampling. We use class weights (`class_weight`, `scale_pos_weight`) and judge by PR-AUC and recall at fixed precision, not accuracy. | Model training and evaluation |
| **Scaling** | Only for Logistic Regression (StandardScaler, fitted on train). Tree models do not need it. | Baselines pipeline |
| **Concept drift** | Time-based split exposes it. | Step 4 |
| **Skewed amounts** | `TransactionAmt` has skew 14.4 (median 68.77, mean 135.03, max 31,937). Left as is for tree models. A log transform is planned for the GNN. | Not yet applied |
| **Outliers** | Not removed. Large amounts can be genuine fraud signals, and tree models are robust to them. | Deliberate choice |
| **Duplicates** | 3 fully duplicated rows, kept (negligible). | None |

---

## 5. What we have NOT done yet (say this openly)

1. **No feature engineering yet.** Candidates: hour of day and day of week from `TransactionDT`, log of amount, a better card identifier (combine `card1`, `card2`, `addr1`), and counts of how often a card or address appears.
2. **The 339 V-columns are untouched.** They are heavily correlated. Dimensionality reduction or selection could help, but tree models handle them acceptably.
3. **Drop threshold not yet re-checked on training data only** (see Step 2).
4. **Imputation for the GNN** is not designed yet. It will need explicit imputation plus "was missing" indicator columns.
5. **Graph construction is a separate step.** Which columns become nodes: `card1` and `DeviceInfo` look useful; email domains (59 values) and `ProductCD` (5 values) are too coarse and would act as giant hubs. The graph will be built from training rows only, to avoid leakage.

---

## 6. Likely panel questions and short answers

**Q1. How did you handle missing values?**
They are 43.8% of all cells, so we did not delete rows. We dropped 9 columns that are over 95% empty. For the rest, boosted-tree models handle missing values natively, and Logistic Regression and Random Forest use median imputation fitted on training data only. Missingness itself can be informative, so we did not hide it.

**Q2. How did you handle class imbalance?**
Fraud is 3.5%. We did not oversample, because synthetic samples can distort relational structure. Instead we use class weights and evaluate with PR-AUC and recall at fixed precision. Plain accuracy would reach 96.5% by predicting "not fraud" every time.

**Q3. How do you prevent data leakage?**
Three ways. (1) The split is by time, so no future information reaches training. (2) Imputers and scalers are fitted on training data only. (3) Encoding uses text values only, never the label. The graph will also be built from training transactions only. One item we still need to tighten: the missing-value threshold was computed on all rows.

**Q4. Why split by time and not randomly?**
In deployment a model predicts new transactions from past ones. A random split mixes future and past and overstates performance. The fraud rate also drifts over the 182 days, which only a time split reveals.

**Q5. How did you encode categorical variables?**
Integer codes, with missing values preserved. One-hot would create thousands of columns for fields like `DeviceInfo` (1,786 values). The codes have no natural order, which suits tree models; we will revisit encoding for the GNN.

**Q6. Did you remove outliers?**
No, deliberately. Amounts are heavily skewed (max 31,937 vs median 68.77), but unusual amounts can be genuine fraud signals, and tree models are not sensitive to scale.

**Q7. Why did you drop those 9 columns?**
They are 99% missing, so they carry almost no information. We used a conservative 95% threshold, because moderate missingness can still be informative.

**Q8. What are the V columns?**
339 anonymised features engineered by the dataset's provider. We cannot interpret them individually, which is a limitation for explainability, so our explanations will focus on graph structure and interpretable columns.

**Q9. Why a left join?**
Only 24.4% of transactions have identity information. An inner join would discard 75.6% of the data.

**Q10. How do you know your results are fair across models?**
All models use the same stored train, validation and test split, and the same feature set.

---

## 7. Suggested figures for the report or slides

1. The pipeline diagram in Section 2.
2. Fraud rate by 30-day block (shows drift).
3. Split table (rows and fraud rate per split).
4. Missing-value bar chart for the top 20 columns.
5. A before and after table: 434 columns to 426, and 29 text columns encoded.
