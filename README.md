# Loan Approval Prediction: A Data-First Project (EDA, Cleaning & Leakage-Free Preprocessing)

Most of the effort in this project went into the **data**, not the model. The goal was to take a raw, messy loan-application table and turn it into a clean, trustworthy, model-ready dataset, with every decision justified and checked, and then confirm it with a simple ensemble.

**What this project demonstrates:**
- Auditing a raw dataset (types, duplicates, missing values, skew, class balance)
- Classifying features by *meaning* instead of trusting pandas dtypes
- Comparing outlier strategies (IQR x1.5 vs x3 vs business caps) with evidence
- A **split-first, fit-on-train-only** pipeline that avoids data leakage
- A **custom KNN-mode imputer** for discrete/categorical columns
- Encoding and scaling choices matched to each feature's nature

---

## Table of Contents
1. [Dataset at a Glance](#dataset-at-a-glance)
2. [Data Pipeline Summary](#data-pipeline-summary)
3. [Step 1: Data Audit](#step-1-data-audit)
4. [Step 2: Exploratory Data Analysis](#step-2-exploratory-data-analysis)
5. [Step 3: Outlier Analysis](#step-3-outlier-analysis)
6. [Step 4: Train/Test Split (Leakage Prevention)](#step-4-traintest-split-leakage-prevention)
7. [Step 5: Custom KNN-Mode Imputation](#step-5-custom-knn-mode-imputation)
8. [Step 6: Encoding & Scaling](#step-6-encoding--scaling)
9. [Step 7: Model Validation](#step-7-model-validation)
10. [Limitations & Next Steps](#limitations--next-steps)
11. [Repository Structure & How to Run](#repository-structure--how-to-run)

---

## Dataset at a Glance

| Item | Detail |
|---|---|
| **Task** | Binary classification, `Loan_Status` (Y = 1, N = 0) |
| **Raw shape** | 598 rows x 12 columns (after dropping the `Loan_ID` identifier) |
| **Class balance** | 411 Approved vs 187 Rejected (about 69% / 31%) |
| **Source file** | `LoanApprovalPrediction.csv` |

---

## Data Pipeline Summary

```
Raw data (598 x 12)
   │
   ├─ Audit: dtypes, duplicates, missing values, summary stats
   ├─ EDA: target balance, distributions, features vs target
   ├─ Outlier handling: business caps  ->  562 rows
   │
   ├─ Stratified 80/20 split  ->  449 train / 113 test
   │
   ├─ (fit on TRAIN only) KNN-mode imputation  ->  0 missing values
   ├─ (fit on TRAIN only) Ordinal + One-Hot encoding  ->  29 features
   └─ (fit on TRAIN only) RobustScaler  ->  model-ready
```

| Stage | Rows | Features | Missing values |
|---|---|---|---|
| Raw | 598 | 11 + target | 96 cells in 4 columns |
| After outlier caps | 562 | 11 + target | 3 columns (see Step 5) |
| After imputation | 562 | 11 + target | **0** |
| After encoding | 562 | **29** | 0 |

---

## Step 1: Data Audit

**Checks performed:** shape, dtypes, duplicate rows (**0 found**), full `describe()` for numeric and categorical columns.

### 1.1 Features classified by meaning, not by dtype
Pandas stores `Credit_History`, `Dependents` and `Loan_Amount_Term` as numbers, but they behave like **categories/discrete codes**, so treating them as continuous would be wrong (e.g. an average credit history of 0.84 is meaningless as a quantity).

| Group | Features |
|---|---|
| **Continuous** | `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount` |
| **Categorical / discrete** | `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Credit_History`, `Property_Area`, `Loan_Amount_Term` |

### 1.2 Missing values (raw)

| Column | Missing | % |
|---|---|---|
| `Credit_History` | 49 | 8.19% |
| `LoanAmount` | 21 | 3.51% |
| `Loan_Amount_Term` | 14 | 2.34% |
| `Dependents` | 12 | 2.01% |

### 1.3 What the summary statistics revealed
- `ApplicantIncome`: median **3,806** but max **81,000** (mean 5,292, std 5,807): heavy right skew.
- `CoapplicantIncome`: median 1,211 but max **41,667**; over 25% of values are 0.
- `LoanAmount`: median 127, max 650.
- `Loan_Amount_Term`: median/mode **360**, but values go as low as 12.

> **Takeaway:** income and loan features are strongly skewed with extreme values, which motivates outlier analysis and a robust scaler later.

---

## Step 2: Exploratory Data Analysis

The raw dataframe is **kept unchanged** during EDA; all fixes happen later, after the split.

- **Target balance:** bar chart of `Loan_Status` confirming a 69/31 imbalance, which justifies a *stratified* split and watching minority-class metrics.
- **Continuous features:** histogram + KDE for each, confirming right-skewed distributions.
- **Categorical vs target:** count plots of every categorical feature split by `Loan_Status` to see which groups are approved more often.
- **Correlation heatmap:** continuous features + encoded target, used to compare relationships before/after outlier removal (Step 3).

---

## Step 3: Outlier Analysis

Instead of blindly deleting rows, three strategies were compared.

### 3.1 Business-logic caps (applied)
Records beyond these thresholds were removed as implausible/extreme:

`ApplicantIncome <= 20,000`, `CoapplicantIncome <= 10,000`, `LoanAmount <= 700`

**Effect:** 598 -> **562 rows** (36 removed).
> Note: in pandas, `NaN <= 700` evaluates to `False`, so this filter also removes the **21 rows with a missing `LoanAmount`**. That means `LoanAmount` ends up with no missing values, and the other 15 removed rows are true extreme-value records. This is why imputation (Step 5) only needs to handle three columns.

### 3.2 IQR fences compared (after the caps)

| Feature | IQR x 1.5: upper bound | Outliers | IQR x 3: upper bound | Outliers |
|---|---|---|---|---|
| `ApplicantIncome` | 9,851 | 40 | 14,027 | **19** |
| `CoapplicantIncome` | 5,703 | 11 | 9,124 | **0** |
| `LoanAmount` | 255 | 35 | 348 | **10** |

The standard 1.5x rule flags a large share of ordinary high earners as "outliers" in a right-skewed distribution, so removing them would throw away legitimate information. The 3x rule isolates only the genuinely extreme points.

### 3.3 Impact on relationships (`df2`, 3xIQR removal)
Removing 3xIQR outliers from a copy (**538 rows**, 24 removed) and re-plotting the correlation heatmap showed how much extreme values distort feature-to-feature and feature-to-target correlations.

### 3.4 Final decision
- Keep the **business-cap dataset (562 rows)** for modeling, so as not to over-trim an already small dataset.
- Handle the remaining skew with a **RobustScaler** (median/IQR-based) rather than deleting more data.
- The aggressive 3xIQR copy (`df2`) was used for **analysis only**, not for training.

---

## Step 4: Train/Test Split (Leakage Prevention)

The split happens **before** any imputation, encoding or scaling.

```
stratified 80/20 split, random_state=42  ->  X_train (449, 11) | X_test (113, 11)
```

- **Stratified** on `Loan_Status` so both sets keep the 69/31 ratio.
- Every transformer below is **fit on the training set only** and then only *applied* to the test set. Test-set statistics never influence preprocessing.

**Missing values after the split (before imputation):**

| Column | Missing in train |
|---|---|
| `Credit_History` | 37 |
| `Loan_Amount_Term` | 13 |
| `Dependents` | 11 |

60 of 449 training rows (and 9 of 113 test rows) contain at least one missing value; the remaining **389 complete training rows** become the reference pool for imputation.

---

## Step 5: Custom KNN-Mode Imputation

A mean/median fill makes no sense for discrete columns like `Credit_History` (0/1) or `Loan_Amount_Term`, and a global mode ignores each applicant's profile. So a **custom imputer** was written:

1. **Build a distance space** from all features:
   - numeric -> median-imputed + standardized
   - categorical -> mode-imputed + one-hot encoded
2. For each row with missing values, find its **k = 3 nearest complete training rows** (Euclidean distance).
3. Fill each missing value with the **mode among those 3 neighbors**.
4. **Fallback:** if neighbors give no mode, use the global training mode.
5. **Test rows** are imputed using the *training* complete rows as the neighbor pool (no test-to-test leakage).

**Verification:** after imputation, both `X_train` and `X_test` were asserted to have **zero remaining nulls**.

**Why it's better than simple imputation:** an applicant with a similar income, loan size, area and employment profile is more likely to share their credit history than a random applicant is.

---

## Step 6: Encoding & Scaling

| Feature type | Method | Reason |
|---|---|---|
| `Education` | **Ordinal** (`Not Graduate = 0`, `Graduate = 1`) | Natural order, only 2 levels |
| `Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`, `Property_Area`, `Loan_Amount_Term` | **One-Hot** (`handle_unknown="ignore"`) | No ordinal meaning; safe for unseen test categories |
| `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount` (+ `Education`) | **RobustScaler** | Uses median & IQR, so extreme values don't squash the rest of the distribution |

**Result:** a final matrix of **29 features** (4 scaled numeric + 25 one-hot columns): train `(449, 29)`, test `(113, 29)`, no missing values, no leakage.

---

## Step 7: Model Validation

A small model was trained *only* to validate that the prepared data is sound and informative: a **Soft Voting Ensemble** (Logistic Regression, SVM, Decision Tree `max_depth=5`, KNN `k=3`), plus a Random Forest for feature importance.

**Soft Voting Ensemble (113 test rows):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (Rejected) | 0.95 | **0.53** | 0.68 | 34 |
| 1 (Approved) | 0.83 | 0.99 | 0.90 | 79 |
| **Accuracy** | | | **0.85** | 113 |

The pipeline yields a model that clearly learns signal (85% accuracy), but it is much better at spotting approvals than rejections. Accuracy is partly lifted by the majority class, so **class-0 recall and macro-F1 are the honest metrics**.

*Random Forest results: add accuracy, OOB score and feature-importance chart after running the final notebook cell.*

---

## Limitations & Next Steps

- [ ] Address minority-class recall: `class_weight="balanced"`, SMOTE *inside* a CV pipeline, or threshold tuning.
- [ ] Replace the single split with **stratified k-fold cross-validation**.
- [ ] Wrap the whole preprocessing in a single `sklearn` `Pipeline`/`ColumnTransformer` for reuse and deployment.
- [ ] Test engineered features (e.g. total income = applicant + coapplicant, loan-to-income ratio, EMI).
- [ ] Evaluate with ROC-AUC / PR-AUC and a confusion matrix.
- [ ] Compare the KNN-mode imputer against simple/iterative imputers to quantify its benefit.

---

## Repository Structure & How to Run

```
.
├── LoanApprovalPrediction.csv
├── 01_eda_preprocessing_modeling.ipynb
├── images/
│   ├── encoder2_loan_status_distribution.png
│   ├── encoder2_numerical_boxplots.png
│   ├── encoder2_numerical_boxplots_2.png
│   └── encoder2_heatmap.png
├── requirements.txt
└── README.md
```

```bash
git clone https://github.com/<your-username>/loan-approval-prediction-eda-ml.git
cd loan-approval-prediction-eda-ml
pip install -r requirements.txt
jupyter notebook 01_eda_preprocessing_modeling.ipynb
```

`requirements.txt`: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

**Tech stack:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## Author
**Abdelrhman Mustafa Kolkela**: AI student (Data Science & ML), Assiut National University.
