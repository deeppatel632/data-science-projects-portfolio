# Student Academic Performance Analysis and Visualization

An interpretable educational data science study examining secondary school student achievement across Mathematics and Portuguese language courses, comparing early-warning baseline models against mid-term trajectory estimation with strict non-causal governance.

## 1. Problem Statement & Research Objectives
Educational institutions seek to understand student achievement patterns to provide timely academic support. However, applying predictive modeling to academic grading carries profound ethical and methodological challenges:
1. **Timing Asymmetry:** Static student background characteristics (parent education, demographics) offer only weak signals at matriculation, while interim period grades ($G1$, $G2$) strongly dictate final marks ($G3$).
2. **Causal Fallacy:** Correlation between non-academic factors (e.g., home internet, family size) and grades must not be mistaken for causal influence.
3. **Surveillance vs. Support:** Machine learning in education must serve as an early warning for constructive academic tutoring, never as automated gatekeeping or punitive tracking.

## 2. Dataset Attribution & Provenance
- **Dataset:** UCI Student Performance Dataset
- **Repository URL:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Citation:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Cohorts Included:**
  - Mathematics (`student-mat.csv`): 395 students.
  - Portuguese Language (`student-por.csv`): 649 students.

## 3. Data Architecture & Preprocessing
- **Target Variable:** Final academic grade ($G3$), measured on the standard Portuguese secondary scale of **0 to 20** (10 is passing).
- **Features Handled:** 33 attributes spanning demographic attributes (`school`, `sex`, `age`, `address`), family background (`Medu`, `Fedu`, `Mjob`, `Fjob`), study habits (`studytime`, `failures`), social activities (`freetime`, `goout`, `Dalc`, `Walc`), health, absences, and period grades (`G1`, `G2`).
- **Scikit-Learn Preprocessing Pipeline:**
  - Standard scaling (`StandardScaler`) for all numerical covariates.
  - One-hot encoding (`OneHotEncoder(handle_unknown='ignore')`) for categorical variables.

## 4. Modeling Honesty & Two-Scenario Design
To avoid artificial model inflation, this project contrasts two strictly distinct operational regimes:
1. **Early-Support Scenario (G1 & G2 Excluded):** Models predict $G3$ strictly using background attributes, study habits, and past failures before the student sits for any course exams. Realistic $R^2 \approx 0.15 - 0.25$, reflecting the high inherent uncertainty of human academic outcomes.
2. **Previous-Grades Scenario (G1 & G2 Included):** Models predict $G3$ incorporating interim exam marks. Test error drops dramatically ($R^2 \approx 0.75 - 0.85$, $RMSE \approx 1.5 - 2.0$ grades), demonstrating that prior academic achievement dominates background covariates.
*(Target $G3$ is never used as a predictive feature in either regime).*

### Algorithms Compared
- **Dummy Regressor:** Constant baseline predicting cohort mean grade.
- **Linear Regression (Ridge):** $L_2$-regularized linear model with interpretable coefficients.
- **Random Forest Regressor:** Non-linear ensemble (100 estimators, max depth 7) capturing feature interactions.

## 5. Key Findings
- **Dominance of Prior Exam Scores:** $G1$ and $G2$ correlate strongly with $G3$ ($r \approx 0.80 - 0.90$), accounting for the majority of explainable variance.
- **Impact of Past Course Failures:** Students with past academic failures average 3–4 points lower on final grades, identifying repeat students as a primary group benefiting from structured mentoring.
- **Study Time Gradient:** Students reporting 5+ hours of weekly study averaged 1.5–2.0 points higher than peers reporting <2 hours/week.

## 6. Limitations & Educational Ethics
- **Narrow Institutional Sample:** Data originates from two specific Portuguese secondary schools in 2005–2006. Findings should not be assumed to generalize across different cultural, pedagogical, or socioeconomic contexts.
- **Non-Causal Interpretation:** Correlated variables (e.g., parental education, internet access) do not imply simple causal levers.
- **Ethical Mandate:** Predictions are statistical approximations and must never be used for high-stakes decisions such as grading, admissions, or disciplinary tracking.

## 7. How to Run
From the repository root:
```bash
streamlit run projects/student_performance/app.py
```
Or run the automated test suite:
```bash
pytest projects/student_performance/tests/
```
