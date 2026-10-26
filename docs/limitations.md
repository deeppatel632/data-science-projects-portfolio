# Analytical Limitations & Governance Framework

Every responsible data science portfolio must explicitly document what its models and datasets **cannot** do.

---

## 1. Customer Behavioral Segmentation (Project 1)
- **Session-Level vs. Human-Level Identity:**
  - The dataset tracks individual web sessions, not persistent individual customers.
  - A user who browses on a phone during lunch and completes the purchase on a desktop in the evening appears as two unrelated sessions: one "non-converting" and one "high-intent converter".
  - Do not use this clustering to infer persistent customer lifetime value (LTV) or long-term loyalty personalities.
- **Non-Causal Catalog Engagement:**
  - Observing that purchasing sessions visit more product pages does not prove that forcing visitors to view more pages will cause them to purchase.
- **Technical Context Drift:**
  - Browser and operating system identifiers reflect the technology landscape of 2019 and may not represent contemporary mobile traffic distributions.

---

## 2. Student Academic Performance (Project 2)
- **Ethical Mandate Against Automated Decision-Making:**
  - Algorithmic grade estimates must **never** be used as the sole or primary basis for grades, course placement, academic probation, or university admissions.
  - Predictive outputs represent historical statistical correlations, not deterministic human potential.
- **Narrow Institutional Cohort:**
  - Data originates from two Portuguese secondary schools in 2005–2006. Educational systems, grading standards, and socioeconomic factors differ significantly across institutions and eras.
- **Self-Reported Covariates:**
  - Variables such as daily/weekly alcohol consumption (`Dalc`, `Walc`) and study hours are self-reported survey answers subject to social desirability bias.
- **Zero-Grade Discontinuities:**
  - Final grades of zero often reflect administrative dropouts, medical non-attendance, or exam absences rather than gradual academic struggle.

---

## 3. Restaurant Sales & Menu Analytics (Project 3)
- **Synthetic Data Boundaries:**
  - The demonstration data is deterministically simulated and calibrated to mirror typical casual dining restaurant patterns. It should not be cited as real proprietary commercial benchmarks.
- **Unit Cost Assumptions:**
  - Food costs are modeled as fixed baseline estimates per item and do not incorporate real-time commodity price fluctuations, bulk supplier rebate tiers, or kitchen spoilage wastage rates.
- **Linear Demand Assumptions:**
  - The 14-day rolling demand baseline assumes stationary macro conditions and will not anticipate unexpected disruptions, severe weather closures, or unmodeled holiday surges.
