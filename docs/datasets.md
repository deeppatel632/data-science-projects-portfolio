# Dataset Provenance, Attribution, and Schema Documentation

All datasets bundled with this portfolio are either officially licensed real benchmark datasets or reproducible, calibrated synthetic datasets.

---

## 1. Online Shoppers Purchasing Intention Dataset (Project 1)
- **Source:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Academic Citation:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31, 6893–6908.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Format:** CSV (`projects/customer_segmentation/data/online_shoppers_intention.csv`).
- **Dimensions:** 12,330 rows, 18 columns.
- **Key Columns:**
  - `Administrative`, `Administrative_Duration`: Page views and time spent in account/management sections.
  - `Informational`, `Informational_Duration`: Page views and time spent in policy/FAQ sections.
  - `ProductRelated`, `ProductRelated_Duration`: Page views and time spent in product catalog browsing.
  - `BounceRates`: Percentage of visitors entering the site from that page who immediately leave.
  - `ExitRates`: Percentage of page views that were the last in the session.
  - `PageValues`: Average value of pages visited prior to transaction (excluded in honest models to avoid leakage).
  - `SpecialDay`: Closeness of site visiting time to a specific holiday.
  - `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`: Technical & temporal session context.
  - `Revenue`: Binary boolean indicator of session purchase completion (Target).

---

## 2. Student Performance Dataset (Project 2)
- **Source:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Academic Citation:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Files:**
  - `projects/student_performance/data/student-mat.csv`: 395 students in Mathematics course.
  - `projects/student_performance/data/student-por.csv`: 649 students in Portuguese language course.
- **Key Attributes:**
  - `school`: Student's school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira).
  - `sex`, `age`, `address`, `famsize`, `Pstatus`: Demographic and family structure.
  - `Medu`, `Fedu`, `Mjob`, `Fjob`: Parental education and employment.
  - `studytime`: Weekly study hours (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h).
  - `failures`: Number of past class failures ($n \in [0, 3]$).
  - `absences`: Total days of absence.
  - `G1`, `G2`: First and second period evaluation marks (0–20).
  - `G3`: Final year grade (0–20, Target).

---

## 3. Calibrated Restaurant Sales Dataset (Project 3)
- **Source:** Reproducible, seeded synthetic generation (`projects/restaurant_analytics/data.py`).
- **Attribution:** Synthetic demonstration dataset designed to mirror realistic POS multi-line order dynamics. No proprietary real data is represented.
- **File:** `projects/restaurant_analytics/data/restaurant_sales_sample.csv`.
- **Dimensions:** 14,785 line items across 7,500 distinct orders.
- **Schema:**
  - `order_id`: Unique order identifier (e.g. `ORD-2025-10042`).
  - `line_item_id`: Unique line item key (e.g. `ORD-2025-10042-1`).
  - `order_timestamp`: Full ISO timestamp capturing lunch and dinner peak hours.
  - `outlet`: Location ('Downtown Bistro', 'Uptown Grill', 'Westside Cantina', 'Suburban Cafe').
  - `ordering_channel`: 'Dine-in', 'Takeout', 'Delivery', 'Mobile App'.
  - `payment_method`: 'Credit Card', 'Digital Wallet', 'Cash', 'Gift Card'.
  - `item_name`, `category`: 20 menu offerings across Mains, Appetizers, Desserts, Beverages, Sides.
  - `quantity`, `unit_price`, `unit_cost`, `discount_rate`: Commercial transaction metrics.
  - `gross_sales`, `discount_amount`, `net_revenue`, `cogs`, `estimated_gross_profit`, `gross_margin`: Audited financial calculations.
