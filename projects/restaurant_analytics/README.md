# Restaurant Performance and Food Sales Analysis

A practical, commercial restaurant intelligence dashboard evaluating sales revenue, ordering channels, operational rush hours, menu engineering profitability, and chronological demand forecasting across multi-outlet operations.

## 1. Problem Statement & Business Objectives
Hospitality managers frequently operate with disconnected transaction data, struggling to distinguish volume drivers from true profit contributors. To optimize menu pricing and kitchen staffing, an operations team needs to:
1. Accurately measure **Net Revenue**, **Estimated Gross Profit**, and **Average Order Value (AOV)**, accounting for multi-item orders and applied discounts.
2. Evaluate sales performance across outlets, ordering channels (Dine-in, Takeout, Delivery, Mobile App), and time of day.
3. Classify menu items using **Menu Engineering Matrices** (Popularity vs. Profitability) and **ABC Pareto Analysis**.
4. Generate chronological baseline forecasts to guide kitchen prep and shift scheduling without opaque "black box" models.

## 2. Dataset Attribution & Generation
- **Data Mode:** Deterministic Synthetic Demonstration Dataset (~15,000 line items across 7,500 orders covering 2025).
- **Attribution Notice:** This dataset is a simulated operational model designed to mirror real POS (Point of Sale) dynamics. It does not represent actual proprietary figures from any single commercial entity.
- **Reproducibility:** Generated with fixed seed `42` via `projects/restaurant_analytics/data.py`. No external API keys or cloud databases required.

## 3. Financial Metric Formulations
Accounting integrity requires precise definitions:
- **Order Count:** Calculated strictly via `nunique(order_id)`. Multiple line items belonging to one order do not inflate the order count.
- **Gross Sales:** $\sum (\text{Quantity} \times \text{Unit Price})$
- **Net Revenue:** $\sum (\text{Gross Sales} - \text{Discount Amount})$
- **Cost of Goods Sold (COGS):** $\sum (\text{Quantity} \times \text{Unit Cost})$
- **Estimated Gross Profit:** $\text{Net Revenue} - \text{COGS}$
- **Estimated Gross Margin:** $\frac{\text{Estimated Gross Profit}}{\text{Net Revenue}}$
- **Average Order Value (AOV):** $\frac{\text{Net Revenue}}{\text{Total Distinct Orders}}$

## 4. Analytical Modules
### A. Temporal & Operational Rush Analysis
- **Hourly Dynamics:** Bimodal lunch peak (11:30–14:00) and dinner rush (17:30–21:30) highlighting staffing requirements.
- **Weekday vs. Weekend Lift:** Quantifies Friday/Saturday evening volume surges versus weekday baseline.

### B. Menu Engineering & ABC Analysis
- **Quadrant Matrix:** Categorizes items across median volume and median gross profit:
  - **Stars (High Volume, High Margin):** Top revenue drivers; protect recipe consistency and presentation.
  - **Plowhorses (High Volume, Low Margin):** High popularity with lower unit margins; prime candidates for modest price increments or portion adjustments.
  - **Puzzles (Low Volume, High Margin):** High-margin specialty dishes needing promotional visibility or menu redesign.
  - **Dogs (Low Volume, Low Margin):** Underperforming items evaluated for replacement.
- **ABC Classification:** Ranks items by cumulative revenue share:
  - **Class A:** Top items generating up to 70% of revenue.
  - **Class B:** Intermediate items generating next 20% (70%–90%).
  - **Class C:** Long-tail items generating bottom 10%.

### C. Chronological Sales Forecasting
- **Methodology:** 7-day chronological rolling trend with trailing slope extrapolation.
- **Evaluation Discipline:** Preserves strict chronological order (no random temporal shuffling).
- **Safety Guard:** Requires at least 14 days of historical data to generate reliable forward estimates.

## 5. Custom CSV Upload Support
Users can test their own transaction data by uploading a CSV in the sidebar. The uploaded file is validated for:
- Required columns: `order_id`, `line_item_id`, `order_timestamp`, `outlet`, `ordering_channel`, `payment_method`, `item_name`, `category`, `quantity`, `unit_price`, `unit_cost`, `discount_rate`, `net_revenue`, `estimated_gross_profit`.
- Positive values for quantities and prices.
- Valid ISO datetime strings.
If validation fails, clear feedback is provided while safely preserving the demo dataset.

## 6. How to Run
From the repository root:
```bash
streamlit run projects/restaurant_analytics/app.py
```
Or run the automated tests:
```bash
pytest projects/restaurant_analytics/tests/
```
