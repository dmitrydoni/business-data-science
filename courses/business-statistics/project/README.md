# E-news Express: A/B Testing

## Business Context

E-news Express is an online news portal aiming to expand its subscriber base. The company suspects that the current landing page is not engaging enough to convert visitors into subscribers. The design team created a new landing page, and an A/B test was conducted: 100 users were randomly split into a control group (old page) and a treatment group (new page).

## Objective

Perform statistical analysis (significance level = 5%) to determine the effectiveness of the new landing page by answering:

1. Do users spend more time on the new landing page than on the old one?
2. Is the conversion rate for the new page greater than for the old page?
3. Does conversion status depend on the preferred language?
4. Is the time spent on the new page the same across different language users?

## Data

**Source:** `data/raw/abtest.csv` (100 observations, 6 features)

| Column | Description |
|--------|-------------|
| `user_id` | Unique user ID |
| `group` | `control` or `treatment` |
| `landing_page` | `old` or `new` |
| `time_spent_on_the_page` | Time in minutes spent on the landing page |
| `converted` | Whether the user subscribed (`yes` / `no`) |
| `language_preferred` | Language: English, Spanish, or French |

## Deliverables

- Marimo notebook with full analysis (EDA, hypothesis tests, conclusions)
- HTML export of the notebook
- Business recommendations based on statistical findings

## Rubric

1. **EDA** — problem definition, univariate and bivariate analysis, key observations
2. **Hypothesis test 1** — time spent: visual analysis, hypotheses, test selection, p-value, inference
3. **Hypothesis test 2** — conversion rate: visual analysis, hypotheses, test selection, p-value, inference
4. **Hypothesis test 3** — conversion vs. language: visual analysis, hypotheses, test selection, p-value, inference
5. **Hypothesis test 4** — time vs. language: visual analysis, hypotheses, test selection, p-value, inference
6. **Overall quality** — structure, commented code, visual appeal, conclusion and business recommendations
