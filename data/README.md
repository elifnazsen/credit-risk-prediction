# Data

`german_credit_raw.csv` - the **Statlog (German Credit Data)** dataset from the UCI Machine Learning Repository
(Hofmann, 1994): 1,000 loan applicants, 20 attributes, no missing values.

- Original: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- This file is a version with readable category labels (instead of codes such as `A11`) and English column names.
  It was downloaded from the `selva86/datasets` GitHub repository.
- Target in this file is `credit_risk` (1 = good, 0 = bad). The project creates `Default = 1 - credit_risk`.
- The dataset has **no income variable**.
