# Short recruiter summary

Developed a semiconductor quality-screening project using SECOM's 1,567 records and 590 anonymous process measurements. Addressed missing data and rare failures through leakage-controlled preprocessing, five-fold validation and a reserved final test.

Compared logistic regression, decision trees, Random Forest and gradient boosting, then selected a Random Forest without missingness indicators at threshold 0.35 based on the development tradeoff between failure detection and review workload.

The frozen model's final test achieved **66.7% failure recall, 12.1% precision, 65.2% specificity, 65.9% balanced accuracy and 0.1872 average precision**. It caught 14 failures, missed 7 and produced 102 false alarms.

The project demonstrated useful screening signal but is **not production-ready because of false alarms and missed failures**. Feature importance was interpreted conservatively as predictive association, not physical root cause, showing engineering judgment alongside data-analysis and machine-learning skills.
