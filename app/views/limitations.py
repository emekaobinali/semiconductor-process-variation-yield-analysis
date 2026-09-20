import streamlit as st
from app.components import report, source_links

def render():
    st.title('Limitations')
    st.markdown('''- **Screening signal, not production readiness:** false alarms add workload and missed failures remain.
- **Small failure sample:** only 83 development failures and 21 reserved-test failures.
- **Anonymous measurements:** no supported physical labels, units, causal mechanisms, or root-cause claims.
- **Feature reliance is descriptive:** permutation importance used the final model’s development training data. Correlated features and imputation complicate ranking.
- **Timing and drift remain unresolved:** timestamps represent quality tests; chronological sensitivity analysis was not completed.
- **Scores are uncalibrated:** thresholds are score cutoffs, not established failure probabilities.
- **Operational costs are unknown:** the selected threshold is an approved research tradeoff, not a proven factory optimum.
- **No yield-improvement claim:** the analysis associates measurements with quality outcomes; it does not demonstrate a manufacturing intervention.
- **Reproducibility boundary:** original clean-environment scientific reproduction remains unverified. Raw data, fitted models and record-level predictions are excluded from the public package.''')
    st.info('This app reads public saved aggregates and figures only. It does not load a fitted model, '
            'score records, fit preprocessing, or execute scientific workflows.')
    report('reproducibility', 'Existing scientific reproducibility record')
    report('attribution', 'Data attribution and license separation')
    source_links('technical', 'interpretation')
