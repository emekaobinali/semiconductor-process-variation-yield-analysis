import pandas as pd
import streamlit as st
from app.data_loader import load, selected_development
from app.components import metrics, table, definitions, source_links

def render():
    st.title('Final Test Results')
    st.caption('STATIC RECORD · one reserved-test evaluation after method freeze · no test exploration')
    test = load('test')
    metrics(test)
    st.metric('Average precision', f'{test["average_precision"]:.4f}')
    table(pd.DataFrame([{key: test[key] for key in ['TP', 'FN', 'FP', 'TN', 'flagged']}]))
    dev = selected_development()
    ap = load('models').query('model == "small_forest" and indicators == False').iloc[0].mean_AP
    comparison = pd.DataFrame([{'Evaluation':'Development validation', **{k:dev[k] for k in
        ['recall','precision','specificity','balanced_accuracy']}, 'average_precision':ap},
        {'Evaluation':'Reserved test', **{k:test[k] for k in
        ['recall','precision','specificity','balanced_accuracy','average_precision']}}])
    table(comparison)
    st.write('Performance degraded, especially failure recall and balanced accuracy. '
             'Ranking performance remained relatively similar. False alarms and missed failures prevent a production-ready claim.')
    st.caption('Development AP is a five-fold mean; test AP is computed across the reserved set. '
               'Development results informed selection and are not independent final estimates. '
               'Only 21 test failures limit certainty; raw counts across differently sized sets are not directly comparable.')
    with st.expander('Frozen configuration'):
        frozen = load('frozen')
        st.write(frozen['preprocessing'])
        st.json({k:frozen[k] for k in ['classifier','threshold','decision_rule','missingness_indicators_in_classifier','model_parameters']})
    definitions()
    source_links('test', 'models', 'intermediate', 'frozen')
