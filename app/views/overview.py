import streamlit as st
from app.data_loader import load
from app.components import metrics, report, source_links

def render():
    st.title('Can process measurements help catch quality failures?')
    st.markdown('A semiconductor screening case study combining **materials engineering judgment, Python, '
                'statistics, and machine learning**. The aim was to identify predictive associations in anonymous '
                'process measurements while controlling leakage and making review workload explicit.')
    st.subheader('One frozen method. One final test.')
    result = load('test')
    metrics(result)
    st.info(f'Caught {result["TP"]} of {result["failures"]} failures, with {result["FP"]} false alarms '
            f'and {result["FN"]} missed failures. Of {result["flagged"]} alerts, {result["TP"]} were failures.')
    st.markdown('**Why it matters:** screening can prioritize investigation, but missed failures and unnecessary '
                'reviews both carry engineering costs. This study demonstrates a tradeoff, not proven yield improvement.')
    st.markdown('**Explore next:** Dataset & Workflow for the method, Threshold Tradeoffs for the decision, '
                'and Final Test Results for the independent check.')
    source_links('test', 'frozen')
    report('recruiter', 'One-minute project summary')
    report('technical', 'Technical engineering summary')
