import pandas as pd
import streamlit as st
from app.data_loader import load, source_path, SOURCES

NOTICE = 'Research portfolio project — useful screening signal, not production-ready.'
REPO = 'https://github.com/emekaobinali/semiconductor-process-variation-yield-analysis/blob/main/'
LABELS = {'recall': 'Failure recall', 'precision': 'Precision', 'specificity': 'Specificity',
          'balanced_accuracy': 'Balanced accuracy', 'mean_AP': 'Mean fold AP',
          'AP_fold_sd': 'Fold AP SD', 'average_precision': 'Average precision',
          'TP': 'Failures caught', 'FN': 'Failures missed', 'FP': 'False alarms',
          'TN': 'Correct passes', 'flagged': 'Sent for review', 'threshold': 'Threshold',
          'indicators': 'Missingness indicators', 'model': 'Model', 'fold': 'Fold', 'AP': 'Fold AP'}

def source_links(*keys):
    st.caption('Sources: ' + ' · '.join(f'[{key.replace("_", " ")}]({REPO}{SOURCES[key]})' for key in keys))

def figure(key):
    st.image(str(source_path(key)), width='stretch')
    source_links(key)

def report(key, title):
    with st.expander(title):
        st.markdown(load(key))
        source_links(key)

def table(df):
    display = df.copy()
    for column in ['recall', 'precision', 'specificity', 'balanced_accuracy']:
        if column in display:
            display[column] = display[column].map(lambda v: 'Not defined' if pd.isna(v) else f'{v:.1%}')
    for column in ['mean_AP', 'AP_fold_sd', 'average_precision', 'AP']:
        if column in display:
            display[column] = display[column].map(lambda v: f'{v:.4f}')
    st.dataframe(display.rename(columns=LABELS), hide_index=True, width='stretch')

def metrics(row):
    # Two columns keep labels legible on narrower screens.
    for fields in [('recall', 'precision'), ('specificity', 'balanced_accuracy')]:
        for col, key in zip(st.columns(2), fields):
            col.metric(LABELS[key], f'{row[key]:.1%}')

def definitions():
    with st.expander('How to read the metrics'):
        st.markdown('**Recall:** share of actual failures caught. **Precision:** share of alerts that are failures. '
                    '**Specificity:** share of passes correctly cleared. **Balanced accuracy:** mean of recall and specificity. '
                    '**Average precision (AP):** failure-ranking quality across score cutoffs; higher is better. '
                    '**False alarms:** passing records sent for review. **Missed failures:** failures cleared by screening.')
