import streamlit as st
from app.data_loader import load
from app.components import figure, report, source_links

def render():
    st.title('Dataset & Workflow')
    data = load('inspection')
    st.markdown(f'**{data["data_rows"]:,} records · {data["measurement_columns"]} anonymous measurements · '
                f'{data["label_counts"]["1"]} failures ({data["fail_fraction"]:.1%})**')
    st.write(f'{data["missing_cell_fraction"]:.2%} of measurement cells were missing. All records had missing values. '
             'Inspection resolved the feature-count discrepancy: 590 measurement columns were actually present.')
    st.info('Feature names and physical units are unknown. Quality-test timestamps were preserved but were not predictors '
            'or the basis of the primary split; acquisition time is not confirmed.')
    figure('workflow_figure')
    st.subheader('Leakage prevention')
    st.markdown('- Fixed stratified split: **1,253 development records / 314 reserved-test records**.\n'
                '- Same five stratified development folds for all model comparisons.\n'
                '- Constant/all-missing filtering and median imputation fitted within each training fold.\n'
                '- Training-only scaling for logistic regression; no scaling for the frozen forest.\n'
                '- Missingness indicators explicitly compared; excluded from the frozen forest.\n'
                '- Row IDs and timestamps excluded as predictors; no incomplete-row deletion.\n'
                '- Final method frozen before fitting on full development data and testing once.')
    source_links('inspection', 'protocol')
    report('protocol', 'Approved validation protocol')
    report('attribution', 'SECOM source, download information, and separate CC BY 4.0 attribution')
