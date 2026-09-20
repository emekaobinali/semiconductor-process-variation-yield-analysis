import pandas as pd
import streamlit as st
from app.data_loader import load, MODELS
from app.components import table, definitions, source_links

def render():
    st.title('Model Comparison')
    st.caption('DEVELOPMENT VALIDATION ONLY · fixed settings · same five folds · threshold 0.50')
    st.write('Gradient Boosting ranked failures best by mean fold average precision. Ranking performance alone '
             'did not determine the final screening operating point.')
    data = load('models')
    chosen = st.multiselect('Models', list(MODELS), default=list(MODELS), format_func=MODELS.get)
    indicators = st.radio('Missingness indicators', ['Without', 'With', 'Both'], horizontal=True)
    filtered = data[data.model.isin(chosen)]
    if indicators != 'Both':
        filtered = filtered[filtered.indicators == (indicators == 'With')]
    sort = st.selectbox('Order by', ['mean_AP', 'recall', 'precision', 'FP'],
                        format_func=lambda x: {'mean_AP':'Average precision', 'recall':'Failure recall',
                                              'precision':'Precision', 'FP':'Fewest false alarms'}[x])
    filtered = filtered.sort_values(sort, ascending=sort == 'FP').copy()
    filtered['model'] = filtered.model.map(MODELS)
    table(filtered)
    st.caption('AP is the mean of five fold scores; AP SD is fold variability, not a confidence interval. '
               'Other metrics use pooled out-of-fold confusion counts. Undefined precision means no alerts.')
    with st.expander('Inspect individual validation folds'):
        folds = pd.concat([load('baseline_folds'), load('stronger_folds')], ignore_index=True)
        folds = folds[folds.model.isin(chosen)]
        if indicators != 'Both':
            folds = folds[folds.indicators == (indicators == 'With')]
        table(folds.drop(columns=['validation_records'], errors='ignore'))
    st.download_button('Download displayed aggregate comparison', filtered.to_csv(index=False),
                       file_name='development-model-comparison.csv', mime='text/csv')
    definitions()
    source_links('models', 'baseline_folds', 'stronger_folds')
