import pandas as pd
import streamlit as st
from app.data_loader import threshold_table, operating_point, MODELS
from app.components import table, figure, definitions, source_links

def render():
    st.title('Threshold Tradeoffs')
    st.info('Explore historical development results. These controls only select saved rows; '
            'the final Random Forest method remains frozen at 0.35 without missingness indicators.')
    data = threshold_table()
    model = st.selectbox('Development model', ['small_forest', 'shallow_boosting', 'tree'], format_func=MODELS.get)
    indicator = st.radio('Missingness indicators', ['Without', 'With'], horizontal=True) == 'With'
    options = data[(data.model == model) & (data.indicators == indicator)].threshold.tolist()
    default = 0.35 if 0.35 in options else 0.5
    threshold = st.selectbox('Previously evaluated threshold', options, index=options.index(default),
                             format_func=lambda x: f'{x:.2f}')
    selected = operating_point(model, indicator, threshold)
    baseline = operating_point('tree', False, 0.5)
    comparison = pd.DataFrame([selected, baseline])
    comparison['model'] = [f'{MODELS[model]} (explored)', 'Shallow tree (reference)']
    table(comparison)
    st.write(f'On 1,253 development records, this option caught **{int(selected.TP)} of 83 failures**, '
             f'missed **{int(selected.FN)}**, and sent **{int(selected.flagged)}** records for review, '
             f'including **{int(selected.FP)}** passing records.')
    st.caption('Scores are not calibrated failure probabilities. A cutoff of 0.35 does not mean a verified 35% failure risk.')
    st.subheader('Why the frozen operating point was selected')
    st.write('RF 0.35 caught 17 more development failures than the shallow tree for 28 additional false alarms. '
             'Relative to RF 0.40, it caught 17 more failures for 139 additional false alarms. '
             'The higher-detection tradeoff was approved before the final test; it is not a universal optimum.')
    figure('tradeoff_figure')
    definitions()
    source_links('thresholds', 'intermediate')
