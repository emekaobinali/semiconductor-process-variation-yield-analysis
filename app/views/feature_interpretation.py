import streamlit as st
from app.data_loader import load
from app.components import figure, report, source_links

def render():
    st.title('Feature Interpretation')
    st.warning('Measurements are anonymous: no physical names or units are supported. '
               'Importance is predictive association, not causation. Permutation importance is an in-sample '
               'development diagnostic, not independent validation.')
    data = load('features')
    method = st.radio('Ranking method', ['Permutation importance', 'Built-in importance'], horizontal=True)
    count = st.selectbox('Features to show', [10,20])
    rank = 'permutation_rank' if method == 'Permutation importance' else 'builtin_rank'
    shown = data.sort_values(rank).head(count)
    st.dataframe(shown[['feature','builtin_rank','permutation_rank','builtin_importance',
                       'permutation_AP_loss','permutation_shuffle_SD','missing_pct']], hide_index=True, width='stretch')
    st.caption('Built-in importance is a fraction of total impurity reduction. Permutation importance is mean AP loss. '
               'Their scales differ. Shuffle SD across five repeats is not a confidence interval. Missingness is in percent.')
    feature = st.selectbox('Inspect anonymous measurement', shown.feature.tolist())
    row = data[data.feature == feature].iloc[0]
    st.subheader(f'{feature}: saved development summaries')
    st.table({'Class':['Pass','Failure'], 'Observed records':[int(row.pass_observed_n),int(row.fail_observed_n)],
              'Missing (%)':[row.pass_missing_pct,row.fail_missing_pct],
              'Q1':[row.pass_q1,row.fail_q1], 'Median':[row.pass_median,row.fail_median],
              'Q3':[row.pass_q3,row.fail_q3]})
    st.caption(f'Anonymous dataset scale; observed values only. Overall missing: {row.missing_pct:.2f}%. '
               f'Descriptive outlier flag: {row.observed_tukey_outlier_pct:.2f}%. '
               f'Shuffle rank range: {int(row.shuffle_rank_min)}–{int(row.shuffle_rank_max)}. '
               'Outliers were not removed; these flags do not prove why a feature matters.')
    with st.expander('Saved correlation cautions for interpreted features'):
        st.dataframe(load('correlations'), hide_index=True, width='stretch')
    st.write('Measurements 511, 131 and 104 rank highly under both methods. Agreement is partial: '
             '4 of the top 10 and 6 of the top 20 overlap. Measurement 060 leads built-in importance '
             'but has negative permutation AP loss. Correlation, imputation and outliers complicate interpretation.')
    figure('feature_figure')
    report('interpretation', 'Full interpretation and limitations')
    source_links('features', 'correlations')
