"""Launch: python -m streamlit run streamlit_app.py"""
import streamlit as st
from app.components import NOTICE
from app.views import (overview, dataset_workflow, model_comparison, threshold_tradeoffs,
                       final_test, confusion_matrix, feature_interpretation, limitations)

PAGES = {
    'Home / Overview': overview,
    'Dataset & Workflow': dataset_workflow,
    'Model Comparison': model_comparison,
    'Threshold Tradeoffs': threshold_tradeoffs,
    'Final Test Results': final_test,
    'Confusion Matrix': confusion_matrix,
    'Feature Interpretation': feature_interpretation,
    'Limitations': limitations,
}

def main():
    st.set_page_config(page_title='SECOM | Engineering portfolio', page_icon='🔬', layout='centered')
    st.sidebar.title('SECOM screening')
    st.sidebar.caption('Semiconductor process variation & yield analysis')
    page = st.sidebar.radio('Explore the project', list(PAGES))
    st.sidebar.caption('Frozen method: Random Forest · no missingness indicators · threshold 0.35')
    st.caption('EMEKA OBINALI / ENGINEERING PORTFOLIO')
    st.warning(NOTICE)
    try:
        PAGES[page].render()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'An approved result file is missing or invalid. No analysis was run. {type(exc).__name__}')
        st.stop()

if __name__ == '__main__':
    main()
