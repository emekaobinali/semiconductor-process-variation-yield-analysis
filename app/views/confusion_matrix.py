import pandas as pd
import streamlit as st
from app.data_loader import load
from app.components import figure, source_links

def render():
    st.title('Confusion Matrix')
    st.caption('STATIC RESERVED-TEST COUNTS · frozen Random Forest at 0.35')
    data = load('test')
    st.table(pd.DataFrame({'Predicted pass':[data['TN'],data['FN']],
                          'Flagged for review':[data['FP'],data['TP']]},
                         index=['Actual pass','Actual failure']))
    st.write(f'**{data["TP"]} failures caught:** correctly directed to review. '
             f'**{data["FN"]} missed failures:** cleared despite a failed quality test.')
    st.write(f'**{data["FP"]} false alarms:** passing records that add review workload. '
             f'**{data["TN"]} correct passes:** passing records correctly cleared.')
    figure('matrix_figure')
    source_links('test')
