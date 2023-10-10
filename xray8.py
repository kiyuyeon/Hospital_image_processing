import streamlit as st
import chest_heatmap
import alz2
import breast_cancer
# import alz2
st.set_page_config(page_title="Etevers Learning",layout="wide")

PAGES = {
    "chest xray": chest_heatmap,
    "alzheimer": alz2,
    'breastcancer':breast_cancer

}

st.sidebar.title('classifer')
selection = st.sidebar.radio("목록", list(PAGES.keys()))



page = PAGES[selection]
page.app()