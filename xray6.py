import streamlit as st
import ab
import tu

PAGES = {
    "Abdominal": ab,
    "brain tumor": tu,

}

st.set_page_config(page_title="Etevers Learning",layout="wide")

st.sidebar.title('Segmentation')
selection = st.sidebar.radio("목록", list(PAGES.keys()))



page = PAGES[selection]
page.app()