import streamlit as st
import onco
import brea
import numpy as np

st.set_page_config(page_title="Etevers Learning",layout="wide")


PAGES = {
    "Colorectal histopathology": onco,
    "Breast Cancer": brea,

}
st.sidebar.title('Pathology')
selection = st.sidebar.radio("목록", list(PAGES.keys()))



page = PAGES[selection]
page.app()