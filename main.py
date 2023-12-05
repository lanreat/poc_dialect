import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv

# Load environment variables at the entry point
load_dotenv('.env')

# Set the background color of the main page
st.set_page_config(
    page_title="Dialect",
    page_icon=":smiley:",
    layout="wide",  # "wide" or "centered"
    initial_sidebar_state="expanded",  # "auto", "expanded", "collapsed"
)    

# Custom styles for the sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #454545;
        
    }
</style>
""", unsafe_allow_html=True)

#Banner image
st.image('./home.png', width=1000)
# Header
st.header("Welcome to Northumbria University :clap:") 
st.header("Advanced Practice 2023")

st.write()
col1, col2, col3 = st.columns(3)

with col1:
    st.write("This is the left column placeholder.")
    st.write("We may redesign this page altogether.")

with col2:
     st.write("This is the middle column placeholder....")
     st.write("...or add content to this.")

with col3:
    st.write("This is the right column placeholder.")