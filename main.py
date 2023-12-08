import os
import streamlit as st
import numpy as np
import pandas as pd
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

# Get the absolute path of the image file
image_path = 'https://drive.google.com/file/d/1kArfbxh8qDCyE77vVxRiNUzgnoI9clKh/view?usp=sharing'

# Banner image
st.image(image_path, width=1000)

# Header
st.header("Welcome to Northumbria University :smiley:") 
st.header("Advanced Practice 2023")

st.write()
col1, col2 = st.columns(2)

with col1:
    st.title("Summary")
    st.write("""Unlock the Power of Sentiment Analysis: 
             Transform your business with our cutting-edge Sentiment Analysis solution. 
             Gain valuable insights from customer sentiments, enhance decision-making, and elevate your brand's success.
             """)
    st.write(" ✔️ Machine learning based sentiment scoring")
    st.write(" ✔️ Real-time notifications")
    st.write(" ✔️ Conversation theme")
    st.write(" ✔️ Policy compliance")
    st.write(" ✔️ Trends")

with col2:
    st.title("Project Team")
    st.write("Dr. Naveed Anwar - Academic supervisor")
    st.write("Oladipupo Elesha - Project Manager")
    st.write("Indhusrijaa Gopalakrishnan - Assitant Project Manager")
    st.write("Olakunle Kuyoro - Research and Development Lead")
    st.write("Olanrewaju Atanda - Technical Lead")