# load datasets
from operator import index
import pandas as pd
import streamlit as st
import plotly.express as px
import pandas_profiling


from pycaret.classification import setup, compare_models, pull, save_model
from streamlit_pandas_profiling import st_profile_report
import os



with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/dodata3.png")
    st.title("AutoML-Streamlit")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Machine Learning", "Download"])
    st.info("Build Automated Machine Learning Pipeline with this application")

if os.path.exists("data_source.csv"):
    df = pd.read_csv("data_source.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Data for Modelling")
    file = st.file_uploader("Upload Dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("data_source.csv", index=None)
        st.dataframe(df)

if  choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Machine Learning":
    st.title("Machine Learning Settings")
    target = st.selectbox("Select your Target", df.columns)
    
    if st.button("Train Model"):  
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("The Machine Learning Experimental Settings")
        st.dataframe(setup_df)
        
    
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the Machine Learning Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
