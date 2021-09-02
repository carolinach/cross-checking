#!/usr/bin/python

####################################################################################################
## 
## Website to present the anomaly classification machine learning model. There are a number of 
## features including the following;
## 
## 1. About this website
## 2. Presentation of the raw data in graphical form
## 3. Presentation of the full test results
## 4. Selecting a random observation from the Test set created
## 5. Manually select one of the 367 Test observations
##
## It must be noted that there is no error checking in this code for easy debugging when doing code 
## changes. For production use, code checking should be added.
##
##
## Required Environment Variables
## ==============================
##
## None
##
## 
## Usage
## =====
## 
## This must be run from a python environment with streamlit capabilities. 
## 
##  example:
## 
##    streamlit run anomaly.py
##
## 
## Libraries Required
## ==================
##
##  numpy==1.19.2
##  joblib==1.0.1
##  scipy==1.7.1
##  scikit-learn==0.24.2
##  pandas==1.3.2
##  streamlit==0.87.0
##  st-annotated-text==2.0.0
##  matplotlib==3.4.3
##  pybase64==1.2.0
##
## 
## Arguments
## ==========
##
## :param None
## 
####################################################################################################
## Author: Richard Teo
## Date: 30/08/2021
## Version: 1.0.0
## Maintainer: Unknown
## Email: richardjteo+devwork@gmail.com
## Status: Development
####################################################################################################


###############################################################################################
# INITIALIZATIONS
###############################################################################################

# Import Dependencies
import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
import random

# Global declarations
root_folder = './'


###############################################################################################
# MACHINE LEARNING FUNCTIONS
###############################################################################################

# Loading the Test data for analysis
@st.cache(allow_output_mutation=True)
def load_data():
    
    # Load the x data 
    X_test = pd.read_csv('X_test.csv')

    # Load the y data 
    y_test = pd.read_csv('y_test.csv')
    
    # Load the labelers
    le_anomaly = joblib.load('le_anomaly.pkl')
    le_pump = joblib.load('le_pump.pkl')
    
    # Load the model
    model_anomaly = joblib.load('trained_model.pkl')

    return X_test, y_test, le_anomaly, le_pump, model_anomaly

# Running the Prediction Model
def prediction_anomaly(X_test_sample, y_test_sample, model_anomaly, le_anomaly):
    
    # Predict Data
    y_pred = model_anomaly.predict(X_test_sample)
    
    # Convert numbers back to labels
    prediction = le_anomaly.inverse_transform(y_pred)[0]
    reality = le_anomaly.inverse_transform(y_test_sample.to_numpy().flatten())[0]

    # Finish the progress bar
    bar.progress(100)
   
    return prediction, reality


###############################################################################################
# WEBSITE FUNCTIONS
###############################################################################################

# Running the predictions on selected data and analysing the results into a 
# user friendly web page
def data_analysis(X_test_sample, y_test_sample, model_anomaly, le_anomaly):
    
    # Fixed variables
    bar.progress(0)
#    change_global_variables(new_start_time)

    # Add a results title
    st.markdown('### Results of the Machine Learning Analysis')
    st.markdown(f"""You've selected the data;""")
    st.write(X_test_sample)

    # Stage 1 - Checking for Anomaly at this point
    progress_text.markdown('Stage 1/1 - Running Anomaly Detection ...')
    prediction, reality = prediction_anomaly(X_test_sample, y_test_sample, model_anomaly, le_anomaly)
    progress_text.markdown('Stage 1/1 - Finished Anomaly Detection')

    # Print the results of our findings on anomaly detection
    if (prediction == "normal"):
        state_text_operating_ok = "running nominally"
        state_color_operating_ok = "#afa"
    else:
        state_text_operating_ok = "running abnormally (type " + str(reality) + ")"
        state_color_operating_ok = "#faa"

    if (reality == "normal"):
        reality_text_operating_ok = "running nominally"
        reality_color_operating_ok = "#afa"
    else:
        reality_text_operating_ok = "running abnormally (type " + str(reality) + ")"
        reality_color_operating_ok = "#faa"
    annotated_text("The ML anomaly model predicts the pump is ",
                   (state_text_operating_ok, "(1)", state_color_operating_ok),
                   " where the reality during this test was ",
                   (reality_text_operating_ok, "(2)", reality_color_operating_ok))

# Processing images for the webpage
@st.cache(allow_output_mutation=True)
def img_to_bytes(img_path):
    file_ = open(img_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


###############################################################################################
# MAIN CODE
###############################################################################################

if __name__ == '__main__':

    # Fix the title
    st.set_page_config(page_title='Cross cheking')

    # Hide the default menu as this will confuse non-technical users
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    # Put in the ThermoAI logo & Copyright
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_to_bytes(root_folder + "logo.png")}" style="display: block;margin-left: auto; margin-right: auto;width:75%;" alt="logo"></br>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(f"""<strong><span style="color:#e27936">THERMO</span><span style="color:#ffffff">AI</span></strong> DEVELOPMENT</br><small>Copyright &#169; 2021 ThermoAI. All rights reserved.</small>""", unsafe_allow_html=True)
    st.sidebar.markdown("Pump Anomaly Detection Demo")

    # Side menu options
    sidebar_selection = st.sidebar.radio(
        'Select an option below:',
        ['Info about this website', 'See the raw data', 'See the full test results', 'Random Select Observation', 'I will select myself'],
    )

    # Reserve the space for the custom time selection; the actual code will be put in later
    custom_time = st.sidebar.empty()

    # Side bar description
    with st.sidebar.expander("Click to learn more about this demonstration"):
        st.markdown(f"""
        This dashboard is a demonstration of the powers of machine learning coupled with the experience of ThermoAI to produce advanced detection and anomaly classification models for pumps.

        In this demonstration, you can view the raw data and play with different observations to see how the models performs. All data used for anomaly detection and classification have not been seen by the model before. The advanced detection system views this data as clean and unknown.

        Enjoy !

        From the ThermoAI Team.

        *MMXXI*  
        """)

    # Main Page Title
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f'<h1 style="color:#e27936;">Pump Anomaly Classification</h1>',unsafe_allow_html=True)#    st.markdown('# Fan Health Monitoring')

    with t2:
        st.write("")
        st.write("")
        st.markdown(f'<strong><span style="color:#e27936">THERMO</span><span style="color:#ffffff">AI</span></strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Combution Perfected',unsafe_allow_html=True)
    st.markdown("""---""")

    # Add a placeholder for the progress bar qhen analysing data
    progress_text = st.empty()
    progress_text.markdown('Nothing happening at the moment ...')
    latest_iteration = st.empty()
    bar = st.progress(0)
    st.markdown("""---""")

    # Load data and model (only done once because cache is on)
    X_test, y_test, le_anomaly, le_pump, model_anomaly = load_data()

    # THE MAIN ACTIONS FROM THE SIDEBAR MENU

    # Lights, Camera, Action !
    if sidebar_selection == 'Info about this website':
    
        # Tell it's not a custom submit
        customsubmit = False
        
        # Write the informaton to the main page
        st.markdown(f'This is a dashboard developed by <strong><span style="color:#e27936">THERMO</span><span style="color:#4c4c4c">AI</span></strong> to show the advanced features of anomaly classification on industrial pumps.',unsafe_allow_html=True)
    elif sidebar_selection == 'See the raw data':
    
        # Tell it's not a custom submit
        customsubmit = False
    
        # Get a copy of the Test Data and put it into a line chart
        temp_data = X_test.copy()
        temp_data = temp_data.sort_values(['pump', 'capacity'], ascending = [True, True])
        temp_data = temp_data.drop(['pump', 'capacity'], axis = 1)
        st.line_chart(temp_data)
        
    elif sidebar_selection == 'See the full test results':
    
        # Tell it's not a custom submit
        customsubmit = False
    
        # Show the results graph and an explanation of what the user is looking at.
        st.markdown(f"""
        The trained machine learning model was used to predict the condition of multiple pumps in different types of conditions.  
        """)
        st.markdown(f"""
        The pumps were put under 19 different conditions - normal operation and 18 different types of problems from cavitation, valves partially or fully closed before and after the pump, etc.  
        """)
        st.markdown(
        f'<img src="data:image/png;base64,{img_to_bytes(root_folder + "training.png")}" style="display: block;margin-left: auto; margin-right: auto;width:75%;" alt="logo"></br>',
        unsafe_allow_html=True,
        )
        st.markdown(f"""
        The results speak for themselves. All the green means the predicted and real conditions were correct. Only 1 time the machine learning model predicted a wrong condition. This model give over 99% accuracy on classifying known anomalies.  
        """)
    elif sidebar_selection == 'Random Select Observation':
    
        # Tell it's not a custom submit
        customsubmit = False
    
        # Randomly select a Test Data observation and send it to the prediction model
        test_sample = random.randint(0, X_test.shape[0] - 1)
        X_test_sample = pd.DataFrame(X_test.iloc[test_sample]).T
        y_test_sample = pd.DataFrame(y_test.iloc[test_sample]).T
        data_analysis(X_test_sample, y_test_sample, model_anomaly, le_anomaly)
    elif sidebar_selection == 'I will select myself':
    
        # Tell it's not a custom submit
        form = custom_time.form("Time selection")
    
        # How a submit form to select a observation from 1 to 367
        custom_time_input = form.number_input("Select an observation from 1 to 367", min_value=1, max_value=367, step=1)
        customsubmit = form.form_submit_button("Run Analysis")

    # This is a custom input to be analysed
    if customsubmit:
    
        # Get the manually selecte Test Data observation and send it to the prediction model
        test_sample = custom_time_input
        X_test_sample = pd.DataFrame(X_test.iloc[test_sample]).T
        y_test_sample = pd.DataFrame(y_test.iloc[test_sample]).T
        data_analysis(X_test_sample, y_test_sample, model_anomaly, le_anomaly)



