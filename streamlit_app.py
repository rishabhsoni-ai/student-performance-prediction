import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("artifacts/model.pkl","rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl","rb"))

# ---------- ADD HEADER SECTION HERE ----------

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction System")



st.divider()

st.subheader("Enter Student Details")

gender = st.selectbox("Gender",["male","female"])

race = st.selectbox("Race/Ethnicity",
["group A","group B","group C","group D","group E"])

education = st.selectbox("Parental Level of Education",[
"bachelor's degree",
"some college",
"master's degree",
"associate's degree",
"high school",
"some high school"
])

lunch = st.selectbox("Lunch Type",["standard","free/reduced"])

test = st.selectbox("Test Preparation Course",["none","completed"])

reading = st.slider("Reading Score",0,100,50)
writing = st.slider("Writing Score",0,100,50)


if st.button("Predict Math Score"):
    data = pd.DataFrame({
        "gender":[gender],
        "race_ethnicity":[race],
        "parental_level_of_education":[education],
        "lunch":[lunch],
        "test_preparation_course":[test],
        "reading_score":[reading],
        "writing_score":[writing]
    })

    # transform input
    data_scaled = preprocessor.transform(data)

    # predict
    result = model.predict(data_scaled)

    st.success(f"Predicted Math Score: {result[0]}")