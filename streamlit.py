import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Page title
st.title("Credit Card Default Prediction")
st.write("Upload data to predict the likelihood of credit card defaults.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load uploaded file
    input_data = pd.read_csv(uploaded_file)

    # Display the data
    st.write("Uploaded Data:")
    st.write(input_data.head())

    # Ensure the input data matches the model's expected features
    if st.button("Predict"):
        try:
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[:, 1]

            # Add predictions to the data
            input_data["Predicted_Default"] = predictions
            input_data["Default_Probability"] = probabilities

            # Display results
            st.write("Prediction Results:")
            st.write(input_data)

            # Download the results
            csv = input_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                csv,
                "prediction_results.csv",
                "text/csv",
                key="download-csv"
            )
        except Exception as e:
            st.error(f"Error in processing: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
