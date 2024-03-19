import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gzip

# Specify the path to the compressed pickle file
compressed_file_path = '../models/best_model.pkl.gz'

# Decompress the file and load the model
with gzip.open(compressed_file_path, 'rb') as f:
    # Load the best model
    best_model = joblib.load(f)
    
scaler = joblib.load('../models/scaler.pkl')

def predict_credit_score():
    st.title("Credit Score Prediction")

    # Collect user input
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income", value=0.0)
        num_bank_accounts = st.number_input("Number of Bank Accounts", value=0)
        num_credit_cards = st.number_input("Number of Credit cards", value=0)
        interest_rate = st.number_input("Interest rate", value=0)
        num_of_loans = st.number_input("Number of Loans", value=0)
        delay_from_due_date = st.number_input("Number of days delayed from due date", value=0)
    with col2:
        num_of_delayed_payment = st.number_input("Number of delayed payments", value=0)
        credit_mix_options = ["Bad", "Standard", "Good"]
        credit_mix = st.selectbox("Credit Mix", credit_mix_options)
        outstanding_debt = st.number_input("Outstanding Debt", value=0.0)
        credit_history_age = st.number_input("Credit History Age", value=0.0)
        monthly_balance = st.number_input("Monthly Balance", value=0.0)

    # Convert Credit Mix to numerical value
    credit_mix_mapping = {"Bad": 0, "Standard": 1, "Good": 2}
    credit_mix_value = credit_mix_mapping[credit_mix]

    if st.button("Predict Credit Score"):
        # Create a feature DataFrame with named columns
        features = pd.DataFrame({
            "annual_income": [annual_income],
            "num_bank_accounts": [num_bank_accounts],
            "num_credit_cards": [num_credit_cards],
            "interest_rate": [interest_rate],
            "num_of_loans": [num_of_loans],
            "delay_from_due_date": [delay_from_due_date],
            "num_of_delayed_payment": [num_of_delayed_payment],
            "credit_mix": [credit_mix_value],
            "outstanding_debt": [outstanding_debt],
            "credit_history_age": [credit_history_age],
            "monthly_balance": [monthly_balance]
        })

        # Feature Scaling
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_score = best_model.predict(features_scaled)
        
        st.subheader("Prediction Result:")
        if predicted_score == 0:
            st.write("Predicted Credit Score: Poor")
        elif predicted_score == 1:
            st.write("Predicted Credit Score: Standard")
        elif predicted_score == 2:
            st.write("Predicted Credit Score: Good")
        else:
            st.write("Unexpected prediction result")

# Call the function to run the Streamlit app
if __name__ == "__main__":
    predict_credit_score()
