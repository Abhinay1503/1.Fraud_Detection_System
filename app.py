import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("FRAUD DETECTION SYSTEM")
st.markdown("pleasse enter the transaction details and use the predict button")
st.divider()
transaction_type = st.selectbox("transaction type",["PAYMENT","TRANSFER","CASH_OUT","DEPOSIT"] )
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceorg = st.number_input("Old Balance (sender)", min_value=0.0, value=10000.0 )
newbalanceOrig = st.number_input("New Balance (sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Reciever)", min_value=0.0, value= 0.0)
newbalanceDest = st.number_input("New Balance (Reciever)", min_value=0.0, value=0.0)

if st.button("predict"):
    input_data = pd.DataFrame([{
        "type" : transaction_type,
        "amount" : amount,
        "oldbalanceOrg" : oldbalanceorg,
        "newbalanceOrig" : newbalanceOrig,
        "oldbalanceDest" : oldbalanceDest,
        "newbalanceDest" : newbalanceDest,
    }])

    prediction = model.predict(input_data)[0]
    st.subheader(f"prediction : '{int(prediction)}'")
    if prediction == 1:
        st.error("This Trancsaction can be fraud")
    else:
        st.success("This transaction look like it is not a fraud")
