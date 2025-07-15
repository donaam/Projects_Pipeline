import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Transaction Prediction")
st.write("Enter transaction details to predict fraudulent or legitimate.")

with st.form("prediction_form"):
    cc_num = st.number_input("Credit Card Number", value=0.0)
    amt = st.number_input("Amount", value=0.0)
    zip_code = st.number_input("ZIP Code", value=0)
    lat = st.number_input("Latitude", value=0.0)
    long = st.number_input("Longitude", value=0.0)
    city_pop = st.number_input("City Population", value=0)
    unix_time = st.number_input("Unix Time", value=0)
    merch_lat = st.number_input("Merchant Latitude", value=0.0)
    merch_long = st.number_input("Merchant Longitude", value=0.0)
    user_tx_count = st.number_input("User Transaction Count", value=0)
    merch_tx_count = st.number_input("Merchant Transaction Count", value=0)

    merchant = st.text_input("Merchant")
    category = st.text_input("Category")
    first = st.text_input("First Name")
    last = st.text_input("Last Name")
    gender = st.selectbox("Gender", ["M", "F"])
    street = st.text_input("Street")
    city = st.text_input("City")
    state = st.text_input("State")
    job = st.text_input("Job")
    dob = st.text_input("Date of Birth (YYYY-MM-DD)")
    date = st.text_input("Transaction Date (YYYY-MM-DD)")
    time = st.text_input("Transaction Time (HH:MM:SS)")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        data = CustomData(
            cc_num=cc_num,
            amt=amt,
            zip=zip_code,
            lat=lat,
            long=long,
            city_pop=city_pop,
            unix_time=unix_time,
            merch_lat=merch_lat,
            merch_long=merch_long,
            user_transaction_count=user_tx_count,
            merchant_transaction_count=merch_tx_count,
            merchant=merchant,
            category=category,
            first=first,
            last=last,
            gender=gender,
            street=street,
            city=city,
            state=state,
            job=job,
            dob=dob,
            date=date,
            time=time
        )

        input_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(input_df)

        if pred[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
