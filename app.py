import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Forecasting Tool", layout="centered")

st.title("📈 AI Forecasting Tool using Prophet")
st.write("Upload your time series CSV file with **'date'** and **'value'** columns.")

# Upload a csv file containg sales data for forecasting
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'date' not in data.columns or 'value' not in data.columns:
        st.error("CSV must contain 'date' and 'value' columns.")
    else:
        data['date'] = pd.to_datetime(data['date'])
        df = data.rename(columns={"date": "ds", "value": "y"})

       
        period = st.slider("Select number of days to forecast", 7, 14, 30)

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader("Forecast Results")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)


        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
