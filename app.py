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

       
        period = st.slider("Select number of days to forecast", 7, 30, 14)

        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Analyze forecast trend
        start_value = forecast['yhat'].iloc[0]
        end_value = forecast['yhat'].iloc[-1]
        trend = end_value - start_value
        percent_change = (trend / start_value) * 100 if start_value != 0 else 0

        # Decision logic
        if percent_change > 5:
            decision = f"Forecast shows an upward trend (+{percent_change:.2f}%). Consider increasing inventory or resources."
        elif percent_change < -5:
            decision = f"Forecast shows a downward trend ({percent_change:.2f}%). Consider reducing inventory or costs."
        else:
            decision = f"Forecast is stable ({percent_change:.2f}%). No major action needed."

        st.subheader("Forecast Results")
        # Style the decision message
        if percent_change > 5:
            st.markdown(f"<div style='background-color:#d4edda;padding:10px;border-radius:5px;color:#155724;font-weight:bold;'>{decision}</div>", unsafe_allow_html=True)
        elif percent_change < -5:
            st.markdown(f"<div style='background-color:#f8d7da;padding:10px;border-radius:5px;color:#721c24;font-weight:bold;'>{decision}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#fff3cd;padding:10px;border-radius:5px;color:#856404;font-weight:bold;'>{decision}</div>", unsafe_allow_html=True)
        fig1 = m.plot(forecast)
        st.pyplot(fig1)


        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
