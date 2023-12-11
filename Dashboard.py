import streamlit as st
st.set_page_config(page_title="Copper Dashboard", page_icon="📈")

import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import datetime


st.markdown('<center><h1 style="color:#FC5E22">Copper Dashboard</h2></center>', unsafe_allow_html=True)

df = pd.read_csv("final_merged_copper_dataset.csv")

# Current copper price
msft = yf.Ticker("HG=F") # copper code
current_data = msft.history(period="1mo")
current_date =current_data.index[-1].strftime("%d/%m/%Y")
price = current_data.iloc[-1]['Close']
st.header(f"Current Copper Price in {current_date}")

card_style = """
    <style>
        .price-card {
            background-color: #F97129;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .price-text {
            font-size: 28px;
            font-weight: bold;
            color: white;
            padding: 0 auto;
            margin: 0 auto;
        }
    </style>
"""

# Display the styled card with the entered price
st.markdown(card_style, unsafe_allow_html=True)
st.markdown(f'<div class="price-card"><div class="price-text">Price: ${price:.4f}</div></div>', unsafe_allow_html=True)
st.text("")
st.text("")
st.text("")


# Visualizing updated copper price
st.header("Historical Copper prices ")

# Create a number slider with a default value
previous_months = st.slider("Select a number:", min_value=1.0, max_value=30.0, value=12.0, step=1.0)
previous_months_str = str(int(previous_months))+'mo'
st.write(previous_months_str)
# get historical market data
hist = msft.history(period=previous_months_str)
fig = px.line(hist, x=hist.index, y=["Close",'Low','High'], title='Copper Price')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)
st.text("")

# Getting Summary about copper prices in a specific day
st.text("")
st.header("Copper prices data summary on choosen date")
summary_Date = st.selectbox('Choose your Date',hist.index[:len(hist.index)])
summary_Date_index = hist.index.get_loc(summary_Date)
st.write('You selected:', summary_Date)
previous_date_data = hist.iloc[summary_Date_index-1]
choosen_date_data= hist.iloc[summary_Date_index]

price_difference = choosen_date_data['Close'] - previous_date_data['Close']
percentage_difference = (price_difference / previous_date_data['Close']) * 100

choosen_date_high = choosen_date_data["High"]
choosen_date_low = choosen_date_data["Low"]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Close Price", f"${choosen_date_data['Close']:.4f}")
with col2:
    st.metric("Price Difference from previous day: ", f"${price_difference:.4f}", f"{percentage_difference:+.4f}%")
with col3:
    st.metric("High", f"${choosen_date_high:.4f}")
with col4:
    st.metric("Low", f"${choosen_date_low:.4f}")
