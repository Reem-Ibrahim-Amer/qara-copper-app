import streamlit as st
st.set_page_config(page_title="Copper Dashboard", page_icon="📈")

import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import datetime
from datetime import datetime as dt
from PIL import Image


image = Image.open('qara_logo.png')
st.image(image, caption='QARA Data Science', use_column_width=True)
st.markdown('<center><h1 style="color:#FC5E22">Copper Futures Forecasting App</h2></center>', unsafe_allow_html=True)

# Current copper price
msft = yf.Ticker("HG=F") # copper code
current_data = msft.history(period="1mo")
current_date =current_data.index[-1].strftime("%d/%m/%Y")
price = current_data.iloc[-1]['Close']
st.header(f"Current Copper Pricing in {current_date}")

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
previous_months = st.slider("Select Month Range:", min_value=1.0, max_value=30.0, value=12.0, step=1.0)
previous_months_str = str(int(previous_months))+'mo'
st.write(previous_months_str)
# get historical market data
hist = msft.history(period=previous_months_str)
hist.index = hist.index.strftime('%Y-%m-%d')
fig = px.line(hist, x=hist.index, y=["Close",'Low','High'], title='Copper Price')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)
st.text("")

# Getting Summary about copper prices in a specific day
st.text("")
st.header("Summary")
summary_Date = st.date_input('Choose your Date', value=dt.now().date(), key='end_date')
summary_Date=pd.to_datetime(summary_Date).strftime('%Y-%m-%d')

try:
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
      st.metric("Closing Price", f"${choosen_date_data['Close']:.4f}")
  with col2:
      st.metric("Closing Delta/Day: ", f"${price_difference:.4f}", f"{percentage_difference:+.4f}%")
  with col3:
      st.metric("High", f"${choosen_date_high:.4f}")
  with col4:
      st.metric("Low", f"${choosen_date_low:.4f}")
except:
  st.write("There is no data available for the chosen day.")
