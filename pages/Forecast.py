import streamlit as st
st.set_page_config(page_title="Qara", page_icon=":rocket:")
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import streamlit as st
from datetime import date
import torch
import torch.nn as nn
from PIL import Image
from datetime import timedelta
from keras.models import load_model


def main():
  image1 = Image.open('open prediction.png')
  image2 = Image.open('close prediction.png')
  #st.image(image, caption='QARA Data Science', use_column_width=True)
  st.title("Copper AI Forecasting App")
  st.write("Forecasting of Copper Prices using LSTM Neural Network")

  col1, col2 = st.columns(2)

  with col1:
    st.image(image1, caption='Open Prediction Image', use_column_width=True)

  with col2:
    st.image(image2, caption='Close Prediction Image', use_column_width=True)
  last_Dates = np.load('last_Dates.npy', allow_pickle=True)
  last_actual_date = last_Dates[-1][0]

  forecast_end_date = st.date_input(f'End date:', value=datetime.now().date(), key='end_date')
  start_date = last_actual_date+timedelta(days=1)
    
# Calculate the difference in days
# -------------------------------------
  time_difference = forecast_end_date - start_date.date()

  # Extract the number of days from the timedelta object
  days_difference = time_difference.days

  forecast_df = pd.DataFrame(columns = ['Open','Close'], index =pd.date_range(start=start_date.date(), periods=days_difference+1, freq='B'))

# Prediction
# -------------------------------------
  model = load_model('model.h5')
  x_seq =np.load('last_sequences_rows.npy',allow_pickle=True)
  y_labels = np.load('last_label_rows.npy',allow_pickle=True)
  curr_row_seq = np.append(x_seq[-1][1:],(y_labels[-1])).reshape((1,50,2))

  if len(forecast_df)>0:
    for i in range(0,days_difference+1):
      up_pred = model.predict(curr_row_seq)
      forecast_df.iloc[i] = up_pred[0]
      curr_row_seq = np.append(curr_row_seq[0][1:],up_pred,axis=0)
      curr_row_seq = curr_row_seq.reshape(x_seq[-1:].shape)
    
    # Inverse Scale data
    MMS = joblib.load('copper_price_scaler.pkl')
    forecast_df[['Open','Close']] = MMS.inverse_transform(forecast_df[['Open','Close']])

  # Display the date input to the user.
  st.write('Start date:', start_date)
  st.write('End date:', forecast_end_date)



  # Button to make predictions
  if st.button("Generate Prediction"):
    st.write(forecast_df)  # Same as st.write(df)

if __name__ == "__main__":
  main()
