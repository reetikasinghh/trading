import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import plotly.graph_objs as go
import yfinance as yf
from requests.exceptions import HTTPError
import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie


from nasdaq_list import symbol_list
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.initializers import Orthogonal
from keras.optimizers import SGD
from tensorflow.python.framework import ops
import tensorflow.compat.v2 as tf


keras.initializers.Orthogonal(gain=1.0, seed=None)
ops.reset_default_graph()

# Page Layout
st.set_page_config(
    page_title="Stock Prediction System",
    layout="wide",
    page_icon="📈"
)

st.title('Stock-Ticker Predictor')



with st.sidebar:
    st.title("Stock Prediction System using Stacked-LSTM with XGBoost")
    st.subheader("Ticker-Predictor")
    st.markdown(
        """The objective is to forecast future prices based on historical data. 
        The complexity of stock market data, characterized by time dependencies and nonlinear relationships, necessitates the use of advanced machine learning techniques. 
        One effective approach involves combining Long Short-Term Memory (LSTM) networks with XGBoost, leveraging their strengths for improved predictive performance."""
    )

    st.success("Deployed", icon="💚")
    st.header("Contributor")
    st.success("Reetika Singh [Github](https://github.com/reetikasinghh)")
    


_1, _2 = st.columns([0.5, 0.5])
with _1:
    start = st.date_input("Start Date", value=None)
with _2:
    end = st.date_input("End Date", value=pd.to_datetime("today"))

# Taking input from user.
ticker_list = symbol_list
user_input = st.selectbox(
    "Enter Company Ticker",
    ticker_list,
    index=None,
    placeholder="Select Company Ticker",
    label_visibility="collapsed"
)



if user_input == None:
    st.subheader("No ticker Selected")
    col1,col2=st.columns([0.2,0.5])
    with col2:
        st_lottie("https://lottie.host/7aeb01e2-f5e7-4cdc-9849-f992f90aae13/Igkjy3ONSC.json",key="user", width=450)
else:
    df = yf.download(user_input, start, end)


def get_company_description(ticker_symbol):
    try:
        # Fetch company info from yfinance
        ticker_data = yf.Ticker(ticker_symbol)
        company_info = ticker_data.info

        # Extract the company description
        company_description = company_info.get(
            'longBusinessSummary', 'Description not available for this company.')

        return company_description

    except HTTPError as e:
        print(f"Error retrieving company description: {e}")

    except Exception as e:
        return f"Error retrieving company description: {str(e)}"


if user_input:

    # Fetch company description
    company_description = get_company_description(user_input)

    # CandleStick Plot
    st.subheader(f"Ticker: {user_input}")
    
    candlestick_chart1 = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    candlestick_chart1.update_layout(
        title=f"{user_input} Candlestick Chart", xaxis_rangeslider_visible=False, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(candlestick_chart1,
                    use_container_width=True,
                    )
    
    
    candlestick_chart = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        
    candlestick_chart.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume'))
    candlestick_chart.update_layout(
        title=f"{user_input} Candlestick Chart", xaxis_rangeslider_visible=False,xaxis_title='Date',yaxis_title='Volume')
    st.plotly_chart(candlestick_chart, 
                    use_container_width=True,
    )


    with st.expander("See Company's Description"):
        st.write(f'''{company_description}''')
        
if df.empty or df['Close'].empty:
    st.error("No stock data found for the selected ticker and date range.")
else:
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):])
    
    if data_training.empty:
        st.error("Insufficient data for training model.")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)


    # Metrics
col1, col2, col3, col4 = st.columns(4)



with col1:
    if df.empty or df['Close'].empty:
        st.metric("Max Close Price", "No data")
    else:
        max_close = df['Close'].max()
        st.metric("Max Close Price", f"${max_close:.2f}")

if df.empty or df['Close'].empty:
    st.metric("Max Close Price", "No data")
else:
    max_close = df['Close'].max()
    if pd.isna(max_close):
        st.metric("Max Close Price", "No data")
    else:
        st.metric("Max Close Price", f"${max_close:.2f}")

with col2:
    if df.empty or df['Open'].empty:
        st.metric("Max Open Price", "No data")
    else:
        max_open = df['Open'].max()
        st.metric("Max Open Price", f"${max_open:.2f}")

with col3:
    if df.empty or df['High'].empty:
        st.metric("52-Week High", "No data")
    else:
        max_high = df['High'].max()
        st.metric("52-Week High", f"${max_high:.2f}")

with col4:
    if df.empty or df['Low'].empty:
        st.metric("52-Week Low", "No data")
    else:
        max_low = df['Low'].max()
        st.metric("52-Week Low", f"${max_low:.2f}")

    
    st.subheader("Company Dataset")
    st.dataframe(df)

    @st.cache_data
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    def convert_df(df):
        return df.to_csv().encode("utf-8")
    csv = convert_df(df)

    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name="dataset.csv",
        mime="text/csv",
        use_container_width=False
    )

    if st.button("Close Feature Prediction", use_container_width=True):
        container1 = st.container(border=True)
        container1.subheader("Close Feature")
        container1.write(
            "Close feature in a stock prediction system refers to the closing price of a stock for a given trading day.")

        container1.write('''It is the final price at which the stock is traded when the market closes.
                        This is a key metric that investors and analysts use to evaluate stock performance over time.
                        In machine learning models for stock prediction, particularly time-series forecasting, the Close price is often used as the target variable because it captures the final market sentiment of a given trading day.
                        The fluctuations in this price reflect the stock's volatility, market trends, and investor behavior, making it a critical feature for predicting future stock movements.The Close price is influenced by numerous factors including market demand and supply, economic indicators, company news, earnings reports, and geopolitical events.
                        For instance, positive earnings or the launch of a successful product by a company might drive the closing price higher, while political instability or poor financial results can lead to a dip in the closing price.
                        In technical analysis, the Close price is often used in conjunction with other features like open, high, low, volume, and moving averages to identify trends and patterns that could signal buying or selling opportunities.
        ''')

        with container1.expander("Graph: Close Price v/s Time"):
            close_fig = plt.figure(figsize=(10, 6))
            ax = plt.axes()
            ax.set_facecolor('#C7E1F4')
            
            plt.plot(df.Close, '#2B24DA', label='Closing Price')
            
            plt.title(f'{user_input} Ticker plot')
            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Close Price($)')
            plt.grid(True, linestyle='--', color='#BDBDBD')
            plt.tight_layout()
            
            st.plotly_chart(
                close_fig, 
                use_container_width=False, 
                theme="streamlit"
            )

        with container1.expander("Graph: Close Price v/s Time (with mean)"):
            ma100 = df.Close.rolling(100).mean()
            close_100_fig = plt.figure(figsize=(10, 6))
            ax = plt.axes()
            ax.set_facecolor('#C7E1F4')

            plt.plot(ma100, '#43f104', label='Mean (100 val)')
            plt.plot(df.Close, '#2B24DA', label='Closing Price')
            
            plt.title(f'{user_input} Ticker plot')
            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Close Price ($)')
            plt.grid(True, linestyle='--', color='#BDBDBD')
            plt.tight_layout()
            
            st.plotly_chart(
                close_100_fig, 
                use_container_width=False, 
                theme="streamlit"
            )

        with container1.expander("Graph: Close Price v/s Time (with mean)"):
            ma100 = df.Close.rolling(100).mean()
            ma50 = df.Close.rolling(50).mean()
            ma75 = df.Close.rolling(75).mean()
            close_mean_fig = plt.figure(figsize=(10, 6))
            ax = plt.axes()
            ax.set_facecolor('#C7E1F4')

            plt.plot(ma50, '#b204e8', label='Mean (50 val)')
            plt.plot(ma75, '#f104c8', label='Mean (75 val)')
            plt.plot(ma100, '#43f104', label='Mean (100 val)')
            plt.plot(df.Close, '#2B24DA', label='Closing Price')
            
            plt.title(f'{user_input} Ticker plot')
            plt.legend()
            plt.xlabel('Year')
            plt.ylabel('Close Price ($)')
            plt.grid(True, linestyle='--', color='#BDBDBD')
            plt.tight_layout()
            
            st.plotly_chart(
                close_mean_fig, 
                use_container_width=False, 
                theme="streamlit"
            )

        if user_input:

            # Model Part
            # splitting the data into training and testing
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(
                df['Close'][int(len(df)*0.70):int(len(df))])
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Load Model
            model = tf.keras.models.load_model('8_15_23_125_LXg.h5', compile=False)

            # Testing part
            past_100_days = data_training.tail(100)
            final_df = pd.concat(
                [past_100_days, data_testing], ignore_index=True)

            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100: i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)

            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Final Graph
            container2 = st.container(border=True)
            container2.subheader('Prediction of Stock Ticker')

            with container2.expander("Model Description"):
                st.write('''R² value (Coefficient of Determination) measures the proportion of variance in the dependent variable that is predictable from the independent variables. 
                        It ranges from 0 to 1, where 1 indicates a perfect fit, meaning the model explains all the variability in the target variable, while 0 means the model does not explain any variability. 
                        A higher R² value suggests a better fit of the model to the data.
                ''')

                st.write('''MSE (Mean Squared Error) quantifies the average squared difference between the actual and predicted values. 
                        A lower MSE indicates that the model's predictions are closer to the true values, making it a critical metric for regression tasks. 
                        The square in MSE emphasizes larger errors, penalizing models that have large prediction deviations.
                ''')

                st.write('''Precision is primarily used in classification tasks and measures the proportion of true positive predictions out of all positive predictions made by the model.
                        High precision means that the model is very accurate when it predicts a positive instance, though it may miss some positives (false negatives). In essence, precision focuses on the reliability of the positive predictions made by the model.
                ''')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R2-Value", f"{0.9749:.4f}")
                with col2:
                    st.metric("MSE Value", f"{0.0011:.4f}")
                with col3:
                    st.metric("Precision", f"{0.9927:.4f}")
                
                
                model_file_path='8_15_23_125_LXg.h5'
                
                with open(model_file_path,'rb') as f:
                    model_data=f.read()
                
                st.download_button(
                    label="Download Model (.h5 format)",
                    data=model_data,
                    file_name="model.h5",
                    mime="application/octet-stream"
                )

            with container2.expander(f"Prediction Graph: {user_input}"):
                pred_fig2 = plt.figure(figsize=(11, 6))
                pred_fig2.patch.set_facecolor('#A59DDE')
                ax = plt.axes()
                ax.set_facecolor('#C7E1F4')

                plt.grid(True, linestyle='--', color='#626784')
                plt.plot(y_test, 'g', label="Original price")
                plt.plot(y_predicted, 'r', label="Predicted price")
                
                plt.xlabel('No of Days')
                plt.ylabel('Ticker Price($)')
                plt.legend()
                
                st.plotly_chart(
                    pred_fig2, 
                    use_container_width=False, 
                    theme="streamlit"
                )

        else:
            container1.write("No dataset generated")