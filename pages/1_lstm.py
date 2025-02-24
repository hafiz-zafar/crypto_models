import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

import streamlit as st

# Set page configuration
st.set_page_config(page_title="Deep Learning Models", page_icon="🤖", layout="wide")

# Custom CSS for Sidebar Styling
st.markdown("""
    <style>
        /* Sidebar Background Color */
        [data-testid="stSidebar"] {
            background-color: lightgoldenrodyellow !important;
        }

        /* Sidebar Navigation Menu Styling */
        [data-testid="stSidebarNav"] a {
            font-size: 18px !important;
            font-weight: bold;
            color: black !important;
        }

        [data-testid="stSidebarNav"] a:hover {
            color: darkblue !important;
            background-color: white !important;
            border-radius: 10px;
        }

        /* Logo Styling (Smaller Size) */
        [data-testid="stSidebarNav"]::before {
            content: "";
            display: block;
            margin: 10px auto;
            width: 80px;  /* Small Logo */
            height: 80px;
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg'); /* Replace with your logo URL */
            background-size: cover;
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)



# Main Content



# Function to fetch cryptocurrency data
def fetch_crypto_data(symbol, interval):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit=100"
        response = requests.get(url)
        data = response.json()
        prices = [float(entry[4]) for entry in data]  # Closing prices
        timestamps = [int(entry[0]) for entry in data]
        return np.array(prices), timestamps
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Function to fetch Greed and Fear Index data
def fetch_greed_and_fear_index():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url)
        data = response.json()
        index_value = data['data'][0]['value']
        index_label = data['data'][0]['value_classification']
        return index_value, index_label
    except Exception as e:
        st.error(f"Error fetching Greed and Fear Index: {e}")
        return None, None

# Load model and make predictions
def load_and_predict(model_filename, symbol, interval):
    model = tf.keras.models.load_model(model_filename)
    prices, timestamps = fetch_crypto_data(symbol, interval)
    if prices is not None:
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
        scaled_prices = scaled_prices.reshape(1, scaled_prices.shape[0], 1)
        prediction = model.predict(scaled_prices)
        predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        return float(prices[-1]), timestamps[-1], float(predicted_price), prices
    return None, None, None, None

# Function to calculate technical indicators using pandas_ta
def calculate_technical_indicators(prices):
    df = pd.DataFrame(prices, columns=["Close"])
    
    # Calculate RSI
    rsi = ta.rsi(df["Close"], length=14).iloc[-1]
    
    # Calculate MA 20
    ma_20 = ta.sma(df["Close"], length=20).iloc[-1]
    
    # Calculate MA 50
    ma_50 = ta.sma(df["Close"], length=50).iloc[-1]

    # Calculate MA 100
    ma_100 = ta.sma(df["Close"], length=100).iloc[-1]
    
    return rsi, ma_20, ma_50, ma_100

# Function to calculate MACD Buy/Sell Signals
def calculate_macd_signal(prices):
    df = pd.DataFrame(prices, columns=["Close"])
    
    # Calculate MACD line and Signal line
    df["MACD"] = ta.ema(df["Close"], length=6) - ta.ema(df["Close"], length=13)
    df["Signal"] = ta.ema(df["MACD"], length=4)
    
    # Check latest crossover for buy/sell signal
    if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] and df["MACD"].iloc[-2] <= df["Signal"].iloc[-2]:
        macd_signal = "Buy"
        macd_color = "green"
    elif df["MACD"].iloc[-1] < df["Signal"].iloc[-1] and df["MACD"].iloc[-2] >= df["Signal"].iloc[-2]:
        macd_signal = "Sell"
        macd_color = "red"
    else:
        macd_signal = "Neutral"
        macd_color = "gray"

    return macd_signal, macd_color


# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_retracements(prices):
    high = max(prices)
    low = min(prices)
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low
    }
    return levels

# Function to calculate support and resistance levels
def calculate_support_resistance(prices):
    high = max(prices)
    low = min(prices)
    close = prices[-1]
    pivot = (high + low + close) / 3
    
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {"S1": s1, "S2": s2, "S3": s3, "R1": r1, "R2": r2, "R3": r3}

# Streamlit UI
st.title("Crypto Price Prediction using LSTM RNN")

# Select cryptocurrency
crypto_options = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "TRUMP", "USUAL", "RENDER", "XLM", "SUI", "NEAR", "THETA", "IOTA"]
symbol = st.selectbox("Select Cryptocurrency", crypto_options)

# Select timeframe
interval_options = {
    "1 Minutes": "1m", 
    "5 Minutes": "5m", 
    "15 Minutes": "15m", 
    "30 Minutes": "30m", 
    "Hourly": "1h", 
    "4 Hours": "4h",  # Added 4h timeframe
    "Daily": "1d", 
    "Weekly": "1w", 
    "Monthly": "1M"
}
timeframe = st.selectbox("Select Timeframe", list(interval_options.keys()))
interval = interval_options[timeframe]

# Predict button
if st.button("Predict Price"):
    st.write(f"Fetching and predicting {symbol} prices for {timeframe} interval...")
    
    model_filename = "crypto_model.h5"
    current_price, timestamp, predicted_price, prices = load_and_predict(model_filename, symbol, interval)
    
    if timestamp is not None:
        # Calculate technical indicators
        rsi, ma_20, ma_50, ma_100 = calculate_technical_indicators(prices)
        
        # Calculate Fibonacci retracement levels
        fib_levels = calculate_fibonacci_retracements(prices)
        
        # Fetch Greed and Fear Index
        greed_fear_value, greed_fear_label = fetch_greed_and_fear_index()

        readable_timestamp_utc = datetime.utcfromtimestamp(timestamp / 1000)

        # Adjust to UAE time (UTC +4)
        uae_time = readable_timestamp_utc + timedelta(hours=4)

        # Format the UAE time to the desired format
        readable_timestamp = uae_time.strftime('%d-%m-%Y %H:%M')

        # Convert predicted_price to float in case it is not already
        if predicted_price is not None:
            predicted_price = float(predicted_price)

        # Display the predicted price comparison with the current price
        if predicted_price < current_price:
            predicted_signal = "Sell"
            predicted_color = "red"
        else:
            predicted_signal = "Buy"
            predicted_color = "green"

        # Technical indicators signals (example using EMA and Fibonacci)
        ema_signal = "Buy" if current_price > ma_50 else "Sell"  # Example based on moving averages
      # fib_signal = "Buy" if current_price > fib_levels["38.2%"] else "Sell"  # Example using Fibonacci levels

        fib_signal ="Buy" if (
                current_price > fib_levels["38.2%"] and  # Price is above 38.2% Fibonacci level
                current_price > ma_50 and                # Price is above MA50 (short-term trend bullish)
                ma_50 > ma_100 and                       # MA50 is above MA100 (confirming bullish trend)
                rsi > 50 and rsi < 70                     # RSI is between 50-70 (indicating bullish momentum)
                )else "Sell"

        # Determine Greed and Fear Index color
        if greed_fear_label == "Extreme Fear" or greed_fear_label == "Fear":
            greed_fear_color = "red"
        else:
            greed_fear_color = "green"

        support_resistance = calculate_support_resistance(prices)
        
        # Prepare data for display
        data = {
            "Indicator": ["S1", "S2", "S3", "R1", "R2", "R3"],
            "Price (USD)": [support_resistance["S1"], support_resistance["S2"], support_resistance["S3"],
                              support_resistance["R1"], support_resistance["R2"], support_resistance["R3"]]
        }
        df_support_resistance = pd.DataFrame(data)
        
        # Display results
        st.write("### Support and Resistance Levels")
        st.table(df_support_resistance)
# Calculate MACD Buy/Sell Signal
        macd_signal, macd_color = calculate_macd_signal(prices)
        # Prepare the data for display
       
        data = {
            "Indicator": ["Model Prediction", "RSI", "MA 20", "MA 50", "MA 100", "MACD Signal"] + list(fib_levels.keys()) + ["EMA Signal", "Fibonacci Signal", "Greed and Fear Index"],
            "Prediction Price (USD)": [predicted_price, rsi, ma_20, ma_50, ma_100, ""] + list(fib_levels.values()) + ["", "", greed_fear_value],
            "Signal": [f'<span style="color: {predicted_color};">{predicted_signal}</span>', "", "", "", "", f'<span style="color: {macd_color};">{macd_signal}</span>'] + [""] * len(fib_levels) + [ema_signal, fib_signal, f'<span style="color: {greed_fear_color};">{greed_fear_label}</span>'],
            "Timeframe": [timeframe] * (6 + len(fib_levels) + 3)  # Updated to match additional MACD entry
        }

        # Convert to DataFrame and display
        df = pd.DataFrame(data)

        # Highlight the predicted price based on signal color
        df.loc[df["Indicator"] == "Model Prediction", "Signal"] = f'<span style="color: {predicted_color};">{predicted_signal}</span>'

        # Display the results
        st.write(f"Latest Actual Price Timestamp: {readable_timestamp}")
        st.write(f"Current Price: ${current_price:.4f} USD")
        st.write(f"Predicted Next Price: ${predicted_price:.4f} USD")
        st.write("Comparison of Price Predictions, Technical Indicators, Fibonacci Levels, and Greed/Fear Index:")
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.error("Failed to fetch data or make predictions. Please try again later.")
