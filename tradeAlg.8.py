# main_script.py
import yfinance as yf
import requests
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import time

# Create a directory to store intermediate data
if not os.path.exists('temp_data'):
    os.makedirs('temp_data')

# Fetch historical price data with retries
def fetch_historical_data(ticker, start_date, end_date, retries=3):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            if i < retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                raise

# Fetch data in parallel
def fetch_all_data():
    tickers = ['BTC-USD', 'ETH-USD', 'BTT-USD']
    start_date = '2015-01-01'
    end_date = '2024-06-18'

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_historical_data, tickers, [start_date]*3, [end_date]*3))
    
    return results

btc_data, eth_data, btt_data = fetch_all_data()

# Save data to temp folder
btc_data.to_csv('temp_data/btc_data.csv')
eth_data.to_csv('temp_data/eth_data.csv')
btt_data.to_csv('temp_data/btt_data.csv')

# Fetch news sentiment using NewsAPI
def fetch_news_sentiment(query):
    api_key = "3ea7f60856f544559111f37671ed2dda"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json()['articles']
    sentiment_scores = [TextBlob(article['title']).sentiment.polarity for article in articles]
    return np.mean(sentiment_scores)

btc_sentiment = fetch_news_sentiment('Bitcoin')
eth_sentiment = fetch_news_sentiment('Ethereum')
btt_sentiment = fetch_news_sentiment('BitTorrent')

def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))  # Corrected line
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

btc_X, btc_y, btc_scaler = preprocess_data(btc_data['Close'])  # Adjusted input
eth_X, eth_y, eth_scaler = preprocess_data(eth_data['Close'])  # Adjusted input
btt_X, btt_y, btt_scaler = preprocess_data(btt_data['Close'])  # Adjusted input

# LSTM Model for Price Prediction
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

btc_model = build_lstm_model((btc_X.shape[1], 1))
eth_model = build_lstm_model((eth_X.shape[1], 1))
btt_model = build_lstm_model((btt_X.shape[1], 1))

btc_model.fit(btc_X, btc_y, epochs=50, batch_size=32)
eth_model.fit(eth_X, eth_y, epochs=50, batch_size=32)
btt_model.fit(btt_X, btt_y, epochs=50, batch_size=32)

# Function to predict future prices and incorporate sentiment analysis
def predict_future_prices(model, data, scaler, sequence_length, steps, sentiment):
    last_sequence = data[-sequence_length:]
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, 1)
        next_prediction = model.predict(last_sequence_scaled)
        next_prediction = next_prediction + sentiment
        next_prediction_inverse = scaler.inverse_transform(next_prediction)
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction_inverse)
        last_sequence = last_sequence.reshape(-1, 1)
    return future_predictions

# Back-testing function
def backtest(model, data, scaler, sequence_length, steps, sentiment):
    test_data = data[-(steps + sequence_length):]
    X_test, y_test, _ = preprocess_data(test_data, sequence_length)
    predictions = predict_future_prices(model, X_test[:, -1, :], scaler, sequence_length, steps, sentiment)
    actual = y_test[:steps]
    return actual, predictions

future_steps = 90  # Predicting for the next 3 months
btc_actual, btc_future_predictions = backtest(btc_model, btc_data['Close'], btc_scaler, btc_X.shape[1], future_steps, btc_sentiment)
eth_actual, eth_future_predictions = backtest(eth_model, eth_data['Close'], eth_scaler, eth_X.shape[1], future_steps, eth_sentiment)
btt_actual, btt_future_predictions = backtest(btt_model, btt_data['Close'], btt_scaler, btt_X.shape[1], future_steps, btt_sentiment)

def generate_signals(prices, predictions):
    signals = np.zeros_like(prices)
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    return signals

btc_signals = generate_signals(btc_data['Close'].values[-future_steps:], btc_future_predictions)
eth_signals = generate_signals(eth_data['Close'].values[-future_steps:], eth_future_predictions)
btt_signals = generate_signals(btt_data['Close'].values[-future_steps:], btt_future_predictions)

def plot_predictions_and_signals(dates, actual, predictions, signals, title):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label=f'{title} Actual')
    plt.plot(dates, predictions, label=f'{title} Predicted')
    buy_signals = dates[signals == 1]
    sell_signals = dates[signals == -1]
    plt.plot(buy_signals, [predictions[i] for i in range(len(signals)) if signals[i] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(sell_signals, [predictions[i] for i in range(len(signals)) if signals[i] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title(f'Predicted and Actual Prices with Buy/Sell Signals for {title}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

btc_future_dates = pd.date_range(start=btc_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
eth_future_dates = pd.date_range(start=eth_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
btt_future_dates = pd.date_range(start=btt_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')

plot_predictions_and_signals(btc_future_dates, btc_actual, btc_future_predictions, btc_signals, 'Bitcoin (BTC)')
plot_predictions_and_signals(eth_future_dates, eth_actual, eth_future_predictions, eth_signals, 'Ethereum (ETH)')
plot_predictions_and_signals(btt_future_dates, btt_actual, btt_future_predictions, btt_signals, 'BitTorrent (BTT)')

# Clean up memory
del btc_data, eth_data, btt_data, btc_X, eth_X, btt_X
gc.collect()

# Remove temp data
import shutil
shutil.rmtree('temp_data')
