# main_script.py
import yfinance as yf
import requests
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import gc
import sys
import os
import openai

# Ensure the directory containing chatgpt_improver.py is in the path
sys.path.append('/Users/dmytropoliak/Downloads/')  # Replace with the actual path
from chatgpt_improver.py import get_chatgpt_suggestions

# Fetch historical price data
def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return data

btc_data = fetch_historical_data('BTC-USD', '2015-01-01', '2024-06-15')
eth_data = fetch_historical_data('ETH-USD', '2015-01-01', '2024-06-15')
btt_data = fetch_historical_data('BTT-USD', '2019-01-01', '2024-06-15')

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
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

btc_X, btc_y, btc_scaler = preprocess_data(btc_data)
eth_X, eth_y, eth_scaler = preprocess_data(eth_data)
btt_X, btt_y, btt_scaler = preprocess_data(btt_data)

# LSTM Model for Price Prediction
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

btc_model = build_lstm_model((btc_X.shape[1], 1))
eth_model = build_lstm_model((eth_X.shape[1], 1))
btt_model = build_lstm_model((btt_X.shape[1], 1))

btc_model.fit(btc_X, btc_y, epochs=100, batch_size=32)
eth_model.fit(eth_X, eth_y, epochs=100, batch_size=32)
btt_model.fit(btt_X, btt_y, epochs=100, batch_size=32)

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

future_steps = 180  # Predicting for the next 6 months
btc_future_predictions = predict_future_prices(btc_model, btc_X[:, -1, :], btc_scaler, sequence_length, future_steps, btc_sentiment)
eth_future_predictions = predict_future_prices(eth_model, eth_X[:, -1, :], eth_scaler, sequence_length, future_steps, eth_sentiment)
btt_future_predictions = predict_future_prices(btt_model, btt_X[:, -1, :], btt_scaler, sequence_length, future_steps, btt_sentiment)

def generate_signals(prices, predictions):
    signals = np.zeros_like(prices)
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    return signals

btc_signals = generate_signals(btc_data['Close'].values[-future_steps:], btc_future_predictions)
eth_signals = generate_signals(eth_data['Close'].values[-future_steps:], eth_future_predictions)
btt_signals = generate_signals(btt_data['Close'].values[-future_steps:], btt_future_predictions)

def plot_predictions_and_signals(dates, predictions, signals, title):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, predictions, label=f'{title} Predicted')
    buy_signals = dates[signals == 1]
    sell_signals = dates[signals == -1]
    plt.plot(buy_signals, [predictions[i] for i in range(len(signals)) if signals[i] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(sell_signals, [predictions[i] for i in range(len(signals)) if signals[i] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.title(f'Predicted Prices and Buy/Sell Signals for {title}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

btc_future_dates = pd.date_range(start=btc_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
eth_future_dates = pd.date_range(start=eth_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
btt_future_dates = pd.date_range(start=btt_data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')

plot_predictions_and_signals(btc_future_dates, btc_future_predictions, btc_signals, 'Bitcoin (BTC)')
plot_predictions_and_signals(eth_future_dates, eth_future_predictions, eth_signals, 'Ethereum (ETH)')
plot_predictions_and_signals(btt_future_dates, btt_future_predictions, btt_signals, 'BitTorrent (BTT)')

# Clean up memory
# Clean up memory
del btc_data, eth_data, btt_data, btc_X, eth_X, btt_X
gc.collect()