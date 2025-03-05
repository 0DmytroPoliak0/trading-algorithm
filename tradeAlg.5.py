import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tweepy
from textblob import TextBlob
import gc

# Twitter API credentials
consumer_key = 'wRSFhX1G9zg4BuTV5pMP3URFP'
consumer_secret = '4g0DUqeu3uiYbqQ3CTf3olGHxMGjtgJfFzG5jloXiDbijKOP6U'
access_token = '1730042024978722816-totdQ9cdUAn63W9wt5u5mwmp3Qc3sS'
access_token_secret = '0eTNFAUelwOEOtgToTpFYzMKk2RPfJzP3LlfnSAntH4vc'
bearer_token = "AAAAAAAAAAAAAAAAAAAAAAFTuQEAAAAAucybW87gBz5H4g5h5wJb1OKfBH0%3Dgy9uMRlfMiv54vMlp3nNZ6thvzyTjbl57ahyPZVEHOLfkatXuo"
# Authenticate to Twitter using OAuth1.0a and Bearer Token
client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret
)

# Fetch tweets using Twitter API v2
def fetch_tweets(keyword, count=10):
    query = f"{keyword} -is:retweet lang:en"
    response = client.search_recent_tweets(query=query, max_results=min(count, 10))
    if response.data:
        return [tweet.text for tweet in response.data]
    return []

# Sentiment analysis
def analyze_sentiment(tweets):
    sentiments = []
    for tweet in tweets:
        analysis = TextBlob(tweet)
        sentiments.append(analysis.sentiment.polarity)
    return np.mean(sentiments) if sentiments else 0

# Fetching historical data for Bitcoin, Ethereum, and BITI
tickers = ['BTC-USD', 'ETH-USD', 'BITI']
data = {ticker: yf.download(ticker, start='2015-01-01', end='2024-06-15', interval='1d') for ticker in tickers}

# Preprocess the data
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Preprocess each ticker
sequence_length = 60
data_processed = {ticker: preprocess_data(data[ticker], sequence_length) for ticker in tickers}

# Building the enhanced LSTM model
def build_model(input_shape):
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

# Train models for each ticker
models = {}
for ticker in tickers:
    X, y, _ = data_processed[ticker]
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=50, batch_size=32)
    models[ticker] = model

# Fetch and analyze sentiment
tweets = fetch_tweets('Bitcoin', count=10)
sentiment = analyze_sentiment(tweets)

# Function to predict future prices and incorporate sentiment
def predict_future_prices(model, data, scaler, sequence_length, steps, sentiment):
    last_sequence = data[-sequence_length:]
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_scaled = np.expand_dims(last_sequence_scaled, axis=0)
        next_prediction = model.predict(last_sequence_scaled)
        next_prediction = next_prediction + sentiment  # Adjust prediction with sentiment
        next_prediction_inverse = scaler.inverse_transform(next_prediction)
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction_inverse)
        last_sequence = last_sequence.reshape(-1, 1)
    return future_predictions

# Predict future prices for the next 12 months (365 days)
future_steps = 365
future_predictions = {ticker: predict_future_prices(models[ticker], data_processed[ticker][0], data_processed[ticker][2], sequence_length, future_steps, sentiment) for ticker in tickers}

# Define buy and sell signals based on predictions
def generate_signals(prices, predictions):
    signals = np.zeros_like(prices)
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    return signals

# Generate signals for each ticker
signals = {ticker: generate_signals(data[ticker]['Close'].values[-future_steps:], future_predictions[ticker]) for ticker in tickers}

# Plot predictions and signals
plt.figure(figsize=(14, 7))
for ticker in tickers:
    future_dates = pd.date_range(start=data[ticker].index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
    plt.plot(future_dates, future_predictions[ticker], label=f'{ticker} Predicted')
    buy_signals = future_dates[signals[ticker] == 1]
    sell_signals = future_dates[signals[ticker] == -1]
    plt.plot(buy_signals, [future_predictions[ticker][i] for i in range(len(signals[ticker])) if signals[ticker][i] == 1], '^', markersize=10, color='g', lw=0, label=f'{ticker} Buy Signal')
    plt.plot(sell_signals, [future_predictions[ticker][i] for i in range(len(signals[ticker])) if signals[ticker][i] == -1], 'v', markersize=10, color='r', lw=0, label=f'{ticker} Sell Signal')

plt.title('Predicted Prices and Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Clean up memory
del data, data_processed, models, future_predictions, signals
gc.collect()
