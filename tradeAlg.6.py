import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import requests
from textblob import TextBlob
import gc

# Function to fetch real-time Bitcoin data using CoinGecko API
def fetch_bitcoin_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',  # Currency to fetch data in USD
        'ids': 'bitcoin'       # Cryptocurrency ID for Bitcoin
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Function to perform basic sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a polarity score between -1 and 1

# Fetching historical data for Bitcoin using yfinance
ticker = 'BTC-USD'
data = yf.download(ticker, start='2015-01-01', end='2024-06-15', interval='1d')

# Preprocess the data for training the LSTM model
def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaler to normalize data between 0 and 1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # Scaling the 'Close' prices
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])  # Creating sequences of length 'sequence_length'
        y.append(scaled_data[i + sequence_length])    # Target value is the next value in the sequence
    
    X, y = np.array(X), np.array(y)  # Converting lists to numpy arrays
    return X, y, scaler

sequence_length = 60
X, y, scaler = preprocess_data(data, sequence_length)

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer with shape matching the sequences
    model.add(LSTM(units=100, return_sequences=True))  # First LSTM layer with 100 units
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(LSTM(units=100, return_sequences=True))  # Second LSTM layer with 100 units
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(LSTM(units=100))  # Third LSTM layer with 100 units
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(units=1))  # Output layer with 1 unit (the predicted value)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Compile the model
    return model

model = build_model((X.shape[1], 1))  # Building the model with the shape of the input data
model.fit(X, y, epochs=50, batch_size=32)  # Training the model

# Example sentiment text (this would typically be fetched from a sentiment source)
example_text = "Bitcoin is doing great today!"
sentiment = analyze_sentiment(example_text)

# Function to predict future prices and incorporate sentiment analysis
def predict_future_prices(model, data, scaler, sequence_length, steps, sentiment):
    last_sequence = data[-sequence_length:]  # Get the last 'sequence_length' data points
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)  # Scale the last sequence
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, 1)  # Reshape for the LSTM model
        next_prediction = model.predict(last_sequence_scaled)  # Predict the next value
        next_prediction = next_prediction + sentiment  # Adjust prediction with sentiment
        next_prediction_inverse = scaler.inverse_transform(next_prediction)  # Inverse transform to original scale
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction_inverse)  # Update the sequence with the new prediction
        last_sequence = last_sequence.reshape(-1, 1)
    return future_predictions

# Predict future prices for the next 12 months (365 days)
future_steps = 365
future_predictions = predict_future_prices(model, X[:, -1, :], scaler, sequence_length, future_steps, sentiment)

# Function to generate buy and sell signals based on predictions
def generate_signals(prices, predictions):
    signals = np.zeros_like(prices)  # Initialize signals array with zeros
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)  # Buy signal if price is predicted to increase, sell if predicted to decrease
    return signals

# Generate signals for Bitcoin
signals = generate_signals(data['Close'].values[-future_steps:], future_predictions)

# Plot predictions and buy/sell signals
plt.figure(figsize=(14, 7))
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
plt.plot(future_dates, future_predictions, label='BTC-USD Predicted')
buy_signals = future_dates[signals == 1]
sell_signals = future_dates[signals == -1]
plt.plot(buy_signals, [future_predictions[i] for i in range(len(signals)) if signals[i] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(sell_signals, [future_predictions[i] for i in range(len(signals)) if signals[i] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Predicted Prices and Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Clean up memory
del data, X, y, scaler, model, future_predictions, signals
gc.collect()
