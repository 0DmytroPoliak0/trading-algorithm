import yfinance as yf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import gc
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure TensorFlow uses the GPU on M1 chip
tf.config.experimental.set_visible_devices([], 'GPU')

# Constants
SEQUENCE_LENGTH = 60
FUTURE_STEPS = 90
MAX_TRIALS = 20
EPOCHS = 50

# Fetch historical price data
def fetch_historical_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        logging.info(f"Fetched data for {ticker} from {start_date} to {end_date}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch news sentiment using NewsAPI and VADER sentiment analysis
def fetch_news_sentiment(query):
    try:
        api_key = "3ea7f60856f544559111f37671ed2dda"
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json()['articles']
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles]
        sentiment = np.mean(sentiment_scores)
        logging.info(f"Fetched sentiment for {query}: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error(f"Error fetching sentiment for {query}: {e}")
        return 0

# Add technical indicators to the dataset
def add_technical_indicators(data):
    try:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data = data.dropna()
        logging.info("Added technical indicators")
        return data
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return None

# Preprocess data for LSTM model
def preprocess_data(data, sequence_length=SEQUENCE_LENGTH):
    try:
        features = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'Volume_SMA_20']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[features].values)
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length, 0])  # Predicting 'Close' price
        
        X, y = np.array(X), np.array(y)
        logging.info("Preprocessed data")
        return X, y, scaler
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None, None

# LSTM Model for Price Prediction with Hyperparameter Tuning
class CryptoPriceHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(hp.Int('sequence_length', min_value=30, max_value=120, step=10), 5)))  # 5 features
        for i in range(hp.Int('num_layers', 1, 4)):
            model.add(Bidirectional(GRU(units=hp.Int(f'units_{i}', min_value=50, max_value=200, step=50), return_sequences=True if i < hp.Int('num_layers', 1, 4) - 1 else False)))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')), loss='mean_squared_error')
        return model

def perform_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    tuner = RandomSearch(
        CryptoPriceHyperModel(),
        objective='val_loss',
        max_trials=MAX_TRIALS,
        executions_per_trial=2,
        directory='hypermodel_dir',
        project_name='crypto_price_prediction'
    )
    
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    logging.info("Performed hyperparameter tuning")
    return best_model

# Function to predict future prices and incorporate sentiment analysis
def predict_future_prices(model, data, scaler, sequence_length, steps, sentiment):
    last_sequence = data[-sequence_length:]
    future_predictions = []
    for _ in range(steps):
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, 5)  # 5 features
        next_prediction = model.predict(last_sequence_scaled)
        next_prediction = next_prediction + sentiment
        next_prediction_inverse = scaler.inverse_transform(np.hstack((next_prediction, np.zeros((next_prediction.shape[0], 4)))))
        future_predictions.append(next_prediction_inverse[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction_inverse[0, :5], axis=0)
    logging.info("Predicted future prices")
    return future_predictions

# Back-testing function
def backtest(model, data, scaler, sequence_length, steps, sentiment):
    try:
        test_data = data[-(steps + sequence_length):]
        X_test, y_test, _ = preprocess_data(test_data, sequence_length)
        predictions = predict_future_prices(model, X_test[:, -1, :], scaler, sequence_length, steps, sentiment)
        actual = y_test[:steps]
        logging.info("Performed backtesting")
        return actual, predictions
    except Exception as e:
        logging.error(f"Error in backtesting: {e}")
        return [], []

def generate_signals(prices, predictions):
    signals = np.zeros_like(prices)
    signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)
    logging.info("Generated buy/sell signals")
    return signals

def plot_predictions_and_signals(dates, actual, predictions, signals, title):
    try:
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
        logging.info(f"Plotted predictions and signals for {title}")
    except Exception as e:
        logging.error(f"Error in plotting: {e}")

btc_data = fetch_historical_data('BTC-USD', '2015-01-01', '2024-06-18')
eth_data = fetch_historical_data('ETH-USD', '2015-01-01', '2024-06-18')
btt_data = fetch_historical_data('BTT-USD', '2019-01-01', '2024-06-18')

btc_sentiment = fetch_news_sentiment('Bitcoin')
eth_sentiment = fetch_news_sentiment('Ethereum')
btt_sentiment = fetch_news_sentiment('BitTorrent')

btc_data = add_technical_indicators(btc_data)
eth_data = add_technical_indicators(eth_data)
btt_data = add_technical_indicators(btt_data)

btc_X, btc_y, btc_scaler = preprocess_data(btc_data)
eth_X, eth_y, eth_scaler = preprocess_data(eth_data)
btt_X, btt_y, btt_scaler = preprocess_data(btt_data)

# Split the data for training and validation
btc_X_train, btc_X_val = btc_X[:int(0.8 * len(btc_X))], btc_X[int(0.8 * len(btc_X)):]
btc_y_train, btc_y_val = btc_y[:int(0.8 * len(btc_y))], btc_y[int(0.8 * len(btc_y)):]

# Hyperparameter tuning
best_model = perform_hyperparameter_tuning(btc_X_train, btc_y_train, btc_X_val, btc_y_val)

# Predict future prices
btc_actual, btc_future_predictions = backtest(best_model, btc_data, btc_scaler, SEQUENCE_LENGTH, FUTURE_STEPS, btc_sentiment)
eth_actual, eth_future_predictions = backtest(best_model, eth_data, eth_scaler, SEQUENCE_LENGTH, FUTURE_STEPS, eth_sentiment)
btt_actual, btt_future_predictions = backtest(best_model, btt_data, btt_scaler, SEQUENCE_LENGTH, FUTURE_STEPS, btt_sentiment)

# Generate buy/sell signals
btc_signals = generate_signals(btc_actual, btc_future_predictions)
eth_signals = generate_signals(eth_actual, eth_future_predictions)
btt_signals = generate_signals(btt_actual, btt_future_predictions)

# Plot the results
btc_future_dates = pd.date_range(start=btc_data.index[-1] + pd.Timedelta(days=1), periods=FUTURE_STEPS, freq='D')
eth_future_dates = pd.date_range(start=eth_data.index[-1] + pd.Timedelta(days=1), periods=FUTURE_STEPS, freq='D')
btt_future_dates = pd.date_range(start=btt_data.index[-1] + pd.Timedelta(days=1), periods=FUTURE_STEPS, freq='D')

plot_predictions_and_signals(btc_future_dates, btc_actual, btc_future_predictions, btc_signals, 'Bitcoin (BTC)')
plot_predictions_and_signals(eth_future_dates, eth_actual, eth_future_predictions, eth_signals, 'Ethereum (ETH)')
plot_predictions_and_signals(btt_future_dates, btt_actual, btt_future_predictions, btt_signals, 'BitTorrent (BTT)')

# Clean up memory
del btc_data, eth_data, btt_data, btc_X, eth_X, btt_X
gc.collect()
