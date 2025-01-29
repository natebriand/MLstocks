import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, send_file, request
import io
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense





# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/plot", methods=['GET', 'POST'])
def plot():

    ticker = request.form['ticker']
    prediction_type = request.form['predictionType']

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=C6YROQLOVLGA2D1E'
    r = requests.get(url)
    data = r.json()

    if prediction_type == "LinearRegression":
        return linear_regression(data)
    if prediction_type == "LSTM":
        return lstm_prediction(data)
    
    

def linear_regression(data):
    """Linear Regression prediction function"""


    # time series data and prepare DataFrame
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df = df.sort_index()  # Sort by date
    df = df.astype(float)  # Convert columns to float for modeling

    user_number = request.form['number']
    user_number = int(user_number)


    # Gets a set amount of training days from the user
    new_df = df[-user_number:]

    new_df['Day'] = (new_df.index - new_df.index[0]).days
    X = new_df[['Day']]  # Feature: Days as numeric values
    y = new_df['Close']  # Target: Closing prices

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    future_days = 30  
    last_day = new_df['Day'].max()  
    future_X = np.arange(last_day + 1, last_day + 1 + future_days).reshape(-1, 1)  # Future day indices
    future_prices = model.predict(future_X)  # Predicted prices

    # Plot Historical Data and Predictions
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(new_df.index, new_df['Close'], label='Historical Prices', marker='o')

    # Plot predictions
    future_dates = [new_df.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
    plt.plot(future_dates, future_prices, label='Predicted Prices', linestyle='--', color='red')

    # Customize plot
    plt.title('Stock Price Prediction (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()


    # Save the figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()  # Close the figure to free memory
    
    # Send the image back to the client
    return send_file(img, mimetype="image/png")



def prepare_data(df, window_size=60):
    """Prepare data for LSTM model"""
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler




def lstm_prediction(data):
    """LSTM prediction function"""

    # Extract and prepare data
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={'4. close': 'Close'})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['Close'] = df['Close'].astype(float)
    
    user_number = int(request.form['number'])
    df = df[-user_number:]  # Use selected history
    
    # LSTM parameters
    window_size = 60
    future_days = 30
    
    if len(df) < window_size + future_days:
        return "Error: Not enough historical data for LSTM prediction", 400
    
    # Prepare data for LSTM
    X, y, scaler = prepare_data(df, window_size)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # build and train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.1)
    

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # create future predictions
    last_sequence = X[-1]
    future_predictions = []
    for _ in range(future_days):
        pred = model.predict(last_sequence.reshape(1, window_size, 1))
        future_predictions.append(pred[0,0])
        last_sequence = np.append(last_sequence[1:], pred[0,0])
    
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    
    # Historical data
    plt.plot(df.index[window_size:], df['Close'][window_size:], 
             label='Historical Prices', marker='o')
    
    # Test predictions
    test_dates = df.index[split+window_size:]
    plt.plot(test_dates, predictions, label='Model Predictions', linestyle='--')
    

    # Future predictions
    future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days+1)]
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
    
    plt.title('Stock Price Prediction (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)


