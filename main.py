import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, send_file, request
import io
import matplotlib





# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/plot", methods=['GET', 'POST'])
def plot():
    # Step 1: Fetch Data
    ticker = request.form['ticker']

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey=C6YROQLOVLGA2D1E'
    r = requests.get(url)
    data = r.json()

    # Extract time series data and prepare DataFrame
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


    # So we only base our prediction off the last 30 days
    new_df = df[-user_number:]

    # Step 2: Feature Engineering
    new_df['Day'] = (new_df.index - new_df.index[0]).days  # Days since the first day in the dataset
    X = new_df[['Day']]  # Feature: Days as numeric values
    y = new_df['Close']  # Target: Closing prices

    # Step 3: Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Step 4: Predict Future Prices
    future_days = 30  # Number of days to predict
    last_day = new_df['Day'].max()  # The last day in the dataset
    future_X = np.arange(last_day + 1, last_day + 1 + future_days).reshape(-1, 1)  # Future day indices
    future_prices = model.predict(future_X)  # Predicted prices

    # Step 5: Plot Historical Data and Predictions
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

if __name__ == "__main__":
    app.run(debug=True)














# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')  # Serve the HTML form

# @app.route('/process', methods=['POST'])
# def process():
#     # Get the number from the form
#     user_number = request.form['number']
#     try:
#         # Convert to integer and process the number
#         user_number = int(user_number)
#         result = user_number ** 2  # Example: Square the number
#         return f"You entered: {user_number}. Squared: {result}"
#     except ValueError:
#         return "Please enter a valid number."

# if __name__ == '__main__':
#     app.run(debug=True)



