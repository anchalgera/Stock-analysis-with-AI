
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
import math
from dateutil.parser import parse
from sklearn.metrics import mean_squared_error

import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf

def run_prediction():
    company = company_entry.get()
    entry_date_str = entry_date_entry.get()
    entry_date = parse(entry_date_str).date()
    start = dt.datetime(entry_date.year -1, entry_date.month, entry_date.day)
    end = dt.datetime(2023, 1, 1)

    # define ticker symbol
    data = yf.download(company, start, end)

    # Load Test Data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(company, test_start, test_end)

    data.tail(10)
    # Feature Engineering part
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # How many days we want to look at the past to predict
    prediction_days = 60

    # defining two empty lists for preparing the training data
    x_train = []
    y_train = []

    # we are counting from the 60th index to the last index
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Sequential provides training and inference features on this model.
    model = Sequential()
    # Add an LSTM layer with:
    #       - unit: Positive integer, the dimensionality of the output space.
    #       - return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Add a dropout layer: To prevent 'overfitting' by setting 0.2 of values to 0
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # prediction of the next closing value
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # fit the model in the training data
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    # reshaping the model
    model_input = model_input.reshape(-1, 1)
    # scaling down the model
    model_input = scaler.transform(model_input)

    x_test = []
    for x in range(prediction_days, len(model_input)):
        x_test.append(model_input[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Use the model to predict the prices
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # plot the test Predictions into the IDE
    plt.plot(actual_prices, color="black", label=f"Actual {company} price")
    plt.plot(predicted_price, color='green', label=f"Predicted {company} Price")
    plt.title(f"{company} Share price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} share price')
    plt.legend()
    plt.show()
    
    # Create the real data to compare to
    real_data = total_dataset.tail(len(test_data) + 1)
    real_data = np.array(real_data).reshape(-1, 1)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    # Create prediction for the next day
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

    # Calculate MSE and RMSE 
    mse = mean_squared_error(actual_prices, predicted_price)
    rmse = math.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot the test predictions for the ui -  The plt part is for debugging in pycharm
    figure, ax = plt.subplots(figsize=(6, 4))
    ax.plot(actual_prices, color="black", label=f"Actual {company} price")
    ax.plot(predicted_price, color='green', label=f"Predicted {company} Price")
    ax.set_title(f"{company} Share price")
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{company} share price')
    ax.legend()

    # Create a FigureCanvasTkAgg object to display the graph in the UI
    canvas = FigureCanvasTkAgg(figure, master=app)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    highest_value = np.max(prediction)
    lowest_value = np.min(prediction)

    # Display the prediction in the UI
    prediction_label.config(text=f"Our model predicts, that the highest value tomorrow is: {highest_value} and the "
                                 f"lowest: {lowest_value}")

    # Display the MSE in the UI
    mse_label = tk.Label(app, text=f"Mean Squared Error: {mse}")
    mse_label.pack()
    rmse_label = tk.Label(app, text=f"Mean Squared Error: {rmse}")
    rmse_label.pack()

# Create the main application window
app = tk.Tk()
app.title("Stock Price Prediction")

# Create labels and entry fields for the company name and entry date
company_label = tk.Label(app, text="Company Name")
company_label.pack()
company_entry = tk.Entry(app)
company_entry.pack()

entry_date_label = tk.Label(app, text="Entry Date (YYYY-MM-DD)")
entry_date_label.pack()

entry_date_entry = tk.Entry(app)
entry_date_entry.pack()

# Create a button to run the prediction
run_button = tk.Button(app, text="Run Prediction", command=run_prediction)
run_button.pack()

# Label to display the prediction
prediction_label = tk.Label(app, text="")
prediction_label.pack()

app.mainloop()