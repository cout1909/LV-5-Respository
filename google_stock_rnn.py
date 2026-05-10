# Google Stock Price Prediction using RNN (Recurrent Neural Network)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


# Load dataset
df = pd.read_csv("Google_Stock_Price.csv", thousands=',')

# Take Open column (opening stock price each day)
data = df['Open'].values.reshape(-1, 1)


# Clean non-numeric values and scale data to range 0-1
data = pd.to_numeric(df['Open'], errors='coerce').dropna().values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# Split into 80% train and 20% test
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]


# Create sequences of 60 timesteps
# X = last 60 days prices, y = next day price
def create_dataset(dataset):
    X = []
    y = []
    for i in range(60, len(dataset)):
        X.append(dataset[i-60:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)


# Reshape to 3D for RNN: (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Build RNN Model
model = Sequential()

# RNN Layer 1: return_sequences=True passes output to next RNN layer
model.add(SimpleRNN(50, return_sequences=True, input_shape=(60, 1)))

# RNN Layer 2: final RNN layer, return_sequences=False (default)
model.add(SimpleRNN(50))

# Output Layer: predict 1 value (next day price)
model.add(Dense(1))

# Compile: mean_squared_error for regression (predicting price)
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()


# Train model for 20 epochs
model.fit(X_train, y_train, epochs=20, batch_size=32)


# Predict on test data
predicted = model.predict(X_test)

# Inverse transform to get actual price values (undo scaling)
predicted = scaler.inverse_transform(predicted)
real = scaler.inverse_transform(y_test.reshape(-1, 1))


# Plot Real vs Predicted stock prices
plt.plot(real, color='red', label='Real Price')
plt.plot(predicted, color='blue', label='Predicted Price')

plt.title("Google Stock Price Prediction (RNN)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
