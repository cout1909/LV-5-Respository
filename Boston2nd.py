# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Step 2: Load Dataset
df = pd.read_csv("1_boston_housing.csv")

# Remove quotes from column names (if present)
df.columns = df.columns.str.replace('"', '')

# Display first 5 rows
print(df.head())


# Step 3: Split Features and Target
X = df.drop("MEDV", axis=1)   # Features
y = df["MEDV"]                # Target

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 5: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Build Model
model = Sequential()

# Single neuron regression model
model.add(Dense(
    1,
    input_shape=(X_train.shape[1],),
    activation='linear'
))


# Step 7: Compile Model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# Step 8: Model Summary
model.summary()


# Step 9: Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


# Step 10: Evaluate Model
loss, mae = model.evaluate(X_test, y_test)

print("\nTest Loss (MSE):", loss)
print("Test MAE:", mae)


# Step 11: Predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted Values:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f} | Predicted: {y_pred[i][0]:.2f}")


# Step 12: Plot Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
