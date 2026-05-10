"""
============================================================
  Practical 6: Linear Regression using Deep Neural Network
  Problem: Boston Housing Price Prediction
  Dataset: Boston House Price Dataset (via sklearn / keras)
============================================================
"""

# ── 1. Import Libraries ──────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)


# ── 2. Load Dataset ──────────────────────────────────────
# Boston dataset was removed from sklearn; we load it directly via keras
(x_all, y_all), _ = keras.datasets.boston_housing.load_data(seed=42)

# Feature names (13 features)
feature_names = [
    "CRIM",    # per capita crime rate
    "ZN",      # residential land zones
    "INDUS",   # non-retail business acres
    "CHAS",    # Charles River dummy variable
    "NOX",     # nitric oxide concentration
    "RM",      # avg rooms per dwelling
    "AGE",     # proportion of old units
    "DIS",     # distance to employment centers
    "RAD",     # highway accessibility index
    "TAX",     # property tax rate
    "PTRATIO", # pupil-teacher ratio
    "B",       # 1000(Bk - 0.63)^2
    "LSTAT",   # % lower-status population
]

df = pd.DataFrame(x_all, columns=feature_names)
df["MEDV"] = y_all  # target: median value in $1000s

print("\n── Dataset Info ──")
print(f"Shape: {df.shape}")
print(df.describe().T[["mean", "std", "min", "max"]])


# ── 3. Split Data ────────────────────────────────────────
X = df[feature_names].values
y = df["MEDV"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")


# ── 4. Feature Scaling (StandardScaler) ──────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ── 5. Build the DNN Model ───────────────────────────────
"""
Architecture (Linear Regression via DNN):
  Input (13) → Dense(64, relu) → Dense(32, relu) → Dense(1, linear)

The final layer has NO activation → acts as linear regression output.
"""

def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)                      # linear output for regression
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",                          # Mean Squared Error loss
        metrics=["mae"]                      # Mean Absolute Error metric
    )
    return model

model = build_model(X_train.shape[1])
model.summary()


# ── 6. Train the Model ───────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    verbose=1
)


# ── 7. Evaluate on Test Set ──────────────────────────────
y_pred = model.predict(X_test).flatten()

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n── Test Set Metrics ──")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")


# ── 8. Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Boston Housing — DNN Linear Regression", fontsize=14, fontweight="bold")

# Plot 1: Training Loss Curve
ax = axes[0]
ax.plot(history.history["loss"],     label="Train Loss (MSE)")
ax.plot(history.history["val_loss"], label="Val Loss (MSE)")
ax.set_title("Loss (MSE) over Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.legend()
ax.grid(True)

# Plot 2: Actual vs Predicted
ax = axes[1]
ax.scatter(y_test, y_pred, alpha=0.7, color="steelblue", edgecolors="white", linewidths=0.5)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], "r--", label="Perfect Prediction")
ax.set_title("Actual vs Predicted Prices")
ax.set_xlabel("Actual Price ($1000s)")
ax.set_ylabel("Predicted Price ($1000s)")
ax.legend()
ax.grid(True)

# Plot 3: Residuals
ax = axes[2]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.7, color="coral", edgecolors="white", linewidths=0.5)
ax.axhline(0, color="black", linestyle="--")
ax.set_title("Residual Plot")
ax.set_xlabel("Predicted Price ($1000s)")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.grid(True)

plt.tight_layout()
plt.savefig("boston_housing_results.png", dpi=150)
plt.show()
print("\nPlot saved as boston_housing_results.png")
