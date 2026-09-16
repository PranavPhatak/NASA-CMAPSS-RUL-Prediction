import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "../models/Reduced_Sensor_Capped/LSTM_FD001_reduced_capped_final.keras"

X_TEST_PATH = "../CMAPSSData/Processed/X_test_official.npy"
Y_TEST_PATH = "../CMAPSSData/Processed/y_test_official.npy"

RESULTS_PATH = "../CMAPSSData/Processed/test_predictions_FD001.csv"


# ============================================================
# 2. LOAD OFFICIAL TEST DATA
# ============================================================

print("Loading official test sequences...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 3. LOAD TRAINED LSTM MODEL
# ============================================================

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 4. CHECK INPUT SHAPE
# ============================================================

print("\nModel input shape:", model.input_shape)
print("Test input shape:", X_test.shape)


if model.input_shape[1:] != X_test.shape[1:]:
    raise ValueError(
        f"\nInput shape mismatch!\n"
        f"Model expects: {model.input_shape[1:]}\n"
        f"Test data has: {X_test.shape[1:]}\n"
        "\nMake sure the testing data was prepared using "
        "the same WINDOW_SIZE and features used during training."
    )


# ============================================================
# 5. MAKE RUL PREDICTIONS
# ============================================================

print("\nMaking RUL predictions...")

predicted_rul = model.predict(
    X_test,
    verbose=1
)

# Convert from shape (100, 1) to (100,)
predicted_rul = predicted_rul.flatten()


# ============================================================
# 6. CREATE RESULTS TABLE
# ============================================================

engine_ids = np.arange(1, len(y_test) + 1)

results = pd.DataFrame({
    "unit_id": engine_ids,
    "actual_RUL": y_test,
    "predicted_RUL": predicted_rul
})


# ============================================================
# 7. CALCULATE ERROR
# ============================================================

mae = mean_absolute_error(
    y_test,
    predicted_rul
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predicted_rul
    )
)

r2 = r2_score(
    y_test,
    predicted_rul
)


# ============================================================
# 8. DISPLAY METRICS
# ============================================================

print("\n========================================")
print("OFFICIAL NASA C-MAPSS FD001 TEST RESULTS")
print("========================================")

print(f"MAE  : {mae:.2f} cycles")
print(f"RMSE : {rmse:.2f} cycles")
print(f"R²   : {r2:.4f}")


# ============================================================
# 9. DISPLAY PREDICTIONS
# ============================================================

print("\n========================================")
print("RUL PREDICTIONS")
print("========================================")

print(results.to_string(index=False))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n========================================")
print("Results saved successfully:")
print(RESULTS_PATH)
print("========================================")

