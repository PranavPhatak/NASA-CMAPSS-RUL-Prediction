import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. SETTINGS
# ============================================================

WINDOW_SIZE = 30

TEST_PATH = "../CMAPSSData/Raw/test_FD001.txt"
RUL_PATH = "../CMAPSSData/Raw/RUL_FD001.txt"

# This is the scaler fitted on the TRAINING data
SCALER_PATH = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_scaler(removed_the_sensor).pkl"
)

# Best trained model
MODEL_PATH = (
    "../Models/Reduced_Sensor_Capped/"
    "LSTM_FD001_reduced_capped_best.keras"
)

OUTPUT_PATH = (
    "../CMAPSSData/Processed/"
    "FD001_test_predictions.csv"
)


# ============================================================
# 2. COLUMN NAMES
# ============================================================

columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
]

columns += [f"sensor_{i}" for i in range(1, 22)]


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_df = pd.read_csv(
    TEST_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Test data shape:", test_df.shape)


# ============================================================
# 4. LOAD ACTUAL RUL VALUES
# ============================================================

print("\nLoading actual RUL values...")

actual_rul = pd.read_csv(
    RUL_PATH,
    sep=r"\s+",
    header=None
).iloc[:, 0].values


# ============================================================
# 5. BASIC CHECK
# ============================================================

number_of_engines = test_df["unit_id"].nunique()

print("Number of test engines:", number_of_engines)
print("Number of RUL values:", len(actual_rul))

if number_of_engines != len(actual_rul):
    raise ValueError(
        "Number of test engines does not match number of RUL values."
    )


# ============================================================
# 6. SORT DATA
# ============================================================

test_df = test_df.sort_values(
    ["unit_id", "cycle"]
).reset_index(drop=True)


# ============================================================
# 7. INPUT FEATURES
# ============================================================

# IMPORTANT:
# We trained this model using the dataset where sensors
# were NOT removed.
#
# Therefore:
# 3 settings + 21 sensors = 24 features

feature_columns = [
    col
    for col in test_df.columns
    if col not in ["unit_id", "cycle"]
]

print("\nNumber of input features:", len(feature_columns))

print("Features:")
print(feature_columns)


# ============================================================
# 8. LOAD TRAINING SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# 9. SCALE TEST DATA
# ============================================================

print("\nScaling test data...")

test_scaled = test_df.copy()

test_scaled[feature_columns] = scaler.transform(
    test_scaled[feature_columns]
)

print("Test data scaled successfully.")


# ============================================================
# 10. CREATE TEST SEQUENCES
# ============================================================

print("\nCreating test sequences...")

X_test = []
engine_ids = []
last_cycles = []

for unit_id, engine_data in test_scaled.groupby("unit_id"):

    engine_data = engine_data.sort_values("cycle")

    # Make sure the engine has enough cycles
    if len(engine_data) < WINDOW_SIZE:
        raise ValueError(
            f"Engine {unit_id} has only "
            f"{len(engine_data)} cycles. "
            f"At least {WINDOW_SIZE} cycles are required."
        )

    # Take the LAST 30 observed cycles
    last_30 = engine_data.tail(WINDOW_SIZE)

    # Extract the 24 input features
    sequence = last_30[feature_columns].values

    X_test.append(sequence)

    engine_ids.append(unit_id)

    # Last observed cycle
    last_cycles.append(
        engine_data["cycle"].iloc[-1]
    )


X_test = np.array(X_test)

print("X_test shape:", X_test.shape)


# ============================================================
# 11. EXPECTED SHAPE CHECK
# ============================================================

expected_features = len(feature_columns)

expected_shape = (
    number_of_engines,
    WINDOW_SIZE,
    expected_features
)

print("Expected X_test shape:", expected_shape)

if X_test.shape != expected_shape:
    raise ValueError(
        f"Unexpected X_test shape.\n"
        f"Expected: {expected_shape}\n"
        f"Got: {X_test.shape}"
    )


# ============================================================
# 12. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nModel input shape:", model.input_shape)


# ============================================================
# 13. CHECK MODEL INPUT SHAPE
# ============================================================

model_features = model.input_shape[-1]

if model_features != expected_features:

    raise ValueError(
        "\nMODEL / DATA MISMATCH!\n"
        f"Model expects {model_features} features.\n"
        f"Test data contains {expected_features} features.\n"
        "Check whether the model was trained with all sensors "
        "or with constant sensors removed."
    )


# ============================================================
# 14. PREDICT RUL
# ============================================================

print("\nPredicting RUL...")

predicted_rul = model.predict(
    X_test,
    verbose=1
).flatten()


# ============================================================
# 15. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({
    "unit_id": engine_ids,
    "last_observed_cycle": last_cycles,
    "actual_RUL": actual_rul,
    "predicted_RUL": predicted_rul
})


# ============================================================
# 16. CALCULATE ERROR
# ============================================================

results["error"] = (
    results["predicted_RUL"]
    - results["actual_RUL"]
)

results["absolute_error"] = (
    results["error"].abs()
)


# ============================================================
# 17. CALCULATE FAILURE CYCLE
# ============================================================

# Actual failure cycle:
#
# last observed cycle + actual RUL
#
# Predicted failure cycle:
#
# last observed cycle + predicted RUL

results["actual_failure_cycle"] = (
    results["last_observed_cycle"]
    + results["actual_RUL"]
)

results["predicted_failure_cycle"] = (
    results["last_observed_cycle"]
    + results["predicted_RUL"]
)


# ============================================================
# 18. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    results["actual_RUL"],
    results["predicted_RUL"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["actual_RUL"],
        results["predicted_RUL"]
    )
)

r2 = r2_score(
    results["actual_RUL"],
    results["predicted_RUL"]
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FD001 TEST RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("=" * 60)


# ============================================================
# 20. DISPLAY FIRST 20 PREDICTIONS
# ============================================================

print("\nFirst 20 predictions:\n")

print(
    results[
        [
            "unit_id",
            "last_observed_cycle",
            "actual_RUL",
            "predicted_RUL",
            "absolute_error"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)


# ============================================================
# 22. EXAMPLE FAILURE-CYCLE OUTPUT
# ============================================================

print("\nFailure cycle examples:\n")

print(
    results[
        [
            "unit_id",
            "last_observed_cycle",
            "actual_RUL",
            "predicted_RUL",
            "actual_failure_cycle",
            "predicted_failure_cycle"
        ]
    ].head(10).to_string(index=False)
)

print("\nTesting completed successfully!")
