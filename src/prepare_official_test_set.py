"""
Prepares the OFFICIAL NASA CMAPSS FD001 test set for evaluation.

Why this file exists: `test_FD001.txt` + `RUL_FD001.txt` are NASA's actual
held-out test engines (their true RUL is only revealed via RUL_FD001.txt,
never derivable from the sensor file itself, because each test engine's
sensor trace is truncated at some arbitrary point before failure). This is
the standard benchmark used in the CMAPSS literature. The 20-engine
"validation" split used elsewhere in this pipeline is drawn from the same
100 engines as training, which is fine for tuning, but it is NOT this
benchmark -- so a model can look good on that split and still be evaluated
too optimistically. Run this once, then use its output as your real,
apples-to-literature accuracy number.

RUL protocol for the test set (standard for FD001):
  - Take only the LAST `WINDOW_SIZE` cycles of each test engine's trace
    (that's the most recent information available at truncation, i.e.
    "right now, mid-flight").
  - If an engine has fewer than WINDOW_SIZE cycles recorded, pad by
    repeating its first row backward (never fabricate future data).
  - Compare the model's prediction at that final cycle against the true
    RUL from RUL_FD001.txt, capped at the same RUL_CAP used in training
    (125) so the comparison is apples-to-apples with what the model was
    trained to predict.

Run this AFTER calculate_rul.ipynb / 03_Feature_Scaling.ipynb have
produced the fitted scaler, and after create_sequence_corrected.py so
TREND_WINDOW here matches. Uses the SAVED scaler with .transform() only
-- never re-fit on test data, or the result stops being a genuine
held-out check.
"""

import joblib
import numpy as np
import pandas as pd

WINDOW_SIZE = 30
TREND_WINDOW = 5
RUL_CAP = 125

# Same 6 sensors dropped in 01_data_cleaning.ipynb
CONSTANT_SENSORS = [
    "sensor_1", "sensor_5", "sensor_16",
    "sensor_10", "sensor_19", "sensor_18",
]

COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

TEST_RAW_PATH = "../CMAPSSData/Raw/test_FD001.txt"
RUL_TRUE_PATH = "../CMAPSSData/Raw/RUL_FD001.txt"

# The scaler fit on TRAINING data only in 03_Feature_Scaling.ipynb --
# reused here with .transform() only.
SCALER_PATH = "../CMAPSSData/Processed/train_FD001_cleaned_added_RUL_scaler(removed_the_sensor).pkl"

X_TEST_OUT = "../CMAPSSData/Processed/X_test_official.npy"
Y_TEST_OUT = "../CMAPSSData/Processed/y_test_official.npy"


def add_trend_features(engine_data, feature_columns, trend_window=TREND_WINDOW):
    base = engine_data[feature_columns]
    roll_mean = base.rolling(window=trend_window, min_periods=1).mean()
    roll_mean.columns = [f"{c}_rollmean{trend_window}" for c in feature_columns]
    roc = base.diff(trend_window).fillna(0.0)
    roc.columns = [f"{c}_roc{trend_window}" for c in feature_columns]
    return pd.concat(
        [engine_data.reset_index(drop=True),
         roll_mean.reset_index(drop=True),
         roc.reset_index(drop=True)],
        axis=1,
    )


def build_official_test_sequences():
    print("Loading raw test set:", TEST_RAW_PATH)
    df = pd.read_csv(TEST_RAW_PATH, names=COLUMNS, sep=r"\s+", header=None)
    print("Raw test shape:", df.shape)
    print("Test engines:", df["unit_id"].nunique())

    df = df.drop(columns=CONSTANT_SENSORS)

    true_rul = pd.read_csv(RUL_TRUE_PATH, header=None, names=["RUL"])
    assert len(true_rul) == df["unit_id"].nunique(), (
        "RUL_FD001.txt row count must match number of test engines -- "
        f"got {len(true_rul)} RUL values for {df['unit_id'].nunique()} engines."
    )

    scaler = joblib.load(SCALER_PATH)
    base_feature_columns = [
        c for c in df.columns if c not in ["unit_id", "cycle"]
    ]
    df[base_feature_columns] = scaler.transform(df[base_feature_columns])

    X_test = []
    y_test = []
    skipped_padded = 0

    for idx, (unit_id, engine_data) in enumerate(df.groupby("unit_id")):
        engine_data = engine_data.sort_values(by="cycle").reset_index(drop=True)

        engine_data = add_trend_features(engine_data, base_feature_columns, TREND_WINDOW)

        engineered_columns = [
            c for c in engine_data.columns if c not in ["unit_id", "cycle"]
        ]

        features = engine_data[engineered_columns].values
        num_cycles = len(features)

        if num_cycles < WINDOW_SIZE:
            # Pad backward by repeating the first row -- never use future
            # data the model wouldn't actually have "mid-flight".
            pad_rows = WINDOW_SIZE - num_cycles
            padding = np.repeat(features[0:1], pad_rows, axis=0)
            window = np.vstack([padding, features])
            skipped_padded += 1
        else:
            window = features[-WINDOW_SIZE:]

        # True RUL for this engine, same cap applied in training
        target = min(true_rul.iloc[idx]["RUL"], RUL_CAP)

        X_test.append(window)
        y_test.append(target)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(f"\nEngines padded (fewer than {WINDOW_SIZE} cycles):", skipped_padded)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    np.save(X_TEST_OUT, X_test)
    np.save(Y_TEST_OUT, y_test)
    print("\nSaved:", X_TEST_OUT)
    print("Saved:", Y_TEST_OUT)

    return X_test, y_test


if __name__ == "__main__":
    build_official_test_sequences()
