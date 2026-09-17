"""
Prepares the OFFICIAL NASA CMAPSS FD001 test set for evaluation.

This file:

1. Loads NASA's official test_FD001.txt
2. Removes the same 6 constant sensors removed during training
3. Loads the scaler fitted ONLY on training data
4. Applies scaler.transform() to the test data
5. Takes the LAST WINDOW_SIZE cycles from each test engine
6. Pads engines with fewer than WINDOW_SIZE cycles
7. Loads the true RUL values from RUL_FD001.txt
8. Applies the same RUL cap used during training (125)
9. Saves X_test and y_test as .npy files

IMPORTANT:
The trained LSTM expects 18 features per cycle.

Therefore, this file DOES NOT create trend features.

Expected output:

X_test shape: (100, 30, 18)
y_test shape: (100,)
"""

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

WINDOW_SIZE = 50
RUL_CAP = 130


# ============================================================
# 2. CONSTANT SENSORS
# ============================================================
# These are the same 6 sensors removed during training.

CONSTANT_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_16",
    "sensor_10",
    "sensor_19",
    "sensor_18",
]


# ============================================================
# 3. COLUMN NAMES
# ============================================================

COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# 4. FILE PATHS
# ============================================================

TEST_RAW_PATH = "../CMAPSSData/Raw/test_FD001.txt"

RUL_TRUE_PATH = "../CMAPSSData/Raw/RUL_FD001.txt"

# Scaler fitted on TRAINING data only
SCALER_PATH = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_"
    "scaler(removed_the_sensor).pkl"
)

X_TEST_OUT = (
    "../CMAPSSData/Processed/"
    "X_test_official.npy"
)

Y_TEST_OUT = (
    "../CMAPSSData/Processed/"
    "y_test_official.npy"
)


# ============================================================
# 5. BUILD OFFICIAL TEST SEQUENCES
# ============================================================

def build_official_test_sequences():

    # --------------------------------------------------------
    # Load raw test data
    # --------------------------------------------------------

    print("Loading raw test set:")
    print(TEST_RAW_PATH)

    df = pd.read_csv(
        TEST_RAW_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    print("\nRaw test shape:", df.shape)
    print("Test engines:", df["unit_id"].nunique())


    # --------------------------------------------------------
    # Remove constant sensors
    # --------------------------------------------------------

    print("\nRemoving constant sensors:")

    for sensor in CONSTANT_SENSORS:
        print("  -", sensor)

    df = df.drop(columns=CONSTANT_SENSORS)


    # --------------------------------------------------------
    # Load true RUL values
    # --------------------------------------------------------

    true_rul = pd.read_csv(
        RUL_TRUE_PATH,
        header=None,
        names=["RUL"]
    )

    print("\nTrue RUL values:", len(true_rul))


    # --------------------------------------------------------
    # Check number of engines and RUL values
    # --------------------------------------------------------

    number_of_engines = df["unit_id"].nunique()

    assert len(true_rul) == number_of_engines, (
        "RUL_FD001.txt row count must match the number "
        "of test engines. "
        f"Got {len(true_rul)} RUL values for "
        f"{number_of_engines} engines."
    )


    # --------------------------------------------------------
    # Identify model features
    # --------------------------------------------------------
    # Do NOT include:
    #   unit_id
    #   cycle
    #
    # After removing 6 constant sensors:
    #
    # 3 settings + 15 sensors = 18 features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column not in ["unit_id", "cycle"]
    ]

    print("\nFeatures used by the model:", len(feature_columns))
    print(feature_columns)


    # --------------------------------------------------------
    # Load scaler fitted on training data
    # --------------------------------------------------------

    print("\nLoading training scaler:")
    print(SCALER_PATH)

    scaler = joblib.load(SCALER_PATH)


    # --------------------------------------------------------
    # Check scaler feature count
    # --------------------------------------------------------

    if len(feature_columns) != scaler.n_features_in_:
        raise ValueError(
            "\nScaler feature mismatch!\n"
            f"Test data has {len(feature_columns)} features.\n"
            f"Scaler expects {scaler.n_features_in_} features.\n"
            "\nMake sure the same sensors were removed "
            "during training and testing."
        )


    # --------------------------------------------------------
    # Scale test data
    # --------------------------------------------------------
    # IMPORTANT:
    # Use transform() only.
    #
    # Never use fit_transform() on test data.
    # --------------------------------------------------------

    df[feature_columns] = scaler.transform(
        df[feature_columns]
    )


    # --------------------------------------------------------
    # Create arrays
    # --------------------------------------------------------

    X_test = []
    y_test = []

    skipped_padded = 0


    # --------------------------------------------------------
    # Process every engine separately
    # --------------------------------------------------------

    for idx, (unit_id, engine_data) in enumerate(
        df.groupby("unit_id")
    ):

        # Sort cycles in chronological order
        engine_data = (
            engine_data
            .sort_values(by="cycle")
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # Extract the 18 model features
        # ----------------------------------------------------

        features = engine_data[feature_columns].values


        # Number of cycles available for this engine
        num_cycles = len(features)


        # ----------------------------------------------------
        # Take last WINDOW_SIZE cycles
        # ----------------------------------------------------

        if num_cycles < WINDOW_SIZE:

            # If fewer than 30 cycles are available,
            # repeat the first row backward.

            pad_rows = WINDOW_SIZE - num_cycles

            padding = np.repeat(
                features[0:1],
                pad_rows,
                axis=0
            )

            window = np.vstack([
                padding,
                features
            ])

            skipped_padded += 1

        else:

            # Take only the most recent 30 cycles

            window = features[-WINDOW_SIZE:]


        # ----------------------------------------------------
        # Get true RUL
        # ----------------------------------------------------
        # RUL_FD001.txt follows the engine order:
        #
        # Engine 1 -> first RUL
        # Engine 2 -> second RUL
        # ...
        #
        # Apply same RUL cap used during training.
        # ----------------------------------------------------

        target = min(
            true_rul.iloc[idx]["RUL"],
            RUL_CAP
        )


        # ----------------------------------------------------
        # Store sequence and target
        # ----------------------------------------------------

        X_test.append(window)
        y_test.append(target)


    # ========================================================
    # 6. CONVERT TO NUMPY ARRAYS
    # ========================================================

    X_test = np.array(
        X_test,
        dtype=np.float32
    )

    y_test = np.array(
        y_test,
        dtype=np.float32
    )


    # ========================================================
    # 7. CHECK FINAL SHAPES
    # ========================================================

    print("\n========================================")
    print("OFFICIAL TEST DATA")
    print("========================================")

    print(
        f"Engines padded "
        f"(fewer than {WINDOW_SIZE} cycles):",
        skipped_padded
    )

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


    # Expected:
    #
    # X_test = (100, 30, 18)
    # y_test = (100,)


    expected_shape = (
        number_of_engines,
        WINDOW_SIZE,
        len(feature_columns)
    )

    if X_test.shape != expected_shape:

        raise ValueError(
            "\nUnexpected X_test shape!\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {X_test.shape}"
        )


    # ========================================================
    # 8. SAVE TEST DATA
    # ========================================================

    np.save(
        X_TEST_OUT,
        X_test
    )

    np.save(
        Y_TEST_OUT,
        y_test
    )


    print("\n========================================")
    print("FILES SAVED")
    print("========================================")

    print("X_test:", X_TEST_OUT)
    print("y_test:", Y_TEST_OUT)


    return X_test, y_test


# ============================================================
# 9. RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    build_official_test_sequences()
