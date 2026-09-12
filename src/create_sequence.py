import pandas as pd
import numpy as np
import os

WINDOW_SIZE = 30

WITH_CONSTANT_TRAIN = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_training_scaled(not_removed_the_sensor).csv"
)

WITH_CONSTANT_VAL = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_validation_scaled(not_removed_the_sensor).csv"
)


WITHOUT_CONSTANT_TRAIN = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_training_scaled(removed_the_sensor).csv"
)

WITHOUT_CONSTANT_VAL = (
    "../CMAPSSData/Processed/"
    "train_FD001_cleaned_added_RUL_validation_scaled(removed_the_sensor).csv"
)

WITH_CONSTANT_X_TRAIN = (
    "../CMAPSSData/Processed/"
    "X_train_sequences(not_removed_the_sensor).npy"
)

WITH_CONSTANT_Y_TRAIN = (
    "../CMAPSSData/Processed/"
    "y_train_sequences(not_removed_the_sensor).npy"
)

WITH_CONSTANT_X_VAL = (
    "../CMAPSSData/Processed/"
    "X_val_sequences(not_removed_the_sensor).npy"
)

WITH_CONSTANT_Y_VAL = (
    "../CMAPSSData/Processed/"
    "y_val_sequences(not_removed_the_sensor).npy"
)

WITHOUT_CONSTANT_X_TRAIN = (
    "../CMAPSSData/Processed/"
    "X_train_sequences(removed_the_sensor).npy"
)

WITHOUT_CONSTANT_Y_TRAIN = (
    "../CMAPSSData/Processed/"
    "y_train_sequences(removed_the_sensor).npy"
)

WITHOUT_CONSTANT_X_VAL = (
    "../CMAPSSData/Processed/"
    "X_val_sequences(removed_the_sensor).npy"
)

WITHOUT_CONSTANT_Y_VAL = (
    "../CMAPSSData/Processed/"
    "y_val_sequences(removed_the_sensor).npy"
)

def create_sequences(df, window_size=30):

    X = []
    y = []

    df = df.sort_values(
        by=["unit_id", "cycle"]
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in df.columns
        if column not in ["unit_id", "cycle", "RUL"]
    ]

    print("\nFeatures used:")
    print(feature_columns)

    print("\nNumber of features:", len(feature_columns))

    for unit_id, engine_data in df.groupby("unit_id"):

        # Sort the engine's data by cycle
        engine_data = engine_data.sort_values(
            by="cycle"
        ).reset_index(drop=True)

        # Get feature values
        features = engine_data[feature_columns].values

        # Get RUL values
        rul = engine_data["RUL"].values

        # Number of cycles for this engine
        num_cycles = len(engine_data)

        if num_cycles < window_size:
            print(
                f"Engine {unit_id} skipped: "
                f"{num_cycles} cycles < {window_size} window"
            )
            continue

        for start in range(
            0,
            num_cycles - window_size + 1
        ):

            end = start + window_size

            # 30 consecutive cycles
            sequence = features[start:end]

            # Target = RUL at the LAST cycle
            target = rul[end - 1]

            X.append(sequence)
            y.append(target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def process_dataset(
    input_path,
    x_output_path,
    y_output_path,
    dataset_name
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    print("\nDataset shape:", df.shape)

    # --------------------------------------------------------
    # 2. Basic checks
    # --------------------------------------------------------

    print(
        "Number of engines:",
        df["unit_id"].nunique()
    )

    print(
        "Minimum cycle:",
        df["cycle"].min()
    )

    print(
        "Maximum cycle:",
        df["cycle"].max()
    )

    X, y = create_sequences(
        df,
        window_size=WINDOW_SIZE
    )

    print("\nSequence creation completed.")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(
        x_output_path,
        X
    )

    np.save(
        y_output_path,
        y
    )

    print("\nFiles saved:")
    print(x_output_path)
    print(y_output_path)

    
    if len(X) > 0:

        print("\nFirst sequence shape:")
        print(X[0].shape)

        print("\nFirst target:")
        print(y[0])

        print("\nFirst sequence:")
        print(X[0])

    return X, y

X_with_train, y_with_train = process_dataset(
    input_path=WITH_CONSTANT_TRAIN,
    x_output_path=WITH_CONSTANT_X_TRAIN,
    y_output_path=WITH_CONSTANT_Y_TRAIN,
    dataset_name="ALL SENSORS - TRAINING"
)

X_with_val, y_with_val = process_dataset(
    input_path=WITH_CONSTANT_VAL,
    x_output_path=WITH_CONSTANT_X_VAL,
    y_output_path=WITH_CONSTANT_Y_VAL,
    dataset_name="ALL SENSORS - VALIDATION"
)

X_without_train, y_without_train = process_dataset(
    input_path=WITHOUT_CONSTANT_TRAIN,
    x_output_path=WITHOUT_CONSTANT_X_TRAIN,
    y_output_path=WITHOUT_CONSTANT_Y_TRAIN,
    dataset_name="CONSTANT SENSORS REMOVED - TRAINING"
)


X_without_val, y_without_val = process_dataset(
    input_path=WITHOUT_CONSTANT_VAL,
    x_output_path=WITHOUT_CONSTANT_X_VAL,
    y_output_path=WITHOUT_CONSTANT_Y_VAL,
    dataset_name="CONSTANT SENSORS REMOVED - VALIDATION"
)

print("\n\n")
print("=" * 70)
print("SEQUENCE CREATION SUMMARY")
print("=" * 70)

print("\nWindow size:", WINDOW_SIZE)

print("\n---------- ALL SENSORS RETAINED ----------")

print("Training X:", X_with_train.shape)

print("Training y:", y_with_train.shape)

print("Validation X:", X_with_val.shape)

print("Validation y:", y_with_val.shape)


print("\n---------- CONSTANT SENSORS REMOVED ----------")

print("Training X:", X_without_train.shape)

print("Training y:", y_without_train.shape)

print("Validation X:", X_without_val.shape)

print("Validation y:", y_without_val.shape)


print("\n")
print("ALL SEQUENCES CREATED SUCCESSFULLY")