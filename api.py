import os
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from tensorflow.keras.models import load_model

MODEL_PATH = os.environ.get(
    "CMAPSS_MODEL_PATH",
    r"C:\Users\TGX-100\Documents\GitHub\NASA-CMAPSS-RUL-Prediction\Models\Reduced_Sensor_Capped\LSTM_FD001_reduced_capped_final.keras",
)

SCALER_PATH = os.environ.get(
    "CMAPSS_SCALER_PATH",
    r"C:\Users\TGX-100\Documents\GitHub\NASA-CMAPSS-RUL-Prediction\CMAPSSData\Processed\train_FD001_cleaned_added_RUL_scaler(removed_the_sensor).pkl",
)

WINDOW_SIZE = 30
RUL_CAP = 125

CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

RAW_COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c not in ["unit_id", "cycle"] + CONSTANT_SENSORS]
N_FEATURES = len(FEATURE_COLUMNS)  # 18

app = FastAPI(
    title="NASA CMAPSS RUL Prediction API",
    description="Predicts turbofan engine Remaining Useful Life (RUL) from a window of sensor cycles.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_scaler = None
_load_error = None

try:
    _model = load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    if _scaler.n_features_in_ != N_FEATURES:
        _load_error = (
            f"Scaler expects {_scaler.n_features_in_} features but the API is "
            f"configured for {N_FEATURES}. Check FEATURE_COLUMNS / CONSTANT_SENSORS."
        )
except Exception as exc:  # noqa: BLE001
    _load_error = f"Failed to load model or scaler: {exc}"

class Cycle(BaseModel):
    """One engine cycle's raw sensor reading (constant sensors may be omitted or included — they're ignored)."""

    cycle: Optional[int] = None
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    cycles: List[Cycle] = Field(..., min_length=1)

    @field_validator("cycles")
    @classmethod
    def _check_len(cls, v):
        if len(v) < 1:
            raise ValueError("At least 1 cycle is required.")
        return v


class PredictResponse(BaseModel):
    predicted_rul: float
    rul_capped_at: int = RUL_CAP
    cycles_received: int
    cycles_used: int
    was_padded: bool
    feature_order: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    window_size: int
    n_features: int
    error: Optional[str] = None

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _load_error is None else "error",
        model_loaded=_model is not None,
        scaler_loaded=_scaler is not None,
        window_size=WINDOW_SIZE,
        n_features=N_FEATURES,
        error=_load_error,
    )


@app.get("/features")
def features():
    return {
        "feature_columns": FEATURE_COLUMNS,
        "n_features": N_FEATURES,
        "window_size": WINDOW_SIZE,
        "dropped_constant_sensors": CONSTANT_SENSORS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)

    rows = np.array(
        [[getattr(c, col) for col in FEATURE_COLUMNS] for c in req.cycles],
        dtype=np.float32,
    )

    n_received = rows.shape[0]
    was_padded = False

    if n_received < WINDOW_SIZE:
        pad_rows = WINDOW_SIZE - n_received
        padding = np.repeat(rows[0:1], pad_rows, axis=0)
        window = np.vstack([padding, rows])
        was_padded = True
    else:
        window = rows[-WINDOW_SIZE:]

    try:
        scaled = _scaler.transform(window)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Scaling failed: {exc}") from exc

    X = scaled.reshape(1, WINDOW_SIZE, N_FEATURES).astype(np.float32)

    try:
        pred = _model.predict(X, verbose=0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    predicted_rul = float(np.clip(pred.flatten()[0], 0, RUL_CAP))

    return PredictResponse(
        predicted_rul=round(predicted_rul, 2),
        cycles_received=n_received,
        cycles_used=WINDOW_SIZE,
        was_padded=was_padded,
        feature_order=FEATURE_COLUMNS,
    )


@app.post("/predict_csv")
def predict_csv(rows: List[dict]):
    cycles = []
    for row in rows:
        try:
            cycles.append(Cycle(**{k: row[k] for k in FEATURE_COLUMNS}))
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"Missing required column: {exc}"
            ) from exc
    return predict(PredictRequest(cycles=cycles))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
