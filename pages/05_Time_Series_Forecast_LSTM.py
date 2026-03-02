import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from src.hf_utils import hf_download, hf_load_pickle, hf_read_json

REPO_ID = "ash001/timeseries-forecast-lstm"

st.set_page_config(page_title="Time Series Forecast", page_icon="📈", layout="wide")
st.title("📈 Time Series Forecasting (PyTorch LSTM)")
st.caption("Repo: [ash001/timeseries-forecast-lstm](https://huggingface.co/ash001/timeseries-forecast-lstm)")

cfg = hf_read_json(REPO_ID, "artifacts/config.json")
window_size = int(cfg["window_size"])
hidden_size = int(cfg["hidden_size"])
horizon_default = int(cfg.get("forecast_horizon", 12))
series_id = cfg.get("series_id")

@st.cache_resource(show_spinner=False)
def load_assets():
    scaler = hf_load_pickle(REPO_ID, "artifacts/scaler.pkl")

    class LSTMnetwork(nn.Module):
        def __init__(self, input_size=1, hidden_size=100, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size)
            self.linear = nn.Linear(hidden_size, output_size)
            self.hidden = (
                torch.zeros(1, 1, hidden_size),
                torch.zeros(1, 1, hidden_size),
            )

        def reset_hidden(self):
            self.hidden = (
                torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size),
            )

        def forward(self, seq):
            lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
            pred = self.linear(lstm_out.view(len(seq), -1))
            return pred[-1]

    model = LSTMnetwork(input_size=1, hidden_size=hidden_size, output_size=1)
    state_path = hf_download(REPO_ID, "artifacts/model_state.pt")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, scaler

model, scaler = load_assets()

@st.cache_data(show_spinner=False)
def load_series_from_repo():
    csv_path = hf_download(REPO_ID, "Alcohol_Sales.csv")
    df = pd.read_csv(csv_path)
    # Try parse first column as dates
    if df.columns[0].lower() in ("date", "time", "datetime"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
        df = df.dropna(subset=[df.columns[0]]).set_index(df.columns[0]).sort_index()
    else:
        # Many FRED exports have date as first column unnamed / index
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        except Exception:
            pass

    col = series_id if (series_id and series_id in df.columns) else df.columns[0]
    s = df[col].astype(float).dropna()
    return s

series = load_series_from_repo()

st.write(f"Using series: `{series_id}`" if series_id else "Using first value column in the CSV")
st.line_chart(series.tail(120))

horizon = st.slider("Forecast horizon (steps)", 1, 36, int(horizon_default))
show_history = st.slider("Show last N points", 60, 300, 120, 10)

def forecast(series_values: np.ndarray, steps: int):
    # scale series
    scaled = scaler.transform(series_values.reshape(-1, 1)).astype(np.float32).reshape(-1)
    window = scaled[-window_size:].copy()

    preds_scaled = []
    for _ in range(steps):
        seq = torch.tensor(window, dtype=torch.float32)
        model.reset_hidden()
        with torch.no_grad():
            pred = model(seq).item()
        preds_scaled.append(pred)
        window = np.concatenate([window[1:], [pred]])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).reshape(-1)
    return preds

if st.button("Run forecast", type="primary"):
    values = series.values.astype(np.float32)
    preds = forecast(values, horizon)

    # build plot df
    hist = series.tail(show_history)
    last_idx = hist.index[-1]
    # If index is datetime-like, extend with monthly steps; else use integer steps
    if hasattr(last_idx, "to_pydatetime"):
        freq = pd.infer_freq(hist.index) or "MS"
        future_index = pd.date_range(start=hist.index[-1], periods=horizon + 1, freq=freq)[1:]
    else:
        future_index = range(len(hist), len(hist) + horizon)

    forecast_s = pd.Series(preds, index=future_index, name="forecast")
    # Add a starting forecast point at the last history timestamp.
    forecast_s = pd.concat(
        [pd.Series([hist.iloc[-1]], index=[hist.index[-1]], name="forecast"), forecast_s]
    )
    out_df = pd.DataFrame({"history": hist.values}, index=hist.index)
    out_df = out_df.join(forecast_s.to_frame(), how="outer")

    st.subheader("Forecast")
    st.line_chart(out_df)

    st.write("Forecast values:", forecast_s.round(2).to_frame())
