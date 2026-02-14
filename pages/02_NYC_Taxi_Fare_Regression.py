import math
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from src.hf_utils import hf_download, hf_read_json

REPO_ID = "ash001/nyc-taxi-fare-regression-ann"

st.set_page_config(page_title="NYC Taxi Fare Regression", page_icon="🚕", layout="wide")
st.title("🚕 NYC Taxi Fare Regression (PyTorch ANN)")
st.caption("Repo: ash001/nyc-taxi-fare-regression-ann (artifacts/)")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


schema = hf_read_json(REPO_ID, "artifacts/schema.json")
target_col = schema.get("target_col", "fare_amount")
cat_cols = schema["cat_cols"]          # ["Hour","AMorPM","Weekday"]
cont_cols = schema["cont_cols"]        # lat/long, passenger_count, dist_km
cat_categories = schema["cat_categories"]

metrics = None
try:
    metrics = hf_read_json(REPO_ID, "artifacts/metrics.json")
except Exception:
    pass

class TabularRegressor(nn.Module):
    def __init__(self, emb_sizes, n_cont, hidden_layers=(200, 100), p=0.4):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(int(ni), int(nf)) for ni, nf in emb_sizes])
        self.emb_drop = nn.Dropout(float(p))

        n_emb = sum(int(nf) for _, nf in emb_sizes)
        n_in = n_emb + int(n_cont)

        layers = []
        for h in hidden_layers:
            h = int(h)
            layers += [
                nn.Linear(n_in, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(float(p)),
            ]
            n_in = h
        layers += [nn.Linear(n_in, 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embs, dim=1)
        x = self.emb_drop(x)
        x = torch.cat([x, x_cont], dim=1)
        return self.layers(x)


@st.cache_resource(show_spinner=False)
def load_model_and_schema():
    model_path = hf_download(REPO_ID, "artifacts/model_state.pt")
    emb_sizes = schema["emb_sizes"]
    hidden_layers = tuple(schema.get("hidden_layers", [200, 100]))
    dropout = float(schema.get("dropout", 0.4))

    model = TabularRegressor(
        emb_sizes=emb_sizes,
        n_cont=len(cont_cols),
        hidden_layers=hidden_layers,
        p=dropout,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    cont_mean = np.array(schema.get("cont_mean", [0.0] * len(cont_cols)), dtype=np.float32)
    cont_std = np.array(schema.get("cont_std", [1.0] * len(cont_cols)), dtype=np.float32)
    standardize = bool(schema.get("standardize_cont", True))
    return model, cont_mean, cont_std, standardize


@st.cache_data(show_spinner=False)
def load_sample_rows():
    p = hf_download(REPO_ID, "artifacts/sample_rows.csv")
    return pd.read_csv(p)


model, cont_mean, cont_std, standardize = load_model_and_schema()


# ---------------- UI state defaults ----------------
HOURS = [int(x) for x in cat_categories["Hour"]]
AMPM = [str(x) for x in cat_categories["AMorPM"]]
WDAYS = [str(x) for x in cat_categories["Weekday"]]

DEFAULTS = {
    "hour": int(HOURS[0]) if HOURS else 0,
    "ampm": AMPM[0] if AMPM else "am",
    "weekday": WDAYS[0] if WDAYS else "Mon",
    # Rough NYC defaults
    "pickup_lat": 40.7580,
    "pickup_lon": -73.9855,
    "dropoff_lat": 40.7128,
    "dropoff_lon": -74.0060,
    "passengers": 1,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------- Sample row loader ----------------
with st.expander("Load a row from dataset sample (optional)", expanded=False):
    df = load_sample_rows()
    st.caption(f"Sample rows: **{len(df)}** (valid row index: 0 to {len(df)-1})")
    idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=0, step=1)
    row_df = df.iloc[[int(idx)]]
    st.write(row_df)
    row = row_df.iloc[0]

    if st.button("Load this row into the inputs"):
        st.session_state["hour"] = int(row["Hour"])
        st.session_state["ampm"] = str(row["AMorPM"])
        st.session_state["weekday"] = str(row["Weekday"])
        st.session_state["pickup_lat"] = float(row["pickup_latitude"])
        st.session_state["pickup_lon"] = float(row["pickup_longitude"])
        st.session_state["dropoff_lat"] = float(row["dropoff_latitude"])
        st.session_state["dropoff_lon"] = float(row["dropoff_longitude"])
        st.session_state["passengers"] = int(row["passenger_count"])
        if target_col in row_df.columns:
            st.success(f"Loaded row {int(idx)}. True {target_col} = {float(row[target_col]):.2f}")
        else:
            st.success(f"Loaded row {int(idx)}.")


# ---------------- Inputs ----------------
c1, c2, c3 = st.columns(3)
with c1:
    hour = st.selectbox("Hour", HOURS, key="hour")
    ampm = st.selectbox("AM/PM", AMPM, key="ampm")
    weekday = st.selectbox("Weekday", WDAYS, key="weekday")
with c2:
    pickup_lat = st.number_input("Pickup latitude", format="%.6f", key="pickup_lat")
    pickup_lon = st.number_input("Pickup longitude", format="%.6f", key="pickup_lon")
    passengers = st.slider("Passenger count", 1, 6, key="passengers")
with c3:
    dropoff_lat = st.number_input("Dropoff latitude", format="%.6f", key="dropoff_lat")
    dropoff_lon = st.number_input("Dropoff longitude", format="%.6f", key="dropoff_lon")


def cat_to_code(col, val):
    cats = cat_categories[col]
    try:
        return int(cats.index(val))
    except ValueError:
        # Unknown category (should be rare if you use dropdowns)
        return 0


def build_features():
    # Categorical codes in the same order as training
    cat_vals = {
        "Hour": int(hour),
        "AMorPM": str(ampm),
        "Weekday": str(weekday),
    }
    x_cat = [cat_to_code(c, cat_vals[c]) for c in cat_cols]

    # Continuous in the same order as training
    dist = haversine_km(float(pickup_lat), float(pickup_lon), float(dropoff_lat), float(dropoff_lon))
    cont_vals = {
        "pickup_latitude": float(pickup_lat),
        "pickup_longitude": float(pickup_lon),
        "dropoff_latitude": float(dropoff_lat),
        "dropoff_longitude": float(dropoff_lon),
        "passenger_count": float(passengers),
        "dist_km": float(dist),
    }
    x_cont = np.array([cont_vals[c] for c in cont_cols], dtype=np.float32)
    if standardize:
        x_cont = (x_cont - cont_mean) / (cont_std + 1e-6)
    return np.array(x_cat, dtype=np.int64), x_cont, dist


if st.button("Predict fare", type="primary"):
    x_cat, x_cont, dist = build_features()
    with torch.no_grad():
        t_cat = torch.tensor(x_cat).view(1, -1)
        t_cont = torch.tensor(x_cont).view(1, -1)
        pred = float(model(t_cat, t_cont).cpu().numpy().reshape(-1)[0])

    st.metric("Predicted fare (USD)", f"{pred:.2f}")
    st.caption(f"Computed trip distance: {dist:.2f} km")
