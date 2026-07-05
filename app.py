"""
Smart Farming: Federated vs Centralized AI
--------------------------------------------
Streamlit app derived from final.ipynb. Trains a centralized XGBoost
regressor and a simulated federated-learning MLP (FedAvg) on the
Smart Farming Crop Yield dataset, then lets the user compare
predictions, get a crop recommendation, and find the optimal
harvest window.

Models are trained once per app instance (cached with
st.cache_resource) directly from the bundled CSV -- no separate
pickled model files are required, which keeps the repo lightweight
and avoids any local/library-version mismatches when deployed.
"""

import os
import copy

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

st.set_page_config(page_title="Smart Farming AI", page_icon="\U0001F33E", layout="wide")

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "Smart_Farming_Crop_Yield_2024.csv")
TARGET = "yield_kg_per_hectare"
CAT_COLS = ["region", "crop_type", "irrigation_type", "fertilizer_type", "crop_disease_status"]
NUMERIC_INPUTS = [
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%",
    "sunlight_hours", "pesticide_usage_ml", "total_days", "days_to_harvest",
    "latitude", "longitude", "NDVI_index",
]


class MLP(nn.Module):
    """Small feed-forward net used as the federated (FedAvg) model."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def get_model_params(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}


def set_model_params(model, params):
    model.load_state_dict({k: v.clone() for k, v in params.items()})


def average_params(param_list, weights):
    avg = {k: torch.zeros_like(v) for k, v in param_list[0].items()}
    for p, w in zip(param_list, weights):
        for k in p:
            avg[k] += p[k].float() * w
    return avg


@st.cache_resource(show_spinner="Training centralized + federated models (one-time, ~10-20s)...")
def load_and_train():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    raw_df = pd.read_csv(DATA_PATH)

    # Dropdown options come from the raw (pre-encoding) data
    options = {c: sorted(raw_df[c].dropna().unique().tolist()) for c in CAT_COLS if c in raw_df.columns}

    df = raw_df.copy()
    for c in ["farm_id", "sensor_id", "timestamp"]:
        if c in df.columns:
            df = df.drop(columns=c)

    if all(c in df.columns for c in ["sowing_date", "harvest_date"]):
        df["sowing_date"] = pd.to_datetime(df["sowing_date"], errors="coerce")
        df["harvest_date"] = pd.to_datetime(df["harvest_date"], errors="coerce")
        df["days_to_harvest"] = (df["harvest_date"] - df["sowing_date"]).dt.days
        df = df.drop(columns=["sowing_date", "harvest_date"])

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("None")
            if "None" not in options.get(c, []):
                options.setdefault(c, []).append("None")

    df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=False)
    df = df.dropna(subset=[TARGET]).fillna(df.median(numeric_only=True)).astype(float)

    model_columns = df.drop(columns=[TARGET]).columns.tolist()
    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    y_train_full = y_train.ravel()

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_x.transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train)

    # ---- Centralized XGBoost ----
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=SEED,
        verbosity=0,
    )
    xgb_model.fit(X_train_scaled, y_train_full)
    y_pred_xgb = xgb_model.predict(X_test_scaled).reshape(-1, 1)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
    r2_xgb = float(r2_score(y_test, y_pred_xgb))

    # ---- Federated (FedAvg) MLP ----
    device = torch.device("cpu")
    in_dim = X_train_scaled.shape[1]
    num_clients, rounds, local_epochs, batch_size, lr = 5, 20, 3, 32, 1e-3

    indices = np.arange(X_train_scaled.shape[0])
    client_splits = np.array_split(indices, num_clients)
    client_loaders = []
    for idxs in client_splits:
        Xc = torch.tensor(X_train_scaled[idxs], dtype=torch.float32)
        yc = torch.tensor(y_train_scaled[idxs], dtype=torch.float32)
        client_loaders.append(DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=True))

    client_sizes = [len(idxs) for idxs in client_splits]
    total_train = sum(client_sizes)
    weights = [s / total_train for s in client_sizes]

    global_model = MLP(in_dim).to(device)
    for _ in range(1, rounds + 1):
        local_params = []
        for cid in range(num_clients):
            client_model = MLP(in_dim).to(device)
            set_model_params(client_model, get_model_params(global_model))
            opt = optim.Adam(client_model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            client_model.train()
            for _epoch in range(local_epochs):
                for xb, yb in client_loaders[cid]:
                    opt.zero_grad()
                    loss = loss_fn(client_model(xb), yb)
                    loss.backward()
                    opt.step()
            local_params.append(get_model_params(client_model))
        set_model_params(global_model, average_params(local_params, weights))

    global_model.eval()
    with torch.no_grad():
        preds_scaled = global_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        preds_real = scaler_y.inverse_transform(preds_scaled)
    rmse_fed = float(np.sqrt(mean_squared_error(y_test, preds_real)))
    r2_fed = float(r2_score(y_test, preds_real))

    return {
        "options": options,
        "model_columns": model_columns,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "xgb": xgb_model,
        "fed_model": global_model,
        "metrics": {"rmse_xgb": rmse_xgb, "r2_xgb": r2_xgb, "rmse_fed": rmse_fed, "r2_fed": r2_fed},
    }


def process_inputs(user_inputs, res):
    data = {col: 0.0 for col in res["model_columns"]}
    for key in NUMERIC_INPUTS:
        if key in data:
            data[key] = float(user_inputs[key])
    for key, value in user_inputs.items():
        col_name = f"{key}_{value}"
        if col_name in data:
            data[col_name] = 1.0
    df_in = pd.DataFrame([data])[res["model_columns"]]
    return res["scaler_x"].transform(df_in)


def predict_fed(X_scaled, res):
    with torch.no_grad():
        p_sc = res["fed_model"](torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    return float(res["scaler_y"].inverse_transform(p_sc)[0][0])


def predict_xgb(X_scaled, res):
    return float(res["xgb"].predict(X_scaled)[0])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
res = load_and_train()

st.title("\U0001F33E Smart Farming: Federated vs Centralized AI")
st.markdown(
    "Compare a **Centralized XGBoost model** against a **Federated model (FedAvg)** "
    "trained on the Smart Farming Crop Yield dataset."
)

m = res["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Centralized RMSE", f"{m['rmse_xgb']:.1f} kg/ha")
c2.metric("Centralized R²", f"{m['r2_xgb']:.3f}")
c3.metric("Federated RMSE", f"{m['rmse_fed']:.1f} kg/ha")
c4.metric("Federated R²", f"{m['r2_fed']:.3f}")

st.sidebar.header("Farm Conditions")
inputs = {}
for cat, opts in res["options"].items():
    inputs[cat] = st.sidebar.selectbox(cat.replace("_", " ").title(), opts)

st.sidebar.subheader("Soil & Weather")
inputs["soil_moisture_%"] = st.sidebar.slider("Soil Moisture (%)", 10.0, 90.0, 30.0)
inputs["soil_pH"] = st.sidebar.slider("Soil pH", 4.0, 9.0, 6.5)
inputs["temperature_C"] = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 25.0)
inputs["rainfall_mm"] = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
inputs["humidity_%"] = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)
inputs["sunlight_hours"] = st.sidebar.slider("Sunlight Hours", 2.0, 14.0, 8.0)

st.sidebar.subheader("Operations")
inputs["pesticide_usage_ml"] = st.sidebar.number_input("Pesticide (ml)", 0.0, 100.0, 10.0)
inputs["total_days"] = st.sidebar.number_input("Total Days", 60, 200, 120)
inputs["days_to_harvest"] = st.sidebar.number_input("Days to Harvest", 60, 200, 120)

st.sidebar.subheader("Location")
inputs["latitude"] = st.sidebar.number_input("Latitude", -90.0, 90.0, 20.0)
inputs["longitude"] = st.sidebar.number_input("Longitude", -180.0, 180.0, 78.0)
inputs["NDVI_index"] = st.sidebar.slider("NDVI Index", -1.0, 1.0, 0.5)

X_scaled = process_inputs(inputs, res)

tab1, tab2, tab3 = st.tabs(["\U0001F4CA Yield Prediction", "\U0001F331 Crop Recommender", "\U0001F4C5 Harvest Timing"])

with tab1:
    st.subheader("Yield Prediction Comparison")
    colA, colB = st.columns(2)
    with colA:
        st.info("**Federated Model (FedAvg)**")
        st.metric("Predicted Yield", f"{predict_fed(X_scaled, res):.2f} kg/ha")
        st.caption("Trained on decentralized data partitions.")
    with colB:
        st.success("**Centralized Model (XGBoost)**")
        st.metric("Predicted Yield", f"{predict_xgb(X_scaled, res):.2f} kg/ha")
        st.caption("Trained on the full dataset at once.")

with tab2:
    st.subheader("Best Crop Recommendation")
    st.write("Using the federated model to find the best crop for the given conditions.")
    recs = []
    for c in res["options"].get("crop_type", []):
        tmp = inputs.copy()
        tmp["crop_type"] = c
        Xt = process_inputs(tmp, res)
        recs.append({"Crop": c, "Yield": predict_fed(Xt, res)})
    rdf = pd.DataFrame(recs).sort_values("Yield", ascending=False)
    if not rdf.empty:
        best = rdf.iloc[0]
        st.success(f"\U0001F3C6 Recommendation: **{best['Crop']}** ({best['Yield']:.0f} kg/ha)")
        st.bar_chart(rdf.set_index("Crop"))

with tab3:
    st.subheader("Harvest Optimization")
    st.write("Projected yield over time (federated model):")
    days = []
    curr = int(inputs["days_to_harvest"])
    for d in range(max(60, curr - 30), curr + 31, 5):
        tmp = inputs.copy()
        tmp["days_to_harvest"] = d
        tmp["total_days"] = d
        Xt = process_inputs(tmp, res)
        days.append({"Days": d, "Yield": predict_fed(Xt, res)})
    ddf = pd.DataFrame(days)
    if not ddf.empty:
        peak = ddf.loc[ddf["Yield"].idxmax()]
        st.info(f"\U0001F4C5 Optimum Harvest: **{int(peak['Days'])} days**")
        st.line_chart(ddf.set_index("Days"))

st.caption("Model source: final.ipynb (centralized XGBoost baseline vs. simulated FedAvg federated learning).")
