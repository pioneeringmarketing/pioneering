# Smart Farming: Federated vs Centralized AI

Streamlit app built from `final.ipynb`. Compares a centralized XGBoost
yield model against a simulated federated-learning (FedAvg) model on
the Smart Farming Crop Yield dataset, plus a crop recommender and
harvest-timing tool.

Models train automatically the first time the app loads (cached after
that) — no separate model files needed, just this folder's three files.

## Files
- `app.py` — the Streamlit app
- `requirements.txt` — pinned dependencies (CPU-only PyTorch)
- `Smart_Farming_Crop_Yield_2024.csv` — training data

## Deploy for free on Streamlit Community Cloud

1. Go to https://github.com/new and create a new **public** repository
   (e.g. `smart-farming-app`). You'll need a free GitHub account if you
   don't already have one.
2. On the new repo's page, click **"uploading an existing file"** and
   drag in all three files from this folder (`app.py`,
   `requirements.txt`, `Smart_Farming_Crop_Yield_2024.csv`). Commit
   the upload. No command line needed.
3. Go to https://share.streamlit.io and sign in with your GitHub
   account (free).
4. Click **"Create app"** → **"Deploy a public app from GitHub"**.
5. Pick the repo you just created, branch `main`, main file path
   `app.py`, then click **Deploy**.
6. First build takes a few minutes (installing PyTorch/XGBoost). Once
   live, you'll get a shareable URL like
   `https://your-app-name.streamlit.app`.

That's it — the link stays live permanently on Streamlit's free tier
(it sleeps after inactivity and wakes on the next visit).
