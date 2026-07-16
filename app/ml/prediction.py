import os
import joblib
import numpy as np
import pandas as pd

# Define paths where the models might be located
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def load_models():
    """
    Tries to load the trained XGBoost models.
    Returns a dictionary of models, or None for individual models if loading fails.
    """
    models = {
        "im": None,
        "ao": None,
        "ac": None
    }
    
    APP_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    model_paths = {
        "im": [
            os.path.join(MODEL_DIR, "xgb_im.pkl"),
            os.path.join(APP_MODEL_DIR, "xgb_im.pkl"),
            "/Users/devroop/Documents/Programs/Projects/SCG/SCG_Analyzer/models/xgb_im.pkl"
        ],
        "ao": [
            os.path.join(MODEL_DIR, "xgb_ao.pkl"),
            os.path.join(APP_MODEL_DIR, "xgb_ao.pkl"),
            "/Users/devroop/Documents/Programs/Projects/SCG/SCG_Analyzer/models/xgb_ao.pkl"
        ],
        "ac": [
            os.path.join(MODEL_DIR, "xgb_ac.pkl"),
            os.path.join(APP_MODEL_DIR, "xgb_ac.pkl"),
            "/Users/devroop/Documents/Programs/Projects/SCG/SCG_Analyzer/models/xgb_ac.pkl"
        ]
    }
    
    for key, paths in model_paths.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    models[key] = joblib.load(path)
                    break
                except Exception:
                    pass
                    
    return models

def predict_events(features_df, models=None):
    """
    Predicts the sample locations of IM, AO, and AC in the heartbeat segments.
    features_df: Pandas DataFrame containing the 46 features.
    models: preloaded models dict. If None, tries to load them.
    
    Returns a tuple: (preds_dict, used_fallback)
      preds_dict: dict containing numpy arrays of predictions: {"im": [...], "ao": [...], "ac": [...]}
      used_fallback: Boolean indicating if the fallback heuristic was used due to missing models.
    """
    if models is None:
        models = load_models()
        
    preds = {}
    used_fallback = False
    
    # 1. IM Prediction
    if models["im"] is not None:
        try:
            preds["im"] = models["im"].predict(features_df)
        except Exception:
            preds["im"] = features_df["IM_TOA"].values
            used_fallback = True
    else:
        preds["im"] = features_df["IM_TOA"].values
        used_fallback = True
        
    # 2. AO Prediction
    if models["ao"] is not None:
        try:
            preds["ao"] = models["ao"].predict(features_df)
        except Exception:
            preds["ao"] = features_df["AO_TOA"].values
            used_fallback = True
    else:
        preds["ao"] = features_df["AO_TOA"].values
        used_fallback = True
        
    # 3. AC Prediction
    if models["ac"] is not None:
        try:
            preds["ac"] = models["ac"].predict(features_df)
        except Exception:
            preds["ac"] = features_df["AC_TOA"].values
            used_fallback = True
    else:
        preds["ac"] = features_df["AC_TOA"].values
        used_fallback = True
        
    # Ensure all predictions are rounded to nearest integer (sample indices) and bounded
    rr_intervals = features_df["RR_Interval"].values
    
    for key in ["im", "ao", "ac"]:
        preds[key] = np.clip(np.round(preds[key]).astype(int), 0, rr_intervals - 1)
        
    return preds, used_fallback
