import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hyperparameters from notebook
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1
}

def train_and_save_models(features_csv_path, labels_csv_path, save_dir):
    """
    Trains XGBoost models for IM, AO, and AC events and saves them to save_dir.
    
    features_csv_path: path to CSV containing extracted features (time, freq, wavelet, morph).
    labels_csv_path: path to CSV containing target variables (IM, AO, AC columns).
    """
    print("Loading data...")
    features = pd.read_csv(features_csv_path)
    labels = pd.read_csv(labels_csv_path)
    
    # Merge on Record and Beat
    dataset = pd.merge(features, labels, on=["Record", "Beat"], how="inner")
    print(f"Dataset shape after merge: {dataset.shape}")
    
    X = dataset.drop(columns=["Record", "Beat", "IM", "AO", "AC"])
    y_im = dataset["IM"].values
    y_ao = dataset["AO"].values
    y_ac = dataset["AC"].values
    
    # Train/test split
    X_train, X_test, y_im_train, y_im_test, y_ao_train, y_ao_test, y_ac_train, y_ac_test = train_test_split(
        X, y_im, y_ao, y_ac, test_size=0.2, random_state=42, shuffle=True
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    targets = {
        "im": (y_im_train, y_im_test, "xgb_im.pkl"),
        "ao": (y_ao_train, y_ao_test, "xgb_ao.pkl"),
        "ac": (y_ac_train, y_ac_test, "xgb_ac.pkl")
    }
    
    for name, (y_tr, y_te, filename) in targets.items():
        print(f"\nTraining {name.upper()} model...")
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_tr)
        
        # Predict & Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds)
        
        print(f"{name.upper()} Metrics:")
        print(f"  MAE  : {mae:.3f}")
        print(f"  RMSE : {rmse:.3f}")
        print(f"  R²   : {r2:.4f}")
        
        # Save model
        save_path = os.path.join(save_dir, filename)
        joblib.dump(model, save_path)
        print(f"Saved {name.upper()} model to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost Models for SCG Event Detection")
    parser.add_argument("--features", type=str, required=True, help="Path to features CSV file")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels CSV file")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory to save models")
    
    args = parser.parse_args()
    
    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
    train_and_save_models(args.features, args.labels, outdir)
