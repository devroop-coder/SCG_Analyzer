import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import io

def generate_shap_plots(model, features_df, max_samples=200):
    """
    Computes SHAP values for the model and returns a matplotlib figure of the summary plot.
    """
    # Sample data to keep computations responsive in the UI
    sample_df = features_df.sample(n=min(max_samples, len(features_df)), random_state=42) if len(features_df) > max_samples else features_df
    
    # Initialize tree explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw summary plot on the axis
    shap.summary_plot(
        shap_values,
        sample_df,
        show=False,
        max_display=12
    )
    
    plt.tight_layout()
    return fig

def get_feature_importance(model, features_df):
    """
    Returns a sorted DataFrame of feature importances based on the model's native feature_importances_.
    """
    importances = model.feature_importances_
    features = features_df.columns
    
    df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    return df
