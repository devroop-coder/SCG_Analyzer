import matplotlib.pyplot as plt
import numpy as np

# Set clean, modern styling parameters
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

def plot_raw_vs_filtered(raw_signal, filtered_signal, fs, duration=5.0):
    """
    Plots raw vs filtered SCG signals for a given duration.
    """
    samples = int(duration * fs)
    samples = min(samples, len(raw_signal), len(filtered_signal))
    
    time = np.arange(samples) / fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    ax1.plot(time, raw_signal[:samples], color="#E06666", alpha=0.9, linewidth=1.0)
    ax1.set_title("Raw SCG Signal", fontsize=12, fontweight="bold", color="#333333")
    ax1.set_ylabel("Amplitude (V)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.plot(time, filtered_signal[:samples], color="#3D85C6", alpha=0.9, linewidth=1.2)
    ax2.set_title("Filtered SCG Signal (5 - 40 Hz Butterworth)", fontsize=12, fontweight="bold", color="#333333")
    ax2.set_ylabel("Normalized Amplitude", fontsize=10)
    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_segmented_beat_with_events(beat_signal, im_idx, ao_idx, ac_idx, fs, beat_num=0):
    """
    Plots a single heartbeat segment with labeled cardiac event detections.
    """
    time = np.arange(len(beat_signal)) / fs * 1000 # Convert to milliseconds
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the SCG beat
    ax.plot(time, beat_signal, color="#4A4A4A", linewidth=1.8, label="SCG Heartbeat")
    
    # R-peak annotation (which occurs at 0.2s = 200ms index in ECG alignment)
    r_peak_time = (0.2 * fs) / fs * 1000
    ax.axvline(x=r_peak_time, color="#CCCCCC", linestyle="--", linewidth=1.2, label="R-Peak Alignment")
    ax.text(r_peak_time + 5, np.max(beat_signal) * 0.9, "R-Peak", color="#888888", fontsize=9, fontweight="bold")
    
    # Annotate IM (Mitral Closure)
    im_time = im_idx / fs * 1000
    ax.scatter(im_time, beat_signal[im_idx], color="#E06666", s=80, zorder=5)
    ax.axvline(x=im_time, color="#E06666", linestyle=":", linewidth=1.5)
    ax.text(im_time + 5, beat_signal[im_idx], "IM", color="#990000", fontsize=10, fontweight="bold")
    
    # Annotate AO (Aortic Opening)
    ao_time = ao_idx / fs * 1000
    ax.scatter(ao_time, beat_signal[ao_idx], color="#6AA84F", s=80, zorder=5)
    ax.axvline(x=ao_time, color="#6AA84F", linestyle=":", linewidth=1.5)
    ax.text(ao_time + 5, beat_signal[ao_idx], "AO", color="#274E13", fontsize=10, fontweight="bold")
    
    # Annotate AC (Aortic Closure)
    ac_time = ac_idx / fs * 1000
    ax.scatter(ac_time, beat_signal[ac_idx], color="#3D85C6", s=80, zorder=5)
    ax.axvline(x=ac_time, color="#3D85C6", linestyle=":", linewidth=1.5)
    ax.text(ac_time + 5, beat_signal[ac_idx], "AC", color="#0B5394", fontsize=10, fontweight="bold")
    
    ax.set_title(f"SCG Heartbeat Segment #{beat_num} (Cardiac Events)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (milliseconds)", fontsize=10)
    ax.set_ylabel("Normalized Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    return fig

def plot_feature_importance_chart(importance_df, top_n=10):
    """
    Renders a bar chart of the top N features.
    """
    df_sorted = importance_df.head(top_n).sort_values(by="Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(df_sorted)))
    bars = ax.barh(df_sorted["Feature"], df_sorted["Importance"], color=colors, height=0.6)
    
    ax.set_title(f"Top {top_n} Features (XGBoost Feature Importance)", fontsize=12, fontweight="bold")
    ax.set_xlabel("F-Score / Relative Importance", fontsize=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005 * np.max(df_sorted["Importance"]), bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", 
                va="center", ha="left", fontsize=9, color="#444444")
                
    plt.tight_layout()
    return fig
