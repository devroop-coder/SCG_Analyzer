import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch, peak_prominences, peak_widths
from scipy.fft import fft
import pywt

def extract_statistical_features(signal):
    """
    Extracts time-domain statistical features.
    """
    features = {}
    features["Mean"] = np.mean(signal)
    features["Std"] = np.std(signal)
    features["Variance"] = np.var(signal)
    features["Maximum"] = np.max(signal)
    features["Minimum"] = np.min(signal)
    features["PeakToPeak"] = np.ptp(signal)
    features["RMS"] = np.sqrt(np.mean(signal**2))
    features["Energy"] = np.sum(signal**2)
    features["Skewness"] = skew(signal)
    features["Kurtosis"] = kurtosis(signal)
    return features

def extract_frequency_wavelet_features(signal, fs=5000):
    """
    Extracts frequency-domain (FFT) and wavelet-domain features.
    """
    features = {}
    
    # FFT Features
    fft_values = np.abs(fft(signal))
    fft_values = fft_values[:len(fft_values)//2]
    freq = np.linspace(0, fs/2, len(fft_values))
    
    features["DominantFrequency"] = freq[np.argmax(fft_values)]
    features["SpectralPower"] = np.sum(fft_values**2)
    
    fft_sum = np.sum(fft_values)
    if fft_sum > 0:
        features["SpectralCentroid"] = np.sum(freq * fft_values) / fft_sum
    else:
        features["SpectralCentroid"] = 0.0
        
    centroid = features["SpectralCentroid"]
    if fft_sum > 0:
        features["SpectralBandwidth"] = np.sqrt(
            np.sum(((freq - centroid)**2) * fft_values) / fft_sum
        )
    else:
        features["SpectralBandwidth"] = 0.0
        
    if fft_sum > 0:
        power = fft_values / fft_sum
        features["SpectralEntropy"] = -np.sum(power * np.log2(power + 1e-12))
    else:
        features["SpectralEntropy"] = 0.0

    # Wavelet Features (db4 wavelet at level 4)
    coeffs = pywt.wavedec(signal, "db4", level=4)
    energy = []
    for c in coeffs:
        energy.append(np.sum(c**2))
        
    for i, e in enumerate(energy):
        features[f"WaveletEnergy_{i}"] = e
        
    energy_sum = np.sum(energy)
    if energy_sum > 0:
        prob = np.array(energy) / energy_sum
        features["WaveletEntropy"] = -np.sum(prob * np.log2(prob + 1e-12))
    else:
        features["WaveletEntropy"] = 0.0
        
    return features

def extract_peak_features(signal, initial_idx, peak_type):
    """
    Refines a peak/valley index and extracts morphological features.
    Returns a dict with amplitude, prominence, peak-to-valley, TOA, normalized TOA, width, and slopes.
    """
    search_window = 10
    left = max(0, initial_idx - search_window)
    right = min(len(signal), initial_idx + search_window + 1)
    
    if len(signal[left:right]) == 0:
        return 0.0, 0.0, 0.0, float(initial_idx), float(initial_idx)/len(signal), 0.0, 0.0, 0.0
        
    if peak_type == "max":
        peak_index = left + np.argmax(signal[left:right])
        work_signal = signal
    else:
        peak_index = left + np.argmin(signal[left:right])
        work_signal = -signal
        
    peak_amp = signal[peak_index]
    
    # Calculate prominence
    prom_out = peak_prominences(work_signal, [peak_index])
    prominence = prom_out[0][0] if len(prom_out[0]) > 0 else 0.0
    
    # Calculate width
    width_out = peak_widths(work_signal, [peak_index], rel_height=0.5)
    width = width_out[0][0] if len(width_out[0]) > 0 else 0.0
    
    # Peak-to-Valley (local window of 100 samples)
    local_window = 100
    l = max(0, peak_index - local_window)
    r = min(len(signal), peak_index + local_window + 1)
    local = signal[l:r]
    
    if peak_type == "max":
        valley = np.min(local)
        peak_to_valley = peak_amp - valley
    else:
        peak_val = np.max(local)
        peak_to_valley = peak_val - peak_amp
        
    # Slopes
    left_idx = max(0, peak_index - 1)
    right_idx = min(len(signal) - 1, peak_index + 1)
    left_slope = signal[peak_index] - signal[left_idx]
    right_slope = signal[right_idx] - signal[peak_index]
    
    return peak_amp, prominence, peak_to_valley, float(peak_index), float(peak_index) / len(signal), width, left_slope, right_slope

def run_feature_extraction_pipeline(beat_signal, fs=5000):
    """
    Processes a single heartbeat segment and returns a dict with all 46 features.
    The returned dictionary keys are in the exact order needed by XGBoost Regressors.
    """
    # 1. Statistical (10)
    features = extract_statistical_features(beat_signal)
    
    # 2. Frequency/Wavelet (11)
    freq_wave_features = extract_frequency_wavelet_features(beat_signal, fs)
    features.update(freq_wave_features)
    
    # 3. Detect initial heuristic locations for peaks
    before = int(0.2 * fs)
    
    # AO is the peak in R + 0.04s to R + 0.12s
    start_ao = before + int(0.04 * fs)
    end_ao = before + int(0.12 * fs)
    AO_init = start_ao + np.argmax(beat_signal[start_ao:end_ao])
    
    # IM is the valley in AO - 0.12s to AO
    start_im = max(0, AO_init - int(0.12 * fs))
    IM_init = start_im + np.argmin(beat_signal[start_im:AO_init])
    
    # AC is the valley in AO + 0.05s to AO + 0.30s
    start_ac = AO_init + int(0.05 * fs)
    end_ac = min(len(beat_signal), AO_init + int(0.30 * fs))
    AC_init = start_ac + np.argmin(beat_signal[start_ac:end_ac])
    
    # 4. Extract morphological features for IM, AO, AC
    peak_configs = [
        ("IM", IM_init, "min"),
        ("AO", AO_init, "max"),
        ("AC", AC_init, "min")
    ]
    
    for name, initial_idx, ptype in peak_configs:
        amp, prom, p2v, toa, norm_toa, width, ls, rs = extract_peak_features(beat_signal, initial_idx, ptype)
        features[f"{name}_Amplitude"] = amp
        features[f"{name}_Prominence"] = prom
        features[f"{name}_PeakToValley"] = p2v
        features[f"{name}_TOA"] = toa
        features[f"{name}_Normalized_TOA"] = norm_toa
        features[f"{name}_PeakWidth"] = width
        features[f"{name}_LeftSlope"] = ls
        features[f"{name}_RightSlope"] = rs
        
    features["RR_Interval"] = len(beat_signal)
    
    # Ensure exact column ordering
    col_order = [
        "Mean", "Std", "Variance", "Maximum", "Minimum", "PeakToPeak", "RMS", "Energy", "Skewness", "Kurtosis",
        "DominantFrequency", "SpectralPower", "SpectralCentroid", "SpectralBandwidth", "SpectralEntropy",
        "WaveletEnergy_0", "WaveletEnergy_1", "WaveletEnergy_2", "WaveletEnergy_3", "WaveletEnergy_4", "WaveletEntropy",
        "IM_Amplitude", "IM_Prominence", "IM_PeakToValley", "IM_TOA", "IM_Normalized_TOA", "IM_PeakWidth", "IM_LeftSlope", "IM_RightSlope",
        "AO_Amplitude", "AO_Prominence", "AO_PeakToValley", "AO_TOA", "AO_Normalized_TOA", "AO_PeakWidth", "AO_LeftSlope", "AO_RightSlope",
        "AC_Amplitude", "AC_Prominence", "AC_PeakToValley", "AC_TOA", "AC_Normalized_TOA", "AC_PeakWidth", "AC_LeftSlope", "AC_RightSlope",
        "RR_Interval"
    ]
    
    ordered_features = {k: features[k] for k in col_order}
    return ordered_features

def build_feature_dataframe(segments, fs=5000):
    """
    Extracts features for all segments and returns a Pandas DataFrame.
    """
    feature_list = []
    for beat in segments:
        feat_dict = run_feature_extraction_pipeline(beat, fs)
        feature_list.append(feat_dict)
    return pd.DataFrame(feature_list)
