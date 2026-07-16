import numpy as np
import scipy.signal as sig
import neurokit2 as nk

def detect_r_peaks_ecg(ecg, fs):
    """
    Detects R peaks in the ECG signal using NeuroKit2.
    Returns a numpy array of indices.
    """
    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        return r_peaks
    except Exception as e:
        # Fallback using scipy find_peaks if neurokit fails
        # R peaks are sharp and high
        std_val = np.std(ecg)
        mean_val = np.mean(ecg)
        peaks, _ = sig.find_peaks(ecg, distance=int(0.5 * fs), height=mean_val + 1.5 * std_val)
        return peaks

def detect_peaks_scg(filtered_scg, fs):
    """
    Detects candidate peaks directly on the filtered SCG signal for ECG-less segmentation.
    Uses find_peaks with distance constraints based on typical heart rate ranges.
    """
    # Healthy resting heart rate is 50-120 bpm -> 0.5s to 1.2s interval.
    # We set min distance of 0.5s to avoid double peak detection.
    min_dist = int(0.5 * fs)
    
    # Calculate prominence or height threshold
    # Prominence of peaks should be high
    peaks, _ = sig.find_peaks(filtered_scg, distance=min_dist, prominence=0.1 * np.ptp(filtered_scg))
    return peaks

def segment_heartbeats(scg, ecg=None, fs=5000):
    """
    Segments the SCG signal into individual heartbeats.
    If ECG is provided, segments are aligned using ECG R-peaks.
    If ECG is not provided, segments are aligned using candidate peaks detected in the SCG.
    
    Returns a tuple: (segments, peak_indices, aligned_to_ecg)
      segments: 2D numpy array of shape (N_beats, segment_length)
      peak_indices: list of indices corresponding to the R peak or SCG candidate peak
      aligned_to_ecg: Boolean indicating if ECG-alignment was used.
    """
    before = int(0.2 * fs)
    after = int(0.8 * fs) - before # Total window is 0.8s
    
    segments = []
    peak_indices = []
    aligned_to_ecg = False
    
    if ecg is not None and len(ecg) == len(scg):
        r_peaks = detect_r_peaks_ecg(ecg, fs)
        if len(r_peaks) > 0:
            aligned_to_ecg = True
            for r in r_peaks:
                start = r - before
                end = r + after
                if start >= 0 and end < len(scg):
                    segments.append(scg[start:end])
                    peak_indices.append(r)
                    
    # Fallback to SCG peak-based segmentation if ECG is not available or peak detection failed
    if not aligned_to_ecg:
        scg_peaks = detect_peaks_scg(scg, fs)
        # For SCG peak, the actual peak corresponds roughly to the systolic AO wave,
        # which is about 0.05s-0.12s after R-peak. So we shift the window back by ~0.26s.
        shift = int(0.26 * fs)
        for p in scg_peaks:
            start = p - shift
            end = start + int(0.8 * fs)
            if start >= 0 and end < len(scg):
                segments.append(scg[start:end])
                peak_indices.append(p)
                
    return np.array(segments), peak_indices, aligned_to_ecg
