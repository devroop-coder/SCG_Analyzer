import io
import os
import tempfile
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import wfdb

def parse_wfdb_record(hea_bytes, dat_bytes, base_name):
    """
    Writes .hea and .dat file contents to temp storage, reads them with wfdb, and returns
    (scg_signal, ecg_signal, fs).
    """
    temp_dir = tempfile.gettempdir()
    hea_path = os.path.join(temp_dir, f"{base_name}.hea")
    dat_path = os.path.join(temp_dir, f"{base_name}.dat")
    
    with open(hea_path, "wb") as f:
        f.write(hea_bytes)
    with open(dat_path, "wb") as f:
        f.write(dat_bytes)
        
    try:
        # wfdb loads from the base filename without extension
        record = wfdb.rdrecord(os.path.join(temp_dir, base_name))
        
        scg_signal = None
        ecg_signal = None
        fs = record.fs
        
        for col_idx, name in enumerate(record.sig_name):
            name_lower = name.lower()
            if "scg" in name_lower or "seismo" in name_lower:
                scg_signal = record.p_signal[:, col_idx]
            elif "ii" in name_lower or "i" in name_lower or "ecg" in name_lower:
                # Prioritize Lead II, fallback to others
                if ecg_signal is None or "ii" in name_lower:
                    ecg_signal = record.p_signal[:, col_idx]
                    
        # Cleanup
        os.remove(hea_path)
        os.remove(dat_path)
        
        if scg_signal is None:
            raise ValueError(f"Could not find SCG channel in WFDB record. Available: {record.sig_name}")
            
        return scg_signal, ecg_signal, fs
        
    except Exception as e:
        if os.path.exists(hea_path):
            os.remove(hea_path)
        if os.path.exists(dat_path):
            os.remove(dat_path)
        raise ValueError(f"Error parsing WFDB record: {e}")

def parse_csv_txt(file_content, filename):
    """
    Parses a CSV or TXT file upload.
    Returns a tuple: (scg_signal, ecg_signal, fs)
    Both signals are 1D numpy arrays or None.
    """
    # Read CSV or TXT
    try:
        # Detect delimiter
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        else:
            # Try whitespace or tab delimiter
            df = pd.read_csv(io.StringIO(file_content.decode("utf-8")), sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Error parsing file: {e}")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    # Convert column names to lowercase for robust matching
    cols = [col.lower() for col in df.columns]
    
    scg_signal = None
    ecg_signal = None

    # Search for SCG column
    scg_col_idx = -1
    for i, col in enumerate(cols):
        if "scg" in col or "seismo" in col:
            scg_col_idx = i
            break
    if scg_col_idx == -1 and len(df.columns) > 0:
        # Fallback to the first numeric column if no matching name
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scg_col_idx = df.columns.get_loc(numeric_cols[0])

    if scg_col_idx != -1:
        scg_signal = df.iloc[:, scg_col_idx].values
    else:
        raise ValueError("Could not find a valid numeric column for SCG signal.")

    # Search for ECG column
    ecg_col_idx = -1
    for i, col in enumerate(cols):
        if "ecg" in col or "electro" in col:
            ecg_col_idx = i
            break
    if ecg_col_idx == -1 and len(df.columns) > 1:
        # Fallback to the second numeric column if SCG is the first
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            for col_name in numeric_cols:
                loc = df.columns.get_loc(col_name)
                if loc != scg_col_idx:
                    ecg_col_idx = loc
                    break

    if ecg_col_idx != -1:
        ecg_signal = df.iloc[:, ecg_col_idx].values

    # Check if there is an explicit time/sampling rate column
    fs = None
    for i, col in enumerate(cols):
        if "time" in col or "sec" in col:
            time_vals = df.iloc[:, i].values
            if len(time_vals) > 1:
                diffs = np.diff(time_vals)
                mean_diff = np.mean(diffs)
                if mean_diff > 0:
                    fs = int(round(1.0 / mean_diff))
            break

    return scg_signal, ecg_signal, fs

def butter_bandpass_filter(signal, fs, lowcut=5.0, highcut=40.0, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter to the signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Cap frequencies at slightly below Nyquist to prevent scipy filter errors
    if low <= 0:
        low = 0.01
    if high >= 1.0:
        high = 0.99
        
    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def min_max_normalize(signal):
    """
    Normalizes the signal to the range [0, 1] using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return normalized

def preprocess_scg(raw_scg, fs, lowcut=5.0, highcut=40.0, order=4):
    """
    Pipeline to filter and normalize raw SCG signal.
    """
    filtered = butter_bandpass_filter(raw_scg, fs, lowcut, highcut, order)
    normalized = min_max_normalize(filtered)
    return normalized
