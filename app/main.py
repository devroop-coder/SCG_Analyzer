import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import tempfile

# Ensure parent directory of 'app' is in path for robust execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modular packages
from app.processing.preprocessing import parse_csv_txt, preprocess_scg
from app.processing.segmentation import segment_heartbeats
from app.processing.features import build_feature_dataframe, run_feature_extraction_pipeline
from app.ml.prediction import load_models, predict_events
from app.ml.explainability import generate_shap_plots, get_feature_importance
from app.visualization.plots import plot_raw_vs_filtered, plot_segmented_beat_with_events, plot_feature_importance_chart
from app.reports.generator import create_pdf_report

# Page layout configurations
st.set_page_config(
    page_title="AI-Based SCG Analysis Software",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom styling for a premium modern look
st.markdown("""
<style>
    /* Main Layout Styling */
    .main {
        background-color: #F8FAFC;
    }
    h1, h2, h3 {
        color: #1E293B;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Custom Headers and Cards */
    .dashboard-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .dashboard-header h1 {
        color: white;
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .dashboard-header p {
        color: #E2E8F0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Metrics Cards */
    .metric-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Alert styles */
    .info-card {
        padding: 1rem;
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        color: #1E40AF;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate simulated SCG & ECG data
def generate_simulated_data(fs=5000, duration_seconds=12):
    """
    Simulates a high-quality SCG and ECG signal recording.
    """
    total_samples = int(duration_seconds * fs)
    time = np.arange(total_samples) / fs
    
    # Simulate a resting heart rate of 72 bpm -> 1.2s per beat -> 6000 samples per beat
    beat_period = int(1.2 * fs)
    n_beats = int(total_samples / beat_period)
    
    ecg = np.zeros(total_samples)
    scg = np.zeros(total_samples)
    
    # Generate beats
    for b in range(n_beats):
        offset = b * beat_period + int(0.1 * fs)
        if offset >= total_samples:
            break
            
        # 1. ECG QRS Complex
        # R peak (sharp narrow peak)
        r_width = int(0.01 * fs)
        r_range = np.arange(-r_width*3, r_width*3)
        r_peak = np.exp(-r_range**2 / (2 * r_width**2))
        
        # P and T waves
        p_width = int(0.04 * fs)
        p_range = np.arange(-p_width*3, p_width*3)
        p_wave = 0.15 * np.exp(-p_range**2 / (2 * p_width**2))
        
        t_width = int(0.08 * fs)
        t_range = np.arange(-t_width*3, t_width*3)
        t_wave = 0.25 * np.exp(-t_range**2 / (2 * t_width**2))
        
        # Add to ECG
        if offset < total_samples:
            ecg[offset] = 1.0 # R peak
            # Add shape around offset
            for i, idx in enumerate(r_range):
                if 0 <= offset + idx < total_samples:
                    ecg[offset + idx] = r_peak[i]
            # Add P wave (before R-peak)
            p_offset = offset - int(0.15 * fs)
            for i, idx in enumerate(p_range):
                if 0 <= p_offset + idx < total_samples:
                    ecg[p_offset + idx] = p_wave[i]
            # Add T wave (after R-peak)
            t_offset = offset + int(0.3 * fs)
            for i, idx in enumerate(t_range):
                if 0 <= t_offset + idx < total_samples:
                    ecg[t_offset + idx] = t_wave[i]
                    
        # 2. SCG Heartbeat Complex
        # Mitral Closure (IM): negative valley around R + 0.05s (R + 250 samples)
        im_loc = offset + int(0.05 * fs)
        # Aortic Opening (AO): sharp positive peak around R + 0.09s (R + 450 samples)
        ao_loc = offset + int(0.09 * fs)
        # Aortic Closure (AC): negative valley around R + 0.32s (R + 1600 samples)
        ac_loc = offset + int(0.32 * fs)
        
        # Build Gaussian shapes for SCG events
        w = int(0.015 * fs)
        x_range = np.arange(-w*4, w*4)
        
        # Add IM
        im_wave = -0.3 * np.exp(-x_range**2 / (2 * (w*0.8)**2))
        for i, idx in enumerate(x_range):
            if 0 <= im_loc + idx < total_samples:
                scg[im_loc + idx] += im_wave[i]
                
        # Add AO
        ao_wave = 0.8 * np.exp(-x_range**2 / (2 * w**2))
        for i, idx in enumerate(x_range):
            if 0 <= ao_loc + idx < total_samples:
                scg[ao_loc + idx] += ao_wave[i]
                
        # Add AC
        ac_wave = -0.4 * np.exp(-x_range**2 / (2 * (w*1.2)**2))
        for i, idx in enumerate(x_range):
            if 0 <= ac_loc + idx < total_samples:
                scg[ac_loc + idx] += ac_wave[i]
                
        # Add a diastolic harmonic component (sine wave envelope)
        d_start = offset + int(0.38 * fs)
        d_len = int(0.4 * fs)
        d_time = np.arange(d_len) / fs
        d_wave = 0.08 * np.sin(2 * np.pi * 18 * d_time) * np.exp(-8 * d_time)
        for idx in range(d_len):
            if 0 <= d_start + idx < total_samples:
                scg[d_start + idx] += d_wave[idx]
                
    # Add minor high-frequency noise and baseline drift
    scg += 0.04 * np.sin(2 * np.pi * 0.1 * time) # drift
    scg += np.random.normal(0, 0.015, total_samples) # noise
    ecg += np.random.normal(0, 0.01, total_samples)
    
    return scg, ecg

# Session state initialization
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "processed_scg" not in st.session_state:
    st.session_state.processed_scg = None

# Header Banner
st.markdown("""
<div class="dashboard-header">
    <h1>❤️ AI-Based SCG Analysis Software</h1>
    <p>Clinical Decision Support Tool for Seismocardiography Signal Analytics & Heartbeat Keypoint Prediction</p>
</div>
""", unsafe_allow_html=True)

# Check model availability
models = load_models()
models_available = all(models[k] is not None for k in ["im", "ao", "ac"])

# Sidebar controls
st.sidebar.header("🔬 Input Signal Configuration")

# Selection between Upload and Simulation
data_source = st.sidebar.radio("Select Data Source:", ["Simulate Resting SCG Data (Recommended)", "Upload SCG Recording File"])

fs = st.sidebar.number_input("Sampling Rate (Hz):", min_value=100, max_value=20000, value=5000, step=100)

uploaded_files = None
if data_source == "Upload SCG Recording File":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV/TXT file OR WFDB Record (.hea and .dat files together):",
        type=["csv", "txt", "hea", "dat"],
        accept_multiple_files=True
    )

# Track data source change and file removal to reset analysis state
if "prev_data_source" not in st.session_state:
    st.session_state.prev_data_source = data_source

if st.session_state.prev_data_source != data_source:
    st.session_state.analysis_complete = False
    st.session_state.prev_data_source = data_source

if data_source == "Upload SCG Recording File" and not uploaded_files:
    st.session_state.analysis_complete = False

st.sidebar.markdown("---")
st.sidebar.header("📋 Patient Information")
p_name = st.sidebar.text_input("Full Name:", "John Doe")
p_id = st.sidebar.text_input("Patient ID / Record No:", "PT-88392")
p_age = st.sidebar.number_input("Age:", min_value=1, max_value=120, value=45)
p_gender = st.sidebar.selectbox("Gender:", ["Male", "Female", "Other"])
p_dob = st.sidebar.date_input("Date of Birth:", datetime.date(1981, 6, 15))
p_date = st.sidebar.date_input("Study/Analysis Date:", datetime.date.today())

st.sidebar.markdown("---")
run_analysis_btn = st.sidebar.button("🚀 Run Analysis Pipeline", use_container_width=True)

# Main Dashboard logic
if run_analysis_btn:
    with st.spinner("Processing signal, segmenting beats, and detecting cardiac events..."):
        try:
            # 1. Acquire raw data
            file_name_display = ""
            if data_source == "Upload SCG Recording File":
                if not uploaded_files:
                    st.error("Please upload signal files first.")
                    st.stop()
                
                # Check for WFDB record vs CSV/TXT
                hea_file = None
                dat_file = None
                csv_txt_file = None
                
                for f in uploaded_files:
                    if f.name.endswith(".hea"):
                        hea_file = f
                    elif f.name.endswith(".dat"):
                        dat_file = f
                    elif f.name.endswith(".csv") or f.name.endswith(".txt"):
                        csv_txt_file = f
                        
                if hea_file and dat_file:
                    base_name = hea_file.name.replace(".hea", "")
                    hea_bytes = hea_file.getvalue()
                    dat_bytes = dat_file.getvalue()
                    
                    from app.processing.preprocessing import parse_wfdb_record
                    raw_scg, raw_ecg, detected_fs = parse_wfdb_record(hea_bytes, dat_bytes, base_name)
                    file_name_display = f"WFDB Record: {base_name}"
                    if detected_fs is not None:
                        fs = detected_fs
                elif csv_txt_file:
                    raw_scg, raw_ecg, detected_fs = parse_csv_txt(csv_txt_file.getvalue(), csv_txt_file.name)
                    file_name_display = csv_txt_file.name
                    if detected_fs is not None:
                        fs = detected_fs
                else:
                    st.error("Please upload either a single CSV/TXT file, or both .hea and .dat files for a WFDB record.")
                    st.stop()
            else:
                # Simulation
                raw_scg, raw_ecg = generate_simulated_data(fs=fs, duration_seconds=15)
                
            # Keep raw copies for plot
            st.session_state.raw_scg = raw_scg
            st.session_state.raw_ecg = raw_ecg
            st.session_state.fs = fs
            
            # 2. Filter & Normalize
            filtered_scg = preprocess_scg(raw_scg, fs)
            st.session_state.filtered_scg = filtered_scg
            
            # 3. Segmentation
            segments, peak_indices, aligned_ecg = segment_heartbeats(filtered_scg, raw_ecg, fs)
            if len(segments) == 0:
                raise ValueError("Could not segment heartbeats. Verify signal amplitude or ECG columns.")
            st.session_state.segments = segments
            st.session_state.peak_indices = peak_indices
            st.session_state.aligned_ecg = aligned_ecg
            
            # 4. Feature Extraction
            features_df = build_feature_dataframe(segments, fs)
            st.session_state.features_df = features_df
            
            # 5. Prediction
            preds, used_fallback = predict_events(features_df, models)
            st.session_state.preds = preds
            st.session_state.used_fallback = used_fallback
            
            # 6. Calculate Average Statistics
            # PEP is interval from R peak (which is at 0.2 * fs index) to predicted AO
            before_idx = int(0.2 * fs)
            peps = (preds["ao"] - before_idx) / fs * 1000 # in ms
            lvets = (preds["ac"] - preds["ao"]) / fs * 1000 # in ms
            im_aos = (preds["ao"] - preds["im"]) / fs * 1000 # in ms
            
            # Average HR based on peak intervals (ECG or SCG anchors)
            if len(peak_indices) > 1:
                intervals = np.diff(peak_indices) / fs
                hr = 60.0 / np.mean(intervals)
            else:
                hr = 72.0
                
            st.session_state.stats = {
                "hr": hr,
                "pep": np.mean(peps),
                "lvet": np.mean(lvets),
                "im_ao": np.mean(im_aos),
                "peps_list": peps,
                "lvets_list": lvets,
                "im_aos_list": im_aos
            }
            
            # Patient Info state
            st.session_state.patient_data = {
                "name": p_name,
                "id": p_id,
                "age": p_age,
                "gender": p_gender,
                "dob": p_dob.strftime("%Y-%m-%d"),
                "date": p_date.strftime("%Y-%m-%d"),
                "source": "Simulated SCG Source" if data_source.startswith("Simulate") else file_name_display
            }
            
            st.session_state.analysis_complete = True
            st.success("Signal analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Display Dashboard tabs if analysis is complete
if st.session_state.analysis_complete:
    
    # Render quick statistics banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.stats['hr']:.1f}</div>
            <div class="metric-label">Heart Rate (bpm)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.stats['pep']:.1f} ms</div>
            <div class="metric-label">R - AO Interval (PEP)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.stats['lvet']:.1f} ms</div>
            <div class="metric-label">AO - AC Interval (LVET)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{st.session_state.stats['im_ao']:.1f} ms</div>
            <div class="metric-label">IM - AO Interval</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Warning banner
    if st.session_state.used_fallback:
        st.warning("⚠️ XGBoost models (`xgb_im.pkl`, `xgb_ao.pkl`, `xgb_ac.pkl`) were not found in `app/ml/models/`. Detections have fallen back to physiological rule-based heuristic calculations.")
    else:
        st.success("✅ Prediction running on trained XGBoost Machine Learning Models.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Signal Analytics", 
        "💓 Cardiac Events & Intervals", 
        "🧠 AI Explainability (SHAP)", 
        "📄 Export Report"
    ])
    
    with tab1:
        st.subheader("Signal Preprocessing & Filtering Visualizer")
        
        # Allow user to view different length ranges
        duration_slider = st.slider("Signal Visualization Window (seconds):", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
        
        fig_raw = plot_raw_vs_filtered(
            st.session_state.raw_scg,
            st.session_state.filtered_scg,
            st.session_state.fs,
            duration=duration_slider
        )
        st.pyplot(fig_raw)
        
        st.markdown("### Signal Metadata")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Duration", f"{len(st.session_state.raw_scg)/st.session_state.fs:.2f} seconds")
        col_m2.metric("Segmented Beats", f"{len(st.session_state.segments)} beats")
        col_m3.metric("Alignment Reference", "ECG R-Peaks" if st.session_state.aligned_ecg else "SCG Systolic Peak")
        
    with tab2:
        st.subheader("Cardiac Event Annotation & Interventricular Intervals")
        
        # Interactive Beat Selector
        beat_num = st.number_input("Select Heartbeat Segment to inspect:", min_value=0, max_value=len(st.session_state.segments)-1, value=0, step=1)
        
        col_b1, col_b2 = st.columns([7, 3])
        with col_b1:
            fig_beat = plot_segmented_beat_with_events(
                st.session_state.segments[beat_num],
                st.session_state.preds["im"][beat_num],
                st.session_state.preds["ao"][beat_num],
                st.session_state.preds["ac"][beat_num],
                st.session_state.fs,
                beat_num=beat_num
            )
            st.pyplot(fig_beat)
            
        with col_b2:
            st.markdown("#### Beat-Specific Metrics")
            st.markdown(f"""
            - **Mitral Closure (IM) index**: {st.session_state.preds["im"][beat_num]} sample
            - **Aortic Opening (AO) index**: {st.session_state.preds["ao"][beat_num]} sample
            - **Aortic Closure (AC) index**: {st.session_state.preds["ac"][beat_num]} sample
            
            - **PEP (R-AO interval)**: {st.session_state.stats["peps_list"][beat_num]:.1f} ms
            - **LVET (AO-AC interval)**: {st.session_state.stats["lvets_list"][beat_num]:.1f} ms
            - **IM-AO interval**: {st.session_state.stats["im_aos_list"][beat_num]:.1f} ms
            """)
            
        st.markdown("### Heartbeat Intervals Dataset")
        intervals_df = pd.DataFrame({
            "Beat Index": np.arange(len(st.session_state.segments)),
            "HR (bpm)": st.session_state.stats["hr"],
            "R-AO Interval (PEP, ms)": st.session_state.stats["peps_list"],
            "AO-AC Interval (LVET, ms)": st.session_state.stats["lvets_list"],
            "IM-AO Interval (ms)": st.session_state.stats["im_aos_list"]
        })
        st.dataframe(intervals_df, use_container_width=True)
        
        # Download intervals CSV button
        csv_data = intervals_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Beat Intervals CSV",
            data=csv_data,
            file_name=f"{st.session_state.patient_data['id']}_beat_intervals.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab3:
        st.subheader("Model Interpretability & Feature Importances")
        
        if st.session_state.used_fallback:
            st.info("💡 SHAP Explainability plots are unavailable because predictions are running in fallback mode (trained XGBoost model files are missing). Place model pkl files in `app/ml/models/` to unlock SHAP analysis.")
            
            # Render a dummy mock feature importance plot to demonstrate UI
            st.markdown("#### Sample Model Feature Importance (Demostration only)")
            mock_df = pd.DataFrame({
                "Feature": ["AO_Normalized_TOA", "AO_Amplitude", "AO_LeftSlope", "WaveletEnergy_0", "AC_Normalized_TOA", "IM_Normalized_TOA", "Maximum", "Energy", "RMS", "PeakToPeak"],
                "Importance": [0.38, 0.15, 0.11, 0.08, 0.06, 0.05, 0.04, 0.02, 0.01, 0.01]
            })
            fig_mock = plot_feature_importance_chart(mock_df, top_n=8)
            st.pyplot(fig_mock)
        else:
            # Dropdowns to select target event model to inspect
            target_event = st.selectbox("Select Event Model to Analyze:", ["AO (Aortic Opening)", "IM (Mitral Closure)", "AC (Aortic Closure)"])
            model_key = target_event[:2].lower()
            active_model = models[model_key]
            
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                st.markdown("#### Native Feature Importance")
                importance_df = get_feature_importance(active_model, st.session_state.features_df)
                fig_imp = plot_feature_importance_chart(importance_df, top_n=8)
                st.pyplot(fig_imp)
                
            with col_sh2:
                st.markdown("#### SHAP Summary Plot")
                with st.spinner("Generating SHAP values summary..."):
                    fig_shap = generate_shap_plots(active_model, st.session_state.features_df)
                    st.pyplot(fig_shap)
                    
            st.markdown("""
            **Understanding the SHAP Plot**:
            - Each point represents a heartbeat segment sample.
            - The position on the horizontal axis indicates whether the feature increased (right) or decreased (left) the predicted index.
            - Color shows the raw value of the feature (red = high, blue = low).
            """)

    with tab4:
        st.subheader("Compile Patient Diagnostic Report")
        
        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.markdown("#### Report Configuration Summary")
            st.markdown(f"""
            - **Patient Name**: {st.session_state.patient_data['name']}
            - **Patient ID**: {st.session_state.patient_data['id']}
            - **Age / Gender**: {st.session_state.patient_data['age']} / {st.session_state.patient_data['gender']}
            - **Analysis Date**: {st.session_state.patient_data['date']}
            - **Source File**: {st.session_state.patient_data['source']}
            """)
            
        with col_rep2:
            st.markdown("#### PDF Generation Options")
            include_expl = st.checkbox("Include AI Feature Importance details in Report", value=True)
            
            # Generate report to temp file path
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{st.session_state.patient_data['id']}_report.pdf")
            
            # Calculate importance df if models are available
            active_importance_df = None
            if include_expl:
                if not st.session_state.used_fallback:
                    active_importance_df = get_feature_importance(models["ao"], st.session_state.features_df)
                else:
                    # Provide mock details for reporting
                    active_importance_df = pd.DataFrame({
                        "Feature": ["AO_Normalized_TOA", "AO_Amplitude", "AO_LeftSlope", "WaveletEnergy_0", "AC_Normalized_TOA", "IM_Normalized_TOA", "Maximum", "Energy"],
                        "Importance": [0.38, 0.15, 0.11, 0.08, 0.06, 0.05, 0.04, 0.02]
                    })
            
            # Single heartbeat sample to print in report
            samp_beat = st.session_state.segments[0]
            im_val = st.session_state.preds["im"][0]
            ao_val = st.session_state.preds["ao"][0]
            ac_val = st.session_state.preds["ac"][0]
            
            # Trigger build
            create_pdf_report(
                temp_pdf_path,
                st.session_state.patient_data,
                st.session_state.stats,
                st.session_state.raw_scg,
                st.session_state.filtered_scg,
                st.session_state.fs,
                samp_beat,
                im_val,
                ao_val,
                ac_val,
                active_importance_df
            )
            
            with open(temp_pdf_path, "rb") as f:
                pdf_bytes = f.read()
                
            st.download_button(
                label="📥 Generate & Download Diagnostic PDF Report",
                data=pdf_bytes,
                file_name=f"SCG_Report_{st.session_state.patient_data['id']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    # Landing page message
    st.info("👈 Please load a recording file or select 'Simulate' and click 'Run Analysis Pipeline' on the sidebar to inspect the signal, predict cardiac events, and compile PDF reports.")
