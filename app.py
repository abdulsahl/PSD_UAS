import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import aeon
import video_processor
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="Gun vs Point Classification",
    page_icon="🎯", # More neutral icon
    layout="centered",
    initial_sidebar_state="collapsed" # Cleaner look
)

# --- Modern Styling (CSS) ---
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f1116;
    }
    
    /* Headings */
    h1 {
        text-align: center;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-weight: 600;
        color: #e0e0e0;
    }

    /* Cards/Containers */
    .stCard {
        background-color: #1e212b;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2d303e;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #1e212b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    
    /* Grid/Table */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success/Error boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Gun vs Point Time Series Classification")
st.markdown("<p style='text-align: center; color: #888; margin-bottom: 30px;'>Advanced Gesture Recognition System</p>", unsafe_allow_html=True)

# --- Model Loading (Quietly) ---
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        model_a = joblib.load('model_A_full.pkl')
        model_b = joblib.load('model_B_roi.pkl')
        return model_a, model_b, None
    except Exception as e:
        return None, None, str(e)

model_a, model_b, error = load_models()

if error:
    st.error("System Error: Could not initialize classification engine.")
    with st.expander("Technical Details"):
        st.write(error)
    st.stop()

# --- Parsing Logic (Batch Support) ---
def parse_content(content, delimiter=None):
    """
    Parses text content into numpy array.
    Supports single sample (1D) or multiple samples (2D).
    Expected width: 150 points.
    """
    try:
        lines = content.strip().splitlines()
        data = []
        for line in lines:
            # Handle comma or space separation
            line = line.replace(',', ' ')
            parts = [float(x) for x in line.split() if x.strip()]
            if len(parts) > 0:
                data.append(parts)
        
        if not data:
            return None
            
        arr = np.array(data)
        
        # Handle 1D input (single sample) -> make 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        return arr
    except ValueError:
        return None

def process_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        # Handle ARFF specific (skip header)
        if uploaded_file.name.lower().endswith('.arff'):
            if "@data" in content.lower():
                content = content.lower().split("@data")[-1]
        
        return parse_content(content)
    except Exception as e:
        return None

# --- Classification Core ---
def classify_batch(X_batch):
    """
    Auto-detects shape. Expects (N, 150).
    Returns DataFrame with ID, Prediction, Confidence(Proxy).
    """
    # Validation
    if X_batch.shape[1] != 150:
        return None, f"Incorrect data dimensions. Expected 150 time steps, got {X_batch.shape[1]}."

    # Prediction
    # Create ROI set
    X_roi = X_batch[:, 20:60]
    
    # Model A (Full)
    pred_a = model_a.predict(X_batch)
    
    # Model B (ROI)
    pred_b = model_b.predict(X_roi)
    
    results = []
    
    for i in range(len(X_batch)):
        # Ensemble Logic: Safety First (Priority to Gun)
        # Notebook: Gun='1', Point='2'
        p_a = str(pred_a[i])
        p_b = str(pred_b[i])
        
        final = p_a
        if p_b == '1': # Gun detected by ROI
            final = '1'
        elif p_b == '2': # Point detected by ROI
            final = p_a # Trust Full model (or it aligns with A)
            
        label = "GUN-DRAW" if final == '1' else "POINT"
        results.append({
            "Sample ID": i + 1,
            "Prediction": label,
            "status": final # for coloring
        })
        
    return pd.DataFrame(results), None

# --- Main Layout ---

# Input Container
with st.container():

    
    # Video Upload Section
    st.info("ℹ️ **Petunjuk Penggunaan:** Upload video berdurasi sekitar **5 detik** di mana aktor **menghadap ke samping** (profil).")
    
    uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'mov', 'avi'])
    
    X_input = None
    
    if uploaded_file:
        # Video Processing
        with st.spinner("Analyzing Video Structure..."):
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                
                vp = video_processor.VideoProcessor()
                # Extract signal and annotated video path
                raw_signal, annotated_video_path = vp.extract_signal(video_path)
                
                if raw_signal is not None:
                    X_input = vp.preprocess_signal(raw_signal)
                    st.success("Video processed successfully! Extracted hand movement.")
                    
                    # Layout for Video and Signal
                    col_video, col_signal = st.columns(2)
                    
                    with col_video:
                        st.markdown("#### Tracked Hand (Centroid)")
                        if annotated_video_path:
                            st.video(annotated_video_path)
                    
                    with col_signal:
                        st.markdown("#### Extracted Signal (Horizontal Motion)")
                        fig_debug, ax_debug = plt.subplots(figsize=(8, 4))
                        ax_debug.plot(X_input[0], color='cyan')
                        ax_debug.set_facecolor('#1e212b')
                        fig_debug.patch.set_facecolor('#1e212b')
                        ax_debug.tick_params(colors='white')
                        ax_debug.spines['bottom'].set_color('white')
                        ax_debug.spines['left'].set_color('white')
                        st.pyplot(fig_debug)
                else:
                    st.error("Could not detect hand in video. Please ensure the right hand is visible.")
                    
                # Cleanup
                tfile.close()
                os.unlink(video_path)
                
            except Exception as e:
                st.error(f"Video Processing Error: {str(e)}")

# Analysis Container
if X_input is not None:
    st.markdown("---")
    
    with st.spinner("Processing Signal..."):
        results_df, error_msg = classify_batch(X_input)
    
    if error_msg:
        st.error(error_msg)
    else:
        # Summary Metrics
        n_samples = len(results_df)
        n_guns = len(results_df[results_df['status'] == '1'])
        n_points = len(results_df[results_df['status'] == '2'])
        
        # Display Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples Processed", n_samples)
        c2.metric("Threats Detected (Gun)", n_guns, delta_color="inverse") # Inverse? Gun is bad usually
        c3.metric("Safe Gestures (Point)", n_points)
        
        # Visuals for Single Sample
        if n_samples == 1:
            # Big Result Card
            status = results_df.iloc[0]['status']
            label = results_df.iloc[0]['Prediction']
            
            color = "#FF4B4B" if status == '1' else "#4CAF50"
            icon = "🚨" if status == '1' else "✅"
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 30px; border-radius: 12px; border: 2px solid {color}; text-align: center; margin: 20px 0;">
                <div style="font-size: 3rem; margin-bottom: 10px;">{icon}</div>
                <h1 style="color: {color}; margin: 0; background: none; -webkit-text-fill-color: {color};">{label}</h1>
                <p style="color: #ccc; margin-top: 10px;">Analysis Complete</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot
            st.markdown("#### Morphological Trace")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(X_input[0], color=color, linewidth=2)
            ax.set_facecolor('#1e212b')
            fig.patch.set_facecolor('#0f1116')
            ax.grid(color='#333', linestyle=':', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#555')
            ax.spines['left'].set_color('#555')
            ax.tick_params(colors='#888')
            st.pyplot(fig)
            
        else:
            # Batch Results Table
            st.subheader("Batch Results")
            
            # Styled Table
            def highlight_threat(val):
                color = '#ff4b4b44' if val == 'GUN-DRAW' else '#4caf5044'
                return f'background-color: {color}'

            st.dataframe(
                results_df[['Sample ID', 'Prediction']],
                use_container_width=True,
                height=300
            )
            
            # Download
            csv = results_df[['Sample ID', 'Prediction']].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Report",
                csv,
                "classification_results.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Sample Inspector
            with st.expander("Inspect Individual Sample"):
                sample_id = st.number_input("Enter Sample ID to visualize", min_value=1, max_value=n_samples, value=1)
                idx = sample_id - 1
                
                sel_status = results_df.iloc[idx]['status']
                sel_color = "#FF4B4B" if sel_status == '1' else "#4CAF50"
                
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(X_input[idx], color=sel_color, linewidth=2)
                ax.set_title(f"Sample {sample_id}: {results_df.iloc[idx]['Prediction']}", color='white')
                ax.set_facecolor('#1e212b')
                fig.patch.set_facecolor('#1e212b') # Match card bg
                ax.tick_params(colors='#888')
                st.pyplot(fig)

elif X_input is None:
    # Empty State
    pass
