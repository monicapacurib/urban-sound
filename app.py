import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import librosa.display
import tempfile
import os
from datetime import datetime
import soundfile as sf

# Set page config
st.set_page_config(
    page_title="CommunitySoundscape",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .safety-high {
        background-color: #ff6b6b;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .safety-medium {
        background-color: #ffd93d;
        padding: 10px;
        border-radius: 5px;
        color: black;
        font-weight: bold;
        text-align: center;
    }
    .safety-low {
        background-color: #6bcf7f;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .safety-unknown {
        background-color: #95a5a6;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_models():
    """Load the trained model and preprocessing objects"""
    try:
        model = load_model('urban_sound_classifier.h5')
        label_encoder = joblib.load('label_encoder.pkl')
        feature_scaler = joblib.load('feature_scaler.pkl')
        class_info = joblib.load('class_info.pkl')
        return model, label_encoder, feature_scaler, class_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def extract_features_for_prediction(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extract features from audio file for prediction"""
    try:
        SAMPLE_RATE = 22050
        DURATION = 4
        SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
        
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Ensure consistent length
        if len(audio) > SAMPLES_PER_TRACK:
            audio = audio[:SAMPLES_PER_TRACK]
        else:
            audio = np.pad(audio, (0, max(0, SAMPLES_PER_TRACK - len(audio))))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Return mean of MFCCs across time
        return np.mean(mfccs.T, axis=0)
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def create_audio_visualizations(audio_path):
    """Create audio visualizations"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
        axes[0, 0].set_title('Audio Waveform', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram', fontweight='bold')
        plt.colorbar(img, ax=axes[0, 1])
        
        # Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title('Mel Spectrogram', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=axes[1, 0], format='%+2.0f dB')
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title('MFCC Coefficients', fontweight='bold')
        axes[1, 1].set_ylabel('MFCC')
        plt.colorbar(img, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏙️ CommunitySoundscape</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Urban Sound Monitor for Public Safety")
    st.markdown("---")
    
    # Load models
    model, label_encoder, feature_scaler, class_info = load_ml_models()
    if model is None:
        st.error("🚨 Failed to load ML models. Please ensure all model files are properly uploaded.")
        st.info("Required model files: urban_sound_classifier.h5, label_encoder.pkl, feature_scaler.pkl, class_info.pkl")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.6, 
        step=0.1,
        help="Higher values require more confidence in predictions"
    )
    
    # Initialize session state for history
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎤 Upload Audio for Analysis")
        
        # Enhanced file uploader with better UI
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload audio file", 
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="Upload urban sound recordings for safety analysis",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            st.write("📄 File Details:")
            for key, value in file_details.items():
                st.write(f"   • {key}: {value}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file.getvalue(), format='audio/wav')
            
            # Analyze button
            if st.button("🔍 Analyze Sound", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio features..."):
                    # Extract features and predict
                    features = extract_features_for_prediction(audio_path)
                    
                    if features is not None:
                        # Scale features
                        features_scaled = feature_scaler.transform(features.reshape(1, -1))
                        features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)
                        
                        # Make prediction
                        prediction = model.predict(features_reshaped, verbose=0)
                        confidence = np.max(prediction)
                        class_index = np.argmax(prediction)
                        class_name = label_encoder.classes_[class_index]
                        
                        # Safety assessment
                        safety_categories = class_info.get('safety_categories', {})
                        safety_level, safety_class = safety_categories.get(
                            class_name, ('UNKNOWN', 'safety-unknown')
                        )
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Results in columns
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric(
                                "Detected Sound", 
                                class_name.replace('_', ' ').title(),
                                help="The type of urban sound detected"
                            )
                        
                        with result_col2:
                            st.metric(
                                "Confidence", 
                                f"{confidence:.2%}",
                                delta=None,
                                help="Model confidence in the prediction"
                            )
                        
                        with result_col3:
                            st.markdown(
                                f'<div class="{safety_class}">{safety_level}</div>', 
                                unsafe_allow_html=True
                            )
                        
                        # Confidence warning
                        if confidence < confidence_threshold:
                            st.warning(
                                f"⚠️ Low confidence prediction (below {confidence_threshold:.0%}). "
                                "This might be an unusual or ambiguous sound."
                            )
                        
                        # Create visualizations
                        st.subheader("📈 Audio Analysis Visualizations")
                        fig = create_audio_visualizations(audio_path)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.error("Could not create audio visualizations")
                        
                        # Log analysis
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        analysis_log = {
                            'timestamp': timestamp,
                            'sound_type': class_name,
                            'confidence': confidence,
                            'safety_level': safety_level,
                            'filename': uploaded_file.name
                        }
                        
                        st.session_state.analysis_history.append(analysis_log)
            
            # Clean up temporary file
            try:
                os.unlink(audio_path)
            except:
                pass
        else:
            # Show instructions when no file is uploaded
            st.info("💡 **How to use:**")
            st.write("1. Upload an audio file (WAV, MP3, FLAC, M4A, OGG)")
            st.write("2. Click 'Analyze Sound' to process the audio")
            st.write("3. View the AI analysis results and safety assessment")
            st.write("4. Check the visualizations for detailed audio analysis")
    
    with col2:
        st.subheader("🏙️ Urban Sound Dashboard")
        
        # Sound type distribution
        st.markdown("**Urban Sound Categories**")
        
        # Create a sample distribution chart
        sound_types = list(class_info.get('safety_categories', {}).keys())
        safety_levels = list(class_info.get('safety_categories', {}).values())
        
        # Count by safety level
        safety_counts = {}
        for level in safety_levels:
            safety_counts[level] = safety_counts.get(level, 0) + 1
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#95a5a6']
        wedges, texts, autotexts = ax.pie(
            list(safety_counts.values()), 
            labels=list(safety_counts.keys()),
            autopct='%1.1f%%',
            colors=colors[:len(safety_counts)],
            startangle=90
        )
        
        # Style the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Sound Safety Level Distribution', fontweight='bold')
        st.pyplot(fig)
        
        # Analysis history
        st.subheader("📋 Recent Analyses")
        if st.session_state.analysis_history:
            recent_analyses = st.session_state.analysis_history[-5:]  # Last 5 analyses
            
            for analysis in reversed(recent_analyses):
                safety_class_map = {
                    'HIGH SAFETY CONCERN': 'safety-high',
                    'EMERGENCY VEHICLE': 'safety-high',
                    'TRAFFIC ALERT': 'safety-medium',
                    'NOISE POLLUTION': 'safety-medium',
                    'TRAFFIC NOISE': 'safety-low',
                    'NORMAL': 'safety-low',
                    'UNKNOWN': 'safety-unknown'
                }
                
                safety_class = safety_class_map.get(analysis['safety_level'], 'safety-unknown')
                
                st.markdown(f"""
                <div style="border-left: 4px solid #1f77b4; padding: 10px; margin: 5px 0; background: #f8f9fa;">
                    <strong>{analysis['timestamp']}</strong><br>
                    Sound: {analysis['sound_type'].replace('_', ' ').title()}<br>
                    Confidence: {analysis['confidence']:.2%}<br>
                    <div class="{safety_class}" style="margin-top: 5px; font-size: 0.9em;">
                        {analysis['safety_level']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analyses yet. Upload an audio file to get started!")
        
        # Safety guidelines
        st.subheader("🚨 Safety Guidelines")
        with st.expander("Understanding Safety Levels"):
            st.markdown("""
            - **🔴 HIGH SAFETY CONCERN**: Immediate attention required (gun shots, emergencies)
            - **🟡 TRAFFIC ALERT**: Potential traffic issues (car horns, sirens)
            - **🟠 NOISE POLLUTION**: Environmental concern (construction, drilling)
            - **🟢 NORMAL**: Typical urban sounds (AC, street music, children)
            - **⚪ UNKNOWN**: Unclassified or ambiguous sounds
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**CommunitySoundscape** | UN SDG 11: Sustainable Cities and Communities | "
        "Using AI for Urban Safety Monitoring | "
        "Built with TensorFlow & Streamlit"
    )

if __name__ == "__main__":
    main()
