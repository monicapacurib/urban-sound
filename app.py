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
        model = load_model('urban_sound_classifier (1).h5')
        label_encoder = joblib.load('label_encoder (1).pkl')
        feature_scaler = joblib.load('feature_scaler (1).pkl')
        class_info = joblib.load('class_info (1).pkl')
        return model, label_encoder, feature_scaler, class_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def extract_features_for_prediction(audio_path, n_mfcc=20):
    """Extract features from audio file for prediction"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Ensure consistent length (4 seconds)
        target_length = 22050 * 4
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, max(0, target_length - len(audio))))
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Return mean of MFCCs across time
        features = np.mean(mfccs.T, axis=0)
        
        # Ensure we have exactly 20 features
        if len(features) != 20:
            if len(features) < 20:
                features = np.pad(features, (0, 20 - len(features)))
            else:
                features = features[:20]
        
        return features
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def create_audio_visualizations(audio_path):
    """Create audio visualizations"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
        axes[0, 0].set_title('Audio Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], format='%+2.0f dB')
        
        # Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title('Mel Spectrogram')
        plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], format='%+2.0f dB')
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        librosa.display.specshow(mfccs, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title('MFCC Coefficients')
        plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None

def get_safety_assessment(class_name, safety_categories):
    """Get safety level and CSS class for a sound class"""
    default_safety_mapping = {
        'air_conditioner': ('NORMAL', 'safety-low'),
        'car_horn': ('TRAFFIC ALERT', 'safety-medium'),
        'children_playing': ('NORMAL', 'safety-low'),
        'dog_bark': ('NORMAL', 'safety-low'),
        'drilling': ('NOISE POLLUTION', 'safety-medium'),
        'engine_idling': ('TRAFFIC NOISE', 'safety-low'),
        'gun_shot': ('HIGH SAFETY CONCERN', 'safety-high'),
        'jackhammer': ('NOISE POLLUTION', 'safety-medium'),
        'siren': ('EMERGENCY VEHICLE', 'safety-high'),
        'street_music': ('NORMAL', 'safety-low')
    }
    
    if safety_categories and class_name in safety_categories:
        safety_info = safety_categories[class_name]
        if isinstance(safety_info, tuple):
            return safety_info
        else:
            if safety_info in ['HIGH SAFETY CONCERN', 'EMERGENCY VEHICLE']:
                return safety_info, 'safety-high'
            elif safety_info in ['TRAFFIC ALERT', 'NOISE POLLUTION']:
                return safety_info, 'safety-medium'
            else:
                return safety_info, 'safety-low'
    else:
        return default_safety_mapping.get(class_name, ('UNKNOWN', 'safety-unknown'))

def main():
    # Header
    st.markdown('<h1 class="main-header">🏙️ CommunitySoundscape</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Urban Sound Monitor for Public Safety")
    
    # Load models
    model, label_encoder, feature_scaler, class_info = load_ml_models()
    if model is None:
        st.error("Failed to load ML models. Please check if all model files are uploaded.")
        return
    
    # Sidebar
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 0.9, 0.6, 0.1
    )
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Audio for Analysis")
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'm4a'],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            # Display audio
            st.audio(uploaded_file.getvalue())
            
            if st.button("Analyze Sound", type="primary"):
                with st.spinner("Analyzing..."):
                    features = extract_features_for_prediction(audio_path)
                    
                    if features is not None:
                        # Scale and reshape features
                        features_scaled = feature_scaler.transform(features.reshape(1, -1))
                        features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)
                        
                        # Predict
                        prediction = model.predict(features_reshaped, verbose=0)
                        confidence = np.max(prediction)
                        class_index = np.argmax(prediction)
                        class_name = label_encoder.classes_[class_index]
                        
                        # Safety assessment
                        safety_categories_dict = class_info.get('safety_categories', {}) if class_info else {}
                        safety_level, safety_class = get_safety_assessment(class_name, safety_categories_dict)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Detected Sound", class_name.replace('_', ' ').title())
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        with col3:
                            st.markdown(f'<div class="{safety_class}">{safety_level}</div>', unsafe_allow_html=True)
                        
                        if confidence < confidence_threshold:
                            st.warning(f"Low confidence (below {confidence_threshold:.0%})")
                        
                        # Visualizations
                        st.subheader("Audio Analysis")
                        fig = create_audio_visualizations(audio_path)
                        if fig:
                            st.pyplot(fig)
                        
                        # Log history
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.analysis_history.append({
                            'timestamp': timestamp,
                            'sound_type': class_name,
                            'confidence': confidence,
                            'safety_level': safety_level,
                            'filename': uploaded_file.name
                        })
            
            # Cleanup
            try:
                os.unlink(audio_path)
            except:
                pass
    
    with col2:
        st.subheader("Urban Sound Dashboard")
        
        # Show recent analyses
        st.subheader("Recent Analyses")
        if st.session_state.analysis_history:
            for analysis in reversed(st.session_state.analysis_history[-3:]):
                safety_class_map = {
                    'HIGH SAFETY CONCERN': 'safety-high',
                    'EMERGENCY VEHICLE': 'safety-high',
                    'TRAFFIC ALERT': 'safety-medium',
                    'NOISE POLLUTION': 'safety-medium',
                    'NORMAL': 'safety-low'
                }
                safety_class = safety_class_map.get(analysis['safety_level'], 'safety-unknown')
                
                st.markdown(f"""
                <div style="border-left: 4px solid #1f77b4; padding: 10px; margin: 5px 0;">
                    <strong>{analysis['timestamp']}</strong><br>
                    {analysis['sound_type'].replace('_', ' ').title()} ({analysis['confidence']:.2%})<br>
                    <div class="{safety_class}">{analysis['safety_level']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analyses yet")
        
        # Safety info
        st.subheader("Safety Guidelines")
        st.markdown("""
        - 🔴 **HIGH SAFETY**: Gun shots, emergencies
        - 🟡 **TRAFFIC ALERT**: Car horns, sirens  
        - 🟠 **NOISE POLLUTION**: Construction, drilling
        - 🟢 **NORMAL**: AC, music, children playing
        """)

if __name__ == "__main__":
    main()
