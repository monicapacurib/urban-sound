import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Urban Sound Classifier",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_features(audio_file, sample_rate=22050):
    """Extract audio features using librosa"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        
        # Calculate statistics for each feature
        features = []
        for feature in [mfccs, chroma, mel, contrast, tonnetz]:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.median(feature),
                np.min(feature),
                np.max(feature)
            ])
        
        return np.array(features)
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🎵 Urban Sound Classifier</h1>', unsafe_allow_html=True)
    
    st.write("Upload an audio file to classify urban sounds like street music, car horns, children playing, and more!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'm4a'])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Classify Sound", type="primary"):
                with st.spinner("Analyzing audio..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load model and preprocessing objects
                        try:
                            model = load_model('urban_sound_classifier (1).h5')
                            label_encoder = joblib.load('label_encoder (1).pkl')
                            feature_scaler = joblib.load('feature_scaler (1).pkl')
                            class_info = joblib.load('class_info (1).pkl')
                            
                            st.success("✅ Model and preprocessing files loaded successfully!")
                            
                        except FileNotFoundError as e:
                            st.error(f"❌ Model file not found: {e}")
                            st.info("Please make sure these files are in your project directory:")
                            st.write("- urban_sound_classifier (1).h5")
                            st.write("- label_encoder (1).pkl") 
                            st.write("- feature_scaler (1).pkl")
                            st.write("- class_info (1).pkl")
                            return
                        except Exception as e:
                            st.error(f"❌ Error loading model files: {e}")
                            return
                        
                        # Extract features
                        features = extract_features(tmp_path)
                        
                        if features is not None:
                            # Debug information
                            with st.expander("🔍 Debug Information"):
                                st.write(f"Features extracted: {len(features)}")
                                if hasattr(feature_scaler, 'n_features_in_'):
                                    st.write(f"Scaler expects: {feature_scaler.n_features_in_} features")
                                if hasattr(model, 'layers'):
                                    input_shape = model.layers[0].input_shape
                                    st.write(f"Model input shape: {input_shape}")
                            
                            # Check feature dimensions and scale
                            try:
                                features_reshaped = features.reshape(1, -1)
                                
                                # Check feature dimension compatibility
                                if hasattr(feature_scaler, 'n_features_in_'):
                                    if features_reshaped.shape[1] != feature_scaler.n_features_in_:
                                        st.error(f"🚨 Feature dimension mismatch!")
                                        st.write(f"Scaler expects: {feature_scaler.n_features_in_} features")
                                        st.write(f"Extracted features: {features_reshaped.shape[1]} features")
                                        st.info("This usually means your feature extraction doesn't match the training process")
                                        return
                                
                                features_scaled = feature_scaler.transform(features_reshaped)
                                
                            except ValueError as e:
                                st.error(f"❌ Feature scaling error: {e}")
                                return
                            except Exception as e:
                                st.error(f"❌ Error during feature processing: {e}")
                                return
                            
                            # Make prediction
                            try:
                                prediction = model.predict(features_scaled)
                                predicted_class_idx = np.argmax(prediction, axis=1)[0]
                                confidence = np.max(prediction) * 100
                                
                                # Get class label
                                predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
                                
                                # Display results
                                st.markdown("---")
                                st.markdown(f'<div class="prediction-box success-box">', unsafe_allow_html=True)
                                st.subheader("🎯 Prediction Result")
                                st.write(f"**Class:** {predicted_class}")
                                st.write(f"**Confidence:** {confidence:.2f}%")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Display class information if available
                                if class_info and predicted_class in class_info:
                                    info = class_info[predicted_class]
                                    st.info(f"**About {predicted_class}:** {info}")
                                
                                # Show confidence scores for all classes
                                with st.expander("📊 View all confidence scores"):
                                    class_probs = {}
                                    for i, class_label in enumerate(label_encoder.classes_):
                                        class_probs[class_label] = prediction[0][i] * 100
                                    
                                    # Sort by confidence
                                    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))
                                    
                                    for class_label, prob in sorted_probs.items():
                                        progress_color = "green" if class_label == predicted_class else "blue"
                                        st.write(f"**{class_label}:** {prob:.2f}%")
                                        st.progress(float(prob/100))
                                
                            except Exception as e:
                                st.error(f"❌ Prediction error: {e}")
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
        
        with col2:
            st.markdown("### ℹ️ About")
            st.write("This app uses a deep learning model to classify urban sounds into different categories.")
            st.write("Supported formats: WAV, MP3, FLAC, M4A")
            
            # Model information
            with st.expander("Model Info"):
                try:
                    model = load_model('urban_sound_classifier (1).h5')
                    st.write("✅ Model loaded successfully")
                    st.write(f"Layers: {len(model.layers)}")
                    if hasattr(model, 'input_shape'):
                        st.write(f"Input shape: {model.input_shape}")
                except:
                    st.write("Model info not available")

    else:
        st.info("👆 Please upload an audio file to get started")
        
        # Example section
        st.markdown("---")
        st.subheader("🎧 Example Sound Classes")
        st.write("""
        The model can classify sounds like:
        - 🚗 Car horns
        - 👶 Children playing
        - 🐶 Dogs barking
        - 🔨 Drilling
        - 🚂 Engine idling
        - 🔫 Gun shots
        - 🚨 Sirens
        - 🎵 Street music
        - And more...
        """)

if __name__ == "__main__":
    main()
