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

def extract_features_advanced(audio_file, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extract advanced audio features for CNN models"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Ensure audio is long enough
        if len(audio) < sample_rate:  # If less than 1 second
            audio = np.pad(audio, (0, max(0, sample_rate - len(audio))), mode='constant')
        
        # Extract MFCC features with more detail (for CNN input)
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Additional features that might be expected by the model
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        
        # For CNN models, we might need the full feature matrix, not just statistics
        # Try both approaches:
        
        # Option 1: Statistical features (for MLP models)
        statistical_features = []
        feature_arrays = [mfccs, chroma, mel_spectrogram, spectral_contrast, tonnetz]
        
        for feature in feature_arrays:
            statistical_features.extend([
                np.mean(feature),
                np.std(feature),
                np.median(feature),
                np.min(feature),
                np.max(feature)
            ])
        
        statistical_features = np.array(statistical_features)
        
        # Option 2: Full MFCC matrix (for CNN models)
        # Reshape MFCCs to expected input shape
        mfcc_processed = mfccs.T  # Transpose to (time_steps, n_mfcc)
        
        # If we need fixed length, pad or truncate
        expected_timesteps = 44  # Common value for 1-second audio at 22050Hz
        if mfcc_processed.shape[0] < expected_timesteps:
            # Pad with zeros
            pad_width = expected_timesteps - mfcc_processed.shape[0]
            mfcc_processed = np.pad(mfcc_processed, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate
            mfcc_processed = mfcc_processed[:expected_timesteps, :]
        
        cnn_features = mfcc_processed.reshape(1, mfcc_processed.shape[0], mfcc_processed.shape[1])
        
        return statistical_features, cnn_features, mfcc_processed.shape
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None, None, None

def get_model_input_shape(model):
    """Safely get model input shape for different layer types"""
    try:
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            
            # Check different ways to get input shape
            if hasattr(first_layer, 'input_shape') and first_layer.input_shape is not None:
                return first_layer.input_shape[1:]  # Skip batch dimension
            
            if hasattr(first_layer, 'batch_input_shape') and first_layer.batch_input_shape is not None:
                return first_layer.batch_input_shape[1:]  # Skip batch dimension
            
            # For Conv1D layers, check the config
            if hasattr(first_layer, 'get_config'):
                config = first_layer.get_config()
                if 'batch_input_shape' in config:
                    return config['batch_input_shape'][1:]
        
        return None
    except Exception as e:
        st.warning(f"Could not determine model input shape: {e}")
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
                        
                        # Extract features using advanced method
                        statistical_features, cnn_features, feature_shape = extract_features_advanced(tmp_path)
                        
                        if statistical_features is not None and cnn_features is not None:
                            # Debug information
                            with st.expander("🔍 Debug Information"):
                                st.write(f"Statistical features: {len(statistical_features)}")
                                st.write(f"CNN features shape: {cnn_features.shape}")
                                
                                model_input_shape = get_model_input_shape(model)
                                if model_input_shape:
                                    st.write(f"Model expects input shape: {model_input_shape}")
                                
                                if hasattr(feature_scaler, 'n_features_in_'):
                                    st.write(f"Scaler expects: {feature_scaler.n_features_in_} features")
                            
                            # Try different prediction approaches
                            prediction = None
                            confidence = 0
                            predicted_class = "Unknown"
                            
                            # Approach 1: Try CNN features first
                            try:
                                if cnn_features is not None:
                                    # Check if shape matches model expectations
                                    model_input_shape = get_model_input_shape(model)
                                    if model_input_shape and len(model_input_shape) == 2:
                                        # Model expects 2D input (timesteps, features) for Conv1D
                                        if cnn_features.shape[1:] == model_input_shape:
                                            st.info("🔄 Using CNN features for prediction")
                                            prediction = model.predict(cnn_features)
                                        else:
                                            st.warning(f"CNN feature shape {cnn_features.shape[1:]} doesn't match model input {model_input_shape}")
                                    else:
                                        # Try anyway
                                        prediction = model.predict(cnn_features)
                            except Exception as e:
                                st.warning(f"CNN prediction failed: {e}")
                            
                            # Approach 2: Try statistical features with scaler
                            if prediction is None and statistical_features is not None:
                                try:
                                    st.info("🔄 Using statistical features for prediction")
                                    
                                    # Check feature dimension compatibility
                                    if hasattr(feature_scaler, 'n_features_in_'):
                                        if len(statistical_features) != feature_scaler.n_features_in_:
                                            st.error(f"🚨 Feature dimension mismatch!")
                                            st.write(f"Scaler expects: {feature_scaler.n_features_in_} features")
                                            st.write(f"Extracted features: {len(statistical_features)} features")
                                            # Try to adjust features
                                            if len(statistical_features) > feature_scaler.n_features_in_:
                                                statistical_features = statistical_features[:feature_scaler.n_features_in_]
                                                st.warning(f"Truncated features to {len(statistical_features)}")
                                            else:
                                                return
                                    
                                    features_scaled = feature_scaler.transform(statistical_features.reshape(1, -1))
                                    prediction = model.predict(features_scaled)
                                    
                                except Exception as e:
                                    st.error(f"Statistical features prediction failed: {e}")
                                    return
                            
                            # Process prediction results
                            if prediction is not None:
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
                            else:
                                st.error("❌ Could not make prediction with any feature type")
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                        st.info("This might be due to model architecture compatibility issues")
        
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
                    
                    input_shape = get_model_input_shape(model)
                    if input_shape:
                        st.write(f"Input shape: {input_shape}")
                    
                    # Show layer types
                    st.write("First few layers:")
                    for i, layer in enumerate(model.layers[:3]):
                        st.write(f"  {i+1}. {layer.__class__.__name__}")
                        
                except Exception as e:
                    st.write(f"Model info: {e}")

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
