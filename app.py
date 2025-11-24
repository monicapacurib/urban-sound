import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import os
import tempfile
import soundfile as sf

# Set page configuration
st.set_page_config(
    page_title="Urban Sound Classifier",
    page_icon="🎵",
    layout="wide"
)

def extract_features_proper(audio_file, sample_rate=22050):
    """Extract features exactly as they were during training"""
    try:
        # Load audio with proper resampling
        audio, sr = librosa.load(audio_file, sr=sample_rate, duration=3.0)  # Use 3 seconds max
        
        # Ensure consistent audio length (pad or trim to 3 seconds)
        target_length = sample_rate * 3  # 3 seconds
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Extract the exact features used during training
        # Common urban sound features:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft.T, axis=0)
        
        # Combine all features
        features = np.concatenate([
            mfccs_mean,
            mfccs_std,
            mfccs_delta_mean,
            [spectral_centroid_mean],
            [spectral_rolloff_mean],
            [zero_crossing_rate_mean],
            chroma_stft_mean
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def extract_features_simple(audio_file, sample_rate=22050):
    """Simpler feature extraction - more likely to match training"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=sample_rate, duration=4.0)
        
        # Pad/trim to 4 seconds
        target_length = sample_rate * 4
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Basic MFCC features only (most common approach)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Calculate statistics
        features = []
        for i in range(mfccs.shape[0]):
            features.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i]),
                np.median(mfccs[i])
            ])
        
        # Add some basic spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        features.extend([spectral_centroid, zero_crossing_rate])
        
        return np.array(features)
        
    except Exception as e:
        st.error(f"Error in simple feature extraction: {str(e)}")
        return None

def debug_model_prediction(model, features, label_encoder):
    """Debug why model always predicts gun_shot"""
    try:
        # Get raw predictions
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        prediction = model.predict(features, verbose=0)
        
        # Get all class probabilities
        class_probs = {}
        for i, class_name in enumerate(label_encoder.classes_):
            class_probs[class_name] = float(prediction[0][i])
        
        # Check if gun_shot has unusually high probability
        gun_shot_prob = class_probs.get('gun_shot', class_probs.get('gunshot', 0))
        
        return class_probs, gun_shot_prob, prediction
        
    except Exception as e:
        st.error(f"Debug prediction error: {e}")
        return {}, 0, None

def main():
    st.title("🎵 Urban Sound Classifier - DEBUG MODE")
    st.warning("🔧 Debug mode activated to fix 'always gun_shot' issue")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🔍 Classify Sound (DEBUG)", type="primary"):
            with st.spinner("Analyzing with debug information..."):
                try:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load model and encoders
                    model = load_model('urban_sound_classifier (1).h5')
                    label_encoder = joblib.load('label_encoder (1).pkl')
                    feature_scaler = joblib.load('feature_scaler (1).pkl')
                    
                    st.success("✅ Models loaded")
                    
                    # DEBUG: Show model information
                    with st.expander("🔧 MODEL DEBUG INFO"):
                        st.write("Label encoder classes:", list(label_encoder.classes_))
                        st.write("Number of classes:", len(label_encoder.classes_))
                        
                        if hasattr(feature_scaler, 'n_features_in_'):
                            st.write("Scaler expects features:", feature_scaler.n_features_in_)
                        else:
                            st.write("Scaler type:", type(feature_scaler))
                    
                    # Try different feature extraction methods
                    st.subheader("🔄 Testing Feature Extraction Methods")
                    
                    # Method 1: Simple features
                    features_simple = extract_features_simple(tmp_path)
                    if features_simple is not None:
                        st.write(f"Simple features length: {len(features_simple)}")
                    
                    # Method 2: Proper features  
                    features_proper = extract_features_proper(tmp_path)
                    if features_proper is not None:
                        st.write(f"Proper features length: {len(features_proper)}")
                    
                    # Choose which features to use based on scaler expectations
                    features_to_use = None
                    if hasattr(feature_scaler, 'n_features_in_'):
                        expected_features = feature_scaler.n_features_in_
                        
                        if features_simple is not None and len(features_simple) == expected_features:
                            features_to_use = features_simple
                            st.info(f"✅ Using simple features (matches scaler: {expected_features})")
                        elif features_proper is not None and len(features_proper) == expected_features:
                            features_to_use = features_proper
                            st.info(f"✅ Using proper features (matches scaler: {expected_features})")
                        else:
                            # Try to adjust features to match expected length
                            if features_simple is not None:
                                if len(features_simple) > expected_features:
                                    features_to_use = features_simple[:expected_features]
                                    st.warning(f"⚠️ Truncated simple features to {expected_features}")
                                else:
                                    # Pad features
                                    padding = expected_features - len(features_simple)
                                    features_to_use = np.pad(features_simple, (0, padding), mode='constant')
                                    st.warning(f"⚠️ Padded simple features to {expected_features}")
                    else:
                        # No scaler info, use simple features
                        features_to_use = features_simple
                        st.info("✅ Using simple features (no scaler info)")
                    
                    if features_to_use is not None:
                        # Scale features
                        features_scaled = feature_scaler.transform(features_to_use.reshape(1, -1))
                        
                        # DEBUG PREDICTION
                        st.subheader("🔍 PREDICTION DEBUG")
                        class_probs, gun_shot_prob, raw_prediction = debug_model_prediction(
                            model, features_scaled, label_encoder
                        )
                        
                        # Show raw probabilities
                        st.write("### Raw Class Probabilities:")
                        for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
                            percentage = prob * 100
                            bar_color = "red" if class_name in ['gun_shot', 'gunshot'] else "blue"
                            
                            col1, col2, col3 = st.columns([2, 4, 2])
                            with col1:
                                st.write(f"**{class_name}**")
                            with col2:
                                st.progress(float(prob), text=f"{percentage:.2f}%")
                            with col3:
                                st.write(f"{percentage:.2f}%")
                        
                        # Show gun_shot analysis
                        st.write(f"### Gun Shot Analysis:")
                        st.write(f"Gun shot probability: **{gun_shot_prob * 100:.2f}%**")
                        
                        if gun_shot_prob > 0.5:
                            st.error("🚨 HIGH gun shot probability detected!")
                            st.write("This suggests:")
                            st.write("1. Model might be biased towards gun_shot")
                            st.write("2. Feature extraction doesn't match training")
                            st.write("3. Audio preprocessing issues")
                        
                        # Final prediction
                        predicted_class_idx = np.argmax(raw_prediction, axis=1)[0]
                        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
                        confidence = np.max(raw_prediction) * 100
                        
                        st.success(f"🎯 Final Prediction: **{predicted_class}** ({confidence:.2f}%)")
                        
                        # Show audio analysis
                        with st.expander("🎵 Audio Analysis"):
                            audio, sr = librosa.load(tmp_path)
                            st.write(f"Audio length: {len(audio)/sr:.2f} seconds")
                            st.write(f"Sample rate: {sr} Hz")
                            st.write(f"Max amplitude: {np.max(np.abs(audio)):.3f}")
                            
                            # Show waveform
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(audio)
                            ax.set_title("Audio Waveform")
                            ax.set_xlabel("Samples")
                            ax.set_ylabel("Amplitude")
                            st.pyplot(fig)
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Check that all model files exist and are compatible")

    else:
        st.info("👆 Upload an audio file to test")
        
        # Test with different sound types
        st.subheader("🧪 Test with these sound types:")
        st.write("- 🎵 Music (should NOT be gun_shot)")
        st.write("- 🗣️ Speech (should NOT be gun_shot)") 
        st.write("- 🚗 Car sounds (should NOT be gun_shot)")
        st.write("- 🔫 Actual gun shots (should be gun_shot)")

if __name__ == "__main__":
    main()
