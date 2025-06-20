import os
import numpy as np
import librosa
import streamlit as st
from pydub import AudioSegment
import tensorflow as tf

# Constants
SR = 16000
DURATION = 15
N_MELS = 32
N_FFT = 1024
HOP_LENGTH = 1024
MAX_PAD_LEN = int(np.ceil(DURATION * SR / HOP_LENGTH))
LABELS = ["Akan", "Dagbani", "Ikposo"]
MODEL_PATH = 'multiclass_model_aug_3_lang.keras'

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def extract_features(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mp3":
            tmp = "tmp.wav"
            audio = AudioSegment.from_file(file_path).set_frame_rate(SR).set_channels(1)[:DURATION * 1000]
            audio.export(tmp, format="wav")
            file_path = tmp

        y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        y = np.pad(y, (0, max(0, DURATION * SR - len(y))))[:DURATION * SR]

        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = np.pad(log_mel, ((0, 0), (0, max(0, MAX_PAD_LEN - log_mel.shape[1]))), mode='constant')
        log_mel = log_mel[:, :MAX_PAD_LEN]
        return log_mel.T
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_language(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, None
    input_tensor = np.expand_dims(features, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    pred_idx = np.argmax(prediction)
    return LABELS[pred_idx], float(prediction[0][pred_idx]) * 100

# Streamlit App UI
st.title("🗣️ Multi-Language Audio Classifier")
uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path)
    with st.spinner("Analyzing..."):
        label, confidence = predict_language(file_path)

    if label:
        st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")
    else:
        st.error("Prediction failed. Try again.")
