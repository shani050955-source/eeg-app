import streamlit as str_lit
import tensorflow as tf
import numpy as np
import cv2
import os
import mne
from scipy import signal
import matplotlib.pyplot as plt

# Set up beautiful tab configuration for the website
str_lit.set_page_config(page_title="EEG Motor Imagery Control Center", page_icon="🧠", layout="centered")

# Custom CSS styling for a professional medical interface look
str_lit.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #4B5563; text-align: center; margin-bottom: 30px; }
    .result-box { padding: 20px; border-radius: 10px; background-color: #F3F4F6; border-left: 5px solid #2563EB; margin-top: 20px; }
    .prediction-text { font-size: 24px; font-weight: bold; color: #1D4ED8; }
    </style>
""", unsafe_allow_html=True)

# Application Header Elements
str_lit.markdown('<div class="main-title">🧠 BCI Motor Imagery Control Center</div>', unsafe_allow_html=True)
str_lit.markdown(
    '<div class="sub-title">Live EEG Signal Processing and Classification using Deep Learning CNN 2D</div>',
    unsafe_allow_html=True)

MODEL_PATH = 'eeg_finetuned_model.h5'


@str_lit.cache_resource
def load_my_eeg_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


model = load_my_eeg_model()

if model is None:
    str_lit.error(f"Error: Trained model file '{MODEL_PATH}' not found! Please run your CNN code first.")
else:
    str_lit.info("System Status: AI Inference Pipeline is Online and Ready.")

    str_lit.write("### Upload Raw EEG Recording File")
    uploaded_file = str_lit.file_uploader("Choose a raw brain recording session file (.gdf)", type=["gdf"])

    if uploaded_file is not None:
        temp_filename = "temp_uploaded_signal.gdf"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Load raw data and extract events
            raw = mne.io.read_raw_gdf(temp_filename, preload=True, verbose=False)
            raw.filter(8.0, 30.0, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Find trial events (769 = Left, 770 = Right)
            ev_left = event_id.get('769') or event_id.get('0x0301')
            ev_right = event_id.get('770') or event_id.get('0x0302')
            valid_event_ids = [idx for idx in [ev_left, ev_right] if idx is not None]

            # Filter events to include only trials
            trial_events = [ev for ev in events if ev[2] in valid_event_ids]
            total_trials = len(trial_events)

            if total_trials == 0:
                str_lit.error("No valid Motor Imagery trials found in this GDF file.")
                data = None
            else:
                # 1. ADDED: Slider to select which Trial to inspect!
                str_lit.write("### Select Trial to Analyze")
                selected_trial_idx = str_lit.slider(f"Choose Trial (Total found: {total_trials})", 1, total_trials,
                                                    1) - 1

                # Extract the selected trial configuration
                chosen_event = trial_events[selected_trial_idx]
                event_start_sample = chosen_event[0]

                # Extract 4 seconds window starting from second 4 of the trial (skip first 4 seconds cue preparation)
                fs = int(raw.info['sfreq'])
                start_sample = event_start_sample + (4 * fs)
                end_sample = event_start_sample + (8 * fs)

                # Fetch C3, Cz, C4 channels
                data, times = raw[:3, start_sample:end_sample]
                true_label = "LEFT" if chosen_event[2] == ev_left else "RIGHT"

        except Exception as e:
            str_lit.error(f"Error processing EEG file structure: {e}")
            data = None
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        if data is not None:
            with str_lit.spinner("Processing Spectrogram..."):
                spectrograms = []
                for ch in range(3):
                    f, t, Zxx = signal.stft(data[ch, :], fs=fs, nperseg=256, noverlap=1)
                    mag = np.abs(Zxx)
                    freq_mask = (f >= 8) & (f <= 30)
                    mag_filtered = mag[freq_mask, :]

                    mag_min, mag_max = mag_filtered.min(), mag_filtered.max()
                    if mag_max > mag_min:
                        norm_img = ((mag_filtered - mag_min) / (mag_max - mag_min) * 255).astype(np.uint8)
                    else:
                        norm_img = np.zeros(mag_filtered.shape, dtype=np.uint8)

                    norm_resized = cv2.resize(norm_img, (64, 21), interpolation=cv2.INTER_AREA)
                    color_slice = cv2.applyColorMap(norm_resized, cv2.COLORMAP_JET)
                    spectrograms.append(color_slice)

                spectrogram_image = np.vstack(spectrograms)
                spectrogram_image = cv2.resize(spectrogram_image, (64, 64), interpolation=cv2.INTER_AREA)

            str_lit.write(f"#### Generated Spectrogram for Trial #{selected_trial_idx + 1}:")
            rgb_render = cv2.cvtColor(spectrogram_image, cv2.COLOR_BGR2RGB)
            str_lit.image(rgb_render, caption=f"Brain Pattern Map (Ground Truth/Real Label: {true_label})", width=350)

            # AI Inference
            input_data = np.expand_dims(spectrogram_image.astype('float32') / 255.0, axis=0)
            prediction_prob = model.predict(input_data, verbose=0)

            # NEW FIX: Flatten the prediction to get a clean scalar value
            prob_value = float(np.squeeze(prediction_prob))

            if prob_value <= 0.5:
                predicted_class = "LEFT HAND MOVEMENT"
                confidence = (1.0 - prob_value) * 100.0
            else:
                predicted_class = "RIGHT HAND MOVEMENT"
                confidence = prob_value * 100.0

            str_lit.markdown('<div class="result-box">', unsafe_allow_html=True)
            str_lit.markdown(f"**True Label (Ground Truth):** {true_label}", unsafe_allow_html=True)
            str_lit.markdown(f"**AI Classification Decision:** <span class='prediction-text'>{predicted_class}</span>",
                             unsafe_allow_html=True)
            str_lit.markdown(f"**AI Model Confidence Level:** {confidence:.2f}%")
            str_lit.markdown('</div>', unsafe_allow_html=True)


