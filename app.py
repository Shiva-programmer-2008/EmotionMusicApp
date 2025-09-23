# app.py
import streamlit as st
import pandas as pd
import random
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Emotion_Music_Dataset_with_Links.csv")

# -----------------------------
# Hugging Face Text Emotion Analyzer
# -----------------------------
text_emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .title {font-size: 48px; color: #FF4B4B; font-weight: bold; text-align: center; margin-bottom:10px;}
    .subtitle {font-size: 22px; color: #3B3B98; text-align: center; margin-bottom:20px;}
    .playlist {background-color: #f0f0f5; padding: 15px; border-radius: 10px; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# App Title
# -----------------------------
st.markdown('<div class="title">🎵 FeelGroove</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect emotion via Text, Face Image, or MCQ and get your playlist!</div>', unsafe_allow_html=True)

# -----------------------------
# Language & Input Method Selection
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    language = st.radio("Select song language:", df['Language'].unique())
with col2:
    input_method = st.radio("Select input method:", ["Text", "Upload Image", "MCQ"])

detected_emotion = None

# -----------------------------
# 1. Text-based Emotion
# -----------------------------
if input_method == "Text":
    user_text = st.text_area("Type how you feel:")
    if user_text:
        st.write("Detecting your emotion…")
        results = text_emotion_analyzer(user_text)
        emotion_scores = {r['label']: r['score'] for r in results[0]}
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[detected_emotion]
        st.write(f"Detected Emotion: **{detected_emotion}** (Confidence: {confidence:.2f})")

# -----------------------------
# 2. Face-based Emotion Detection (Simulated)
# -----------------------------
elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload your face image:", type=["jpg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        frame = np.array(img.convert("RGB"))

        # Load pre-trained OpenCV face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Simulated emotion detection using dataset unique emotions
            possible_emotions = df['Emotion'].unique().tolist()
            detected_emotion = random.choice(possible_emotions)
            st.write(f"Detected Emotion from face (simulated): **{detected_emotion}**")
        else:
            st.write("No face detected. Try another image.")

# -----------------------------
# 3. MCQ-based Emotion Detection
# -----------------------------
elif input_method == "MCQ":
    st.subheader("Answer the following questions:")
    questions = [
        {"question": "How energetic do you feel?", 
         "options": {"Very low 😴": "Sad/Chill", "Low 😌": "Chill", "Medium 🙂": "Chill/Energetic", "High 😎": "Energetic"}},
        {"question": "How happy/positive do you feel?", 
         "options": {"Very low 😔": "Sad/Chill", "Medium 🙂": "Chill", "High 😄": "Happy/Energetic"}},
        {"question": "How romantic or loving are you feeling?", 
         "options": {"Not at all 😐": None, "Somewhat 🙂": "Romantic", "Very 😍": "Romantic"}},
        {"question": "Do you want relaxing or upbeat music?", 
         "options": {"Relaxing 😌": "Chill/Relax", "Upbeat 😎": "Energetic"}}
    ]
    selected_emotions = []
    for q in questions:
        ans = st.radio(q["question"], list(q["options"].keys()), key=q["question"])
        mapped_emotion = q["options"][ans]
        if mapped_emotion:
            selected_emotions.append(mapped_emotion)
    if selected_emotions:
        detected_emotion = max(set(selected_emotions), key=selected_emotions.count)
        st.write(f"Detected Dominant Emotion: **{detected_emotion}**")

# -----------------------------
# Emotion Mapping for Dataset
# -----------------------------
emotion_mapping = {
    "joy": "Happy/Energetic",
    "sadness": "Sad/Chill",
    "anger": "Energetic",
    "fear": "Chill/Relax",
    "love": "Romantic",
    "surprise": "Energetic",
    "neutral": "Chill"
}

if detected_emotion:
    mapped_emotion = emotion_mapping.get(detected_emotion.lower(), detected_emotion)
    st.markdown(f"**Mapped Emotion for Dataset:** {mapped_emotion}")

    # -----------------------------
    # Playlist Selection
    # -----------------------------
    filtered_songs = df[(df['Emotion'].str.contains(mapped_emotion, case=False)) & 
                        (df['Language'].str.contains(language, case=False))]

    if not filtered_songs.empty:
        playlist_size = min(5, len(filtered_songs))
        playlist = filtered_songs.sample(playlist_size)
        st.markdown(f"🎶 Playlist ({playlist_size} songs) for **{mapped_emotion}** in **{language}**:")
        for idx, song in playlist.iterrows():
            st.markdown(f"""
            <div class="playlist">
                <h3>🎵 {song['Song Name']}</h3>
                <p>By: {song['Artist(s)']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.video(song['Link'])
        st.balloons()
    else:
        st.write("No songs found for this emotion & language combination.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px;'>FeelGroove 🎵 - Emotion-driven playlist generator | Developed by YourName</p>
""", unsafe_allow_html=True)






