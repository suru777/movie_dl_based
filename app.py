import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
from imdb import IMDb
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import io

# Load all necessary files
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sa_text.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_resource
def load_max_length():
    with open('max_length.pickle', 'rb') as handle:
        return pickle.load(handle)

@st.cache_data
def load_dataset():
    with open('dataset.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_length = load_max_length()
df_filtered1 = load_dataset()

label_mapping = {0: "😢 Sadness", 1: "😂 Joy", 2: "❤ Love", 3: "😡 Anger", 4: "😨 Fear", 5: "😲 Surprise"}
mood_mapping = {"😂 Joy": 0, "❤ Love": 0, "😢 Sadness": 1, "😨 Fear": 1, "😡 Anger": 1, "😲 Surprise": 2}

def normalize(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = {"i", "am", "the", "is", "and", "a", "to", "of", "in", "for", "on", "with", "as", "by", "at", "an"}
    word_tokens = text.split()
    return ' '.join(word for word in word_tokens if word not in stop_words)

def predict_emotion(text):
    normalized_input = normalize(text)
    input_sequence = tokenizer.texts_to_sequences([normalized_input])
    input_padded = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(input_padded)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return predicted_label, label_mapping[predicted_label]

def recommend_movies(cluster_value):
    filtered_movies = df_filtered1[df_filtered1['cluster'] == cluster_value]
    top_movies = filtered_movies.head(16)  # Increased to 16
    return top_movies[['title']]

def get_movie_info(title):
    try:
        ia = IMDb()
        movies = ia.search_movie(title)
        if not movies:
            return {'title': title, 'poster': None, 'description': 'N/A', 'year': 'N/A', 'rating': 'N/A', 'imdb_link': 'N/A'}

        movie = movies[0]
        ia.update(movie)

        query = title + ' poster'
        url = 'https://www.google.com/search?q=' + query + '&tbm=isch'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        images = soup.findAll('img')
        poster_url = images[1].get('src') if len(images) > 1 else None

        return {
            'title': title,
            'poster': poster_url,
            'description': movie.get('plot outline', 'N/A'),
            'year': movie.get('year', 'N/A'),
            'rating': movie.get('rating', 'N/A'),
            'imdb_link': f"https://www.imdb.com/title/tt{movie.movieID}/"
        }
    except:
        return {'title': title, 'poster': None, 'description': 'N/A', 'year': 'N/A', 'rating': 'N/A', 'imdb_link': 'N/A'}

# Streamlit UI
st.set_page_config(page_title="🎥 Sentiment Movie Recommender", layout="wide")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<script>
document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('read-more')) {
        const info = e.target.previousElementSibling;
        if (info.classList.contains('show-all')) {
            info.classList.remove('show-all');
            e.target.textContent = 'See More';
        } else {
            info.classList.add('show-all');
            e.target.textContent = 'Show Less';
        }
    }
});
</script>
""", unsafe_allow_html=True)

st.title("🎭 Sentiment-Based Movie Recommender")
st.markdown("Enter a sentence describing your mood or feelings, or upload your voice!")

input_mode = st.radio("Choose input mode:", ('Text', 'Audio'))

user_input = ""

if input_mode == 'Text':
    user_input = st.text_input("Your text:", "I am feeling happy today!")
else:
    audio_file = st.file_uploader("Upload an audio file (MP3, WAV, or M4A format)", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        recognizer = sr.Recognizer()
        try:
            audio_bytes = audio_file.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
                audio.export(tmp_wav_file.name, format="wav")
                with sr.AudioFile(tmp_wav_file.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        user_input = recognizer.recognize_google(audio_data)
                        st.success(f"Recognized Text: {user_input}")
                    except sr.UnknownValueError:
                        st.error("Sorry, could not understand the audio.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")

if st.button("Predict & Recommend") and user_input.strip() != "":
    with st.spinner('Analyzing your mood...'):
        predicted_label, predicted_emotion = predict_emotion(user_input)
        mood = mood_mapping.get(predicted_emotion, 2)

    st.success(f"*Detected Emotion:* {predicted_emotion}")

    with st.spinner('Finding great movie recommendations...'):
        recommended_movies = recommend_movies(mood)

    st.subheader("🎬 Recommended Movies for You")

    movie_data = []
    for title in recommended_movies['title']:
        movie_info = get_movie_info(title)
        movie_data.append(movie_info)

    cols = st.columns(4)
    for idx, movie in enumerate(movie_data):
        col = cols[idx % 4]
        with col:
            fallback_poster = 'https://via.placeholder.com/150x250?text=No+Image'
            poster = movie['poster'] if movie['poster'] else fallback_poster
            short_description = movie['description'][:150] + "..." if len(movie['description']) > 150 else movie['description']

            st.markdown(f"""
            <div class='movie-card'>
                <div class='card-inner'>
                    <div class='card-front'>
                        <a href='{movie['imdb_link']}' target='_blank'>
                            <img src='{poster}' class='movie-poster' />
                        </a>
                        <div class='movie-title'>{idx+1}. {movie['title']}</div>
                    </div>
                    <div class='card-back'>
                        <div class='movie-info'>⭐ {movie['rating']} | {movie['year']}</div>
                        <div class='movie-info' data-read-more>{short_description}</div>
                        {"<span class='read-more'>See More</span>" if len(movie['description']) > 150 else ""}
                        <div style='margin-top:8px;'><a href='{movie['imdb_link']}' target='_blank' style='color: #ffcc00;'>IMDb Page 🔗</a></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")