# 🎬 AI-Powered Emotion-Based Movie Recommendation System

This project integrates **Computer Vision (CV)** and **Natural Language Processing (NLP)** to build an intelligent **movie recommendation system** that adapts to the user’s emotions and sentiments.  

It combines multiple modules:
1. **Facial Emotion Recognition** → Identifies emotions (happy, sad, angry, neutral, etc.) using CNN + OpenCV.
2. **Text Sentiment Analysis** → Analyzes user reviews/comments to detect emotional sentiment (positive, negative, neutral).
3. **IMDb Scraper** → Extracts latest movie data (ratings, genres, popularity) for recommendations.
4. **Hybrid Movie Recommender** → Suggests movies based on facial emotions, text sentiment, and IMDb popularity.

---

## 🚀 Features
- Detects **facial expressions** from live video feed or images.
- Classifies **text input sentiment** using NLP models.
- Scrapes **IMDb** for up-to-date movie details.
- Recommends **personalized movie suggestions** based on combined signals.
- Supports **data visualization** (word clouds, charts, accuracy plots).
- Built modularly with Jupyter notebooks for easy experimentation.

---

## 📂 Project Structure
- `Face_Emotion_Recognition.ipynb` → Trains & tests CNN model for detecting facial emotions using OpenCV + TensorFlow.
- `text_based_sentiment_analysis_and_movie_recomender.ipynb` → Sentiment analysis on text reviews & hybrid recommendations.
- `text_based.ipynb` → Preprocessing text (tokenization, TF-IDF, word embeddings).
- `Imdb_scraper.ipynb` → Scrapes IMDb for movies, ratings, and genres.
- `movie_recommender.ipynb` → Final integration of all modules to recommend movies.

---

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ai-movie-recommender.git
   cd ai-movie-recommender
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage
### 1. Facial Emotion Recognition
```bash
jupyter notebook Face_Emotion_Recognition.ipynb
```
- Uses CNN + OpenCV to classify emotions from images or live feed.

### 2. Text Sentiment Analysis
```bash
jupyter notebook text_based_sentiment_analysis_and_movie_recomender.ipynb
```
- Input a review/comment → outputs sentiment (positive/negative/neutral).

### 3. IMDb Scraper
```bash
jupyter notebook Imdb_scraper.ipynb
```
- Fetches IMDb movie data (ratings, cast, genres).

### 4. Movie Recommender
```bash
jupyter notebook movie_recommender.ipynb
```
- Generates **personalized recommendations** based on detected mood + sentiment.

---

## 📊 Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **Deep Learning**: TensorFlow, Keras
  - **Computer Vision**: OpenCV (cv2)
  - **NLP**: NLTK, TextBlob, scikit-learn
  - **Web Scraping**: BeautifulSoup (bs4), Requests
  - **Data Handling**: NumPy, Pandas, Pickle
  - **Visualization**: Matplotlib, Seaborn, WordCloud
- **Deployment (optional)**: Google Colab / Jupyter Notebook

---

## 🎯 Future Scope
- Deploy as a **Flask/Django web application**.
- Add **voice-based sentiment analysis** (speech-to-text + sentiment).
- Enhance recommender with **collaborative filtering + hybrid ML models**.
- Build a **real-time streaming app** with recommendation dashboard.

---

## 👨‍💻 Contributors
- Developed by: **[Your Name]**
- Guided by: **Mentors / Professors (if any)**

---

## 📜 License
This project is licensed under the **MIT License**.
