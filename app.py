import streamlit as st
import pandas as pd
import re
import string
import zipfile
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

@st.cache_data
def load_dataset(zip_path="archive.zip"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        zip_ref.extract(csv_filename)
        col_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(csv_filename, encoding='latin1', header=None, names=col_names)
    df['sentiment'] = df['sentiment'].replace(4, 1)
    df = df[df['sentiment'].isin([0, 1])]
    return df[['sentiment', 'text']]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s" + string.punctuation + "]+", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def train_model(df):
    df['clean_text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=10000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, vectorizer

def predict_sentiment(tweet, model, vectorizer):
    cleaned = clean_text(tweet)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect).max()
    label = "Positive 😊" if pred == 1 else "Negative 😠"
    return label, round(prob * 100, 2)

# 🚀 Streamlit UI
st.title("Tweet Sentiment Classifier")
st.write("Predict whether a tweet is positive or negative!")

# File uploader
uploaded = st.file_uploader("Upload archive.zip", type="zip")
if uploaded:
    with open("archive.zip", "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Training model..."):
        df = load_dataset("archive.zip")
        model, vectorizer = train_model(df)

    tweet = st.text_area("Enter a tweet:")
    if tweet and st.button("Analyze"):
        label, confidence = predict_sentiment(tweet, model, vectorizer)
        st.success(f"**Sentiment:** {label} ({confidence}% confidence)")
