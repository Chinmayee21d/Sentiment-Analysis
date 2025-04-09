import streamlit as st
import pickle
import re
import nltk
import tweepy
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load Twitter API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Authenticate with Twitter API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(username, count=5):
    """Fetch recent tweets from a Twitter user."""
    try:
        user = client.get_user(username=username, user_auth=False)
        user_id = user.data.id
        tweets = client.get_users_tweets(id=user_id, max_results=count)
        return [tweet.text for tweet in tweets.data] if tweets.data else []
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return []

# Download stopwords once
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Function to preprocess text
def preprocess_text(text, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    processed_text = preprocess_text(text, stop_words)
    transformed_text = vectorizer.transform([processed_text])
    sentiment = model.predict(transformed_text)[0]
    sentiment_map = {0: "Negative", 1: "Positive"}  # Neutral removed
    return sentiment_map.get(sentiment, "Unknown")

# Function to create colored sentiment cards
def create_card(text, sentiment):
    color_map = {"Positive": "green", "Negative": "red"}
    color = color_map.get(sentiment, "gray")
    return f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{text}</p>
    </div>
    """

# Main app logic
def main():
    st.title("Sentiment Analysis")
    st.markdown("Analyze sentiment from text, or CSV files effortlessly! 🚀")
    
    # Dropdown for input method
    option = st.selectbox("🎯 Choose an option", ["Input text", "Get tweets from user", "Upload CSV"])
    
    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    
    if option == "Input text":
        user_text = st.text_area("Enter text:")
        if st.button("Analyze Text"):
            if user_text:
                sentiment = predict_sentiment(user_text, model, vectorizer, stop_words)
                st.markdown(create_card(user_text, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")
    
    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter Username:")
        if st.button("Analyze Tweets"):
            if username:
                tweets_data = fetch_tweets(username, count=5)
                if not tweets_data:
                    st.warning("No tweets found for this user.")
                else:
                    for tweet_text in tweets_data:
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
            else:
                st.warning("Please enter a valid Twitter username.")
    
    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "text" in df.columns:
                df["sentiment"] = df["text"].apply(lambda x: predict_sentiment(x, model, vectorizer, stop_words))
                st.write(df)
            else:
                st.error("CSV file must contain a 'text' column.")

if __name__ == "__main__":
    main()