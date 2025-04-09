import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# Download stopwords once using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)
    
    # Predict sentiment
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"

# Initialize Nitter scraper
@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# Function to create a colored card for each tweet
def create_card(tweet_text, sentiment):
    color = "#28a745" if sentiment == "Positive" else "#dc3545"
    card_html = f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 8px; margin: 10px 0; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Main app logic
def main():
    # Custom CSS for UI
    st.markdown("""
        <style>
            body {
                background-color: #f8f9fa;
            }
            .css-1d391kg {
                background-color: #1e1e1e !important;
                color: #fff;
            }
            .stButton > button {
                background-color: #28a745;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }
            .stButton > button:hover {
                background-color: #218838;
                color: white !important;
                transform: translateY(-2px);
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }
            .stButton > button:active {
                background-color: #1e7e34;
                transform: translateY(1px);
                box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            }
            .stFileUploader {
                border: 2px dashed #28a745;
                padding: 10px;
                border-radius: 10px;
                margin-top: 10px;
            }
            h1 {
                color: #1e7e34;
                text-align: center;
                font-size: 2.5rem;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # App Title
    st.markdown("<h1>📊 Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.write("Analyze sentiment from text, or CSV files effortlessly! 🚀")

    # Load stopwords, model, vectorizer, and scraper only once
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    # User input: Select an option from dropdown
    option = st.selectbox(
        "🎯 Choose an option",
        ["Input text","Get tweets from user","Upload CSV"],
        index=0
    )

    # Option 1: Input text
    if option == "Input text":
        text_input = st.text_area("📝 Enter text to analyze sentiment", height=150)
        if st.button("Analyze Sentiment"):
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"✅ Sentiment: **{sentiment}**")

    # Option 2: Get tweets from a Twitter user
    elif option == "Get tweets from user":
        username = st.text_input("🐦 Enter Twitter username")
        if st.button("Fetch Tweets"):
            if not username.strip():
                st.warning("⚠️ Please enter a valid Twitter username.")
            elif not re.match(r'^[A-Za-z0-9_]{1,15}$', username):
                st.error("❌ Invalid Twitter username. Usernames can contain letters, numbers, and underscores (1-15 characters).")
            else:
                try:
                    tweets_data = scraper.get_tweets(username, mode='user', number=5)

                    st.write("📡 Debug - Raw Tweets Data:", tweets_data)

                    # Check if valid and contains tweets
                    if not tweets_data or not isinstance(tweets_data, list) or len(tweets_data) == 0:
                        st.warning(f"⚠️ No tweets found for '{username}'. Please check the username or try again later.")
                        return

                    # Loop through and process tweets
                    for tweet in tweets_data:
                        if 'text' in tweet:
                            tweet_text = tweet['text']
                            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                            card_html = create_card(tweet_text, sentiment)
                            st.markdown(card_html, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Skipping a tweet with no text.")
                except Exception as e:
                    st.error(f"❌ Error fetching tweets: {e}")


    # Option 3: Upload and analyze CSV
    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'tweets' column", type=["csv"])
        if uploaded_file is not None:
            try:
                # Validate file size (limit to 10MB)
                if uploaded_file.size > 10 * 1024 * 1024:
                    st.error("⚠️ File size exceeds 10 MB. Please upload a smaller file.")
                else:
                    df = pd.read_csv(uploaded_file)

                    # Validate 'tweets' column existence
                    if "tweets" not in df.columns:
                        st.error("⚠️ CSV must contain a 'tweets' column!")
                    else:
                        # Analyze sentiment for all rows in CSV
                        df["sentiment"] = df["tweets"].apply(lambda x: predict_sentiment(x, model, vectorizer, stop_words))
                        st.success("✅ Sentiment analysis completed successfully!")
                        st.write("### 📊 Analyzed Results")
                        st.dataframe(df)

                        # Download button for the analyzed CSV
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()
