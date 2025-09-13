from newsapi import NewsApiClient
import streamlit as st
import datetime
from transformers import pipeline

# ==========================
# 1. Init Clients
# ==========================
newsapi = NewsApiClient(api_key="673d312a0584433aae4bd396b0d64d36")

# Load FinBERT sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# ==========================
# 2. Fetch News
# ==========================
def get_gold_news():
    articles = newsapi.get_everything(
        q="XAUUSD OR Gold price",
        language="en",
        sort_by="publishedAt",
        page_size=10
    )
    return articles["articles"]

# ==========================
# 3. Analyze News
# ==========================
def analyze_news(news_list):
    results = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for article in news_list:
        text = article.get("title", "") + " " + str(article.get("description", ""))
        sentiment = sentiment_pipeline(text[:512])[0]  # limit to 512 tokens
        label = sentiment["label"].lower()
        score = sentiment["score"]
        results.append((article, label, score))
        if label in counts:
            counts[label] += 1

    # Decide final direction
    if counts["positive"] > counts["negative"]:
        direction = "📈 Bullish"
        color = "green"
    elif counts["negative"] > counts["positive"]:
        direction = "📉 Bearish"
        color = "red"
    else:
        direction = "⚖️ Neutral"
        color = "gray"

    return results, direction, color, counts

# ==========================
# 4. Streamlit UI
# ==========================
st.set_page_config(page_title="XAUUSD News & Analysis", layout="wide")
st.title("📰 Daily Gold (XAUUSD) News & AI Analysis")

today = datetime.date.today().strftime("%B %d, %Y")
st.write(f"Latest News for **{today}**")

# Fetch + Analyze
news = get_gold_news()
analysis, direction, color, counts = analyze_news(news)

# Show summary
st.markdown(f"## Market Sentiment: <span style='color:{color}'>{direction}</span>", unsafe_allow_html=True)
st.write(f"Positive: {counts['positive']} | Negative: {counts['negative']} | Neutral: {counts['neutral']}")

st.write("---")

# Show each news item with sentiment
for article, label, score in analysis:
    st.markdown(f"### [{article['title']}]({article['url']})")
    st.caption(f"🗓️ {article['publishedAt']} | 📰 {article['source']['name']}")
    st.write(article["description"])
    st.markdown(f"**Sentiment:** `{label.upper()}` (score={score:.2f})")
    st.write("---")
