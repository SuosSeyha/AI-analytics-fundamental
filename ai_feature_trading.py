import pandas as pd
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# ==========================
# 1. Load Data
# ==========================
@st.cache_data
def load_data():
    df = yf.download("XAUUSD=X", period="1mo", interval="1h")
    df.dropna(inplace=True)
    return df

df = load_data()

# ==========================
# 2. Feature Engineering
# ==========================
df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["ema20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
df["macd"] = ta.trend.MACD(df["Close"]).macd()
df.dropna(inplace=True)

# Target: Up(1) if next close > current close, else Down(0)
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Features & Labels
X = df[["rsi", "ema20", "ema50", "macd"]]
y = df["target"]

# ==========================
# 3. Train Model
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# ==========================
# 4. Streamlit UI
# ==========================
st.set_page_config(page_title="XAUUSD AI Trading", layout="wide")

st.title("📈 XAUUSD AI Trading System")
st.write("AI-powered trading signals combining **technical indicators** and ML.")

# Show data
st.subheader("Recent Market Data")
st.dataframe(df.tail(10))

# Plot price chart
st.subheader("Gold Price Chart")
st.line_chart(df["Close"])

# Show indicators
st.subheader("Technical Indicators")
st.line_chart(df[["ema20", "ema50"]])

# Show model performance
st.subheader("Model Performance")
st.write(f"✅ Model Accuracy: **{acc:.2f}**")

# Make prediction on latest data
latest = X.iloc[-1:].values
prediction = model.predict(latest)[0]
direction = "⬆️ BUY (Bullish)" if prediction == 1 else "⬇️ SELL (Bearish)"

st.subheader("AI Trading Signal")
st.write(f"Latest Prediction: **{direction}**")
