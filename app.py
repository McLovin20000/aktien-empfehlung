import streamlit as st
import yfinance as yf
import pandas as pd

# Beispielhafte Aktienliste
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]

def analyze_ticker(ticker):
    data = yf.download(ticker, period="15d", interval="1d", progress=False)
    if data.empty or len(data) < 10:
        return None

    last_close = data["Close"].iloc[-1]
    prev_5_avg = data["Close"].iloc[-6:-1].mean()
    trend_pct = ((last_close - prev_5_avg) / prev_5_avg) * 100

    return {
        "Aktie": ticker,
        "Letzter Kurs": round(last_close, 2),
        "Durchschnitt (vor 5 Tagen)": round(prev_5_avg, 2),
        "Trend (5 Tage %)": round(trend_pct, 2)
    }

st.title("📈 Aktien mit Aufwärtspotenzial (5–10 Tage)")
st.write("Diese App analysiert Aktien mit guter Kurzfrist-Tendenz.")

results = []
for ticker in TICKERS:
    res = analyze_ticker(ticker)
    if res:
        results.append(res)

df = pd.DataFrame(results)

if not df.empty and "Trend (5 Tage %)" in df.columns:
    df = df.sort_values(by="Trend (5 Tage %)", ascending=False)
    st.dataframe(df)
else:
    st.warning("Keine verwertbaren Daten verfügbar.")
st.dataframe(df)
