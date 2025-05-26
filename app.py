import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- Mobilfreundliches Seitenlayout ---
st.set_page_config(layout="wide")

# --- Tickerlisten ---
DAX40 = ["ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "DHER.DE", "DPW.DE", "DTE.DE", "FME.DE"]
NASDAQ = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "INTC", "PEP"]

st.title("📈 Aktien-Empfehlung & KI-Prognose")

col1, col2 = st.columns([1, 2])
with col1:
    auswahl = st.selectbox("Quelle:", ["DAX40", "NASDAQ", "Manuell"])
    if auswahl == "DAX40":
        tickers = DAX40
    elif auswahl == "NASDAQ":
        tickers = NASDAQ
    else:
        user_input = st.text_input("Eigene Ticker (z. B. AAPL,MSFT,NVDA):")
        tickers = [x.strip().upper() for x in user_input.split(",") if x.strip()]

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="30d", interval="1d", progress=False)
        if df.empty or len(df) < 12:
            return None
        last_close = df["Close"].iloc[-1]
        trend_5d = (last_close - df["Close"].iloc[-6]) / df["Close"].iloc[-6]
        trend_10d = (last_close - df["Close"].iloc[-11]) / df["Close"].iloc[-11]
        avg_volume = df["Volume"].iloc[-5:].mean()
        return {
            "Ticker": ticker,
            "Letzter Kurs": round(last_close, 2),
            "Trend_5d": round(trend_5d * 100, 2),
            "Trend_10d": round(trend_10d * 100, 2),
            "Volumen": int(avg_volume),
            "Ziel (Demo)": int(trend_5d > 0),
            "Daten": df  # für Charts
        }
    except:
        return None

# --- Analyse starten ---
daten = [analyze_ticker(t) for t in tickers]
daten = [d for d in daten if d is not None]

if not daten:
    st.warning("Keine gültigen Daten gefunden.")
else:
    df = pd.DataFrame([{k: v for k, v in d.items() if k != "Daten"} for d in daten])
    for col in ["Trend_5d", "Trend_10d", "Volumen", "Ziel (Demo)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    X = df[["Trend_5d", "Trend_10d", "Volumen"]]
    y = df["Ziel (Demo)"].astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df["KI-Vorhersage"] = model.predict(X)
    df["KI-Vorhersage"] = df["KI-Vorhersage"].map({1: "📈 Steigt", 0: "📉 Fällt"})
    df = df.sort_values(by="Trend_5d", ascending=False)
    df = df.drop(columns=["Ziel (Demo)"])

    st.success("Analyse abgeschlossen – Ergebnisse:")
    st.dataframe(df.set_index("Ticker"))

    # Einzel-Chart-Ansicht auf Mobilgeräten
    st.markdown("### 📊 Kursverlauf anzeigen")
    selected = st.selectbox("Ticker auswählen:", df["Ticker"].tolist())

    daten_dict = {d["Ticker"]: d["Daten"] for d in daten}
    chart_df = daten_dict[selected]
    chart_df["SMA_5"] = chart_df["Close"].rolling(window=5).mean()
    chart_df["SMA_10"] = chart_df["Close"].rolling(window=10).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chart_df["Close"], label="Kurs", linewidth=2)
    ax.plot(chart_df["SMA_5"], label="SMA 5", linestyle="--")
    ax.plot(chart_df["SMA_10"], label="SMA 10", linestyle=":")
    ax.set_title(f"Kursverlauf {selected}")
    ax.legend()
    st.pyplot(fig) 
