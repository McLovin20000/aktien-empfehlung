import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Tickerlisten ---
DAX40 = ["ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "DHER.DE", "DPW.DE", "DTE.DE", "FME.DE"]
NASDAQ = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "INTC", "PEP"]

# --- Titel ---
st.title("📈 Aktien-Empfehlung & KI-Prognose")
st.markdown("Wähle ein Aktienpaket oder gib eigene Ticker ein. Die App zeigt Trenddaten und KI-Prognosen für 5–10 Tage.")

# --- Auswahlfeld ---
quelle = st.selectbox("Aktienauswahl:", ["DAX40", "NASDAQ", "Manuell"])

if quelle == "DAX40":
    tickers = DAX40
elif quelle == "NASDAQ":
    tickers = NASDAQ
else:
    user_input = st.text_input("Gib Tickersymbole ein (z. B. AAPL,MSFT,NVDA):")
    tickers = [x.strip().upper() for x in user_input.split(",") if x.strip()]

# --- Datenanalysefunktion ---
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
            "Ziel (Demo)": int(trend_5d > 0)  # Dummy-Ziel für KI
        }
    except:
        return None

# --- Daten sammeln ---
daten = [analyze_ticker(t) for t in tickers]
daten = [d for d in daten if d is not None]

if not daten:
    st.warning("Keine gültigen Daten gefunden.")
else:
    df = pd.DataFrame(daten)

    # Spalten in richtigen Typ konvertieren
    for col in ["Trend_5d", "Trend_10d", "Volumen", "Ziel (Demo)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Modell trainieren
    X = df[["Trend_5d", "Trend_10d", "Volumen"]]
    y = df["Ziel (Demo)"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df["KI-Vorhersage"] = model.predict(X)
    df["KI-Vorhersage"] = df["KI-Vorhersage"].map({1: "📈 Steigt", 0: "📉 Fällt"})

    df = df.drop(columns=["Ziel (Demo)"])
    df = df.sort_values(by="Trend_5d", ascending=False)

    st.success("Analyse abgeschlossen:")
    st.dataframe(df.set_index("Ticker"))
