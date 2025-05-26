import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Tickerlisten ---
DAX40 = [
    "ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE",
    "DAI.DE", "DHER.DE", "DPW.DE", "DTE.DE", "FME.DE"
]
NASDAQ = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "META", "ADBE", "INTC", "PEP"
]

# --- Titel ---
st.title("📈 Aktien-Empfehlung & KI-Prognose")
st.markdown("Wähle ein Aktienpaket oder gib eigene Ticker ein. Die App zeigt Trenddaten und KI-Prognosen für 5–10 Tage.")

# --- Auswahlfeld ---
auswahl = st.selectbox("Quellliste wählen:", ["DAX40", "NASDAQ", "Manuelle Eingabe"])

if auswahl == "DAX40":
    tickers = DAX40
elif auswahl == "NASDAQ":
    tickers = NASDAQ
else:
    user_input = st.text_input("Gib eigene Tickersymbole ein (z. B. AAPL,MSFT,NVDA):")
    tickers = [x.strip().upper() for x in user_input.split(",") if x.strip()]

# --- Ticker analysieren ---
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
            "Trend 5d (%)": round(trend_5d * 100, 2),
            "Trend 10d (%)": round(trend_10d * 100, 2),
            "Volumen": int(avg_volume),
            "Ziel (Demo)": int(trend_5d > 0)  # Dummy-Ziel für KI-Training
        }
    except:
        return None

# --- Daten sammeln ---
daten = []
for t in tickers:
    res = analyze_ticker(t)
    if res:
        daten.append(res)

# --- Anzeige & KI-Modell ---
if daten:
    df = pd.DataFrame(daten)

    # Nur die relevanten Spalten extrahieren & in float umwandeln
    for col in ["Trend 5d (%)", "Trend 10d (%)", "Volumen", "Ziel (Demo)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ungültige Zeilen entfernen
    df = df.dropna(subset=["Trend 5d (%)", "Trend 10d (%)", "Volumen", "Ziel (Demo)"])

    if df.empty:
        st.warning("Keine gültigen Daten für die Analyse gefunden.")
    else:
        # KI trainieren
        X = df[["Trend 5d (%)", "Trend 10d (%)", "Volumen"]]
        y = df["Ziel (Demo)"].astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        df["KI-Vorhersage"] = model.predict(X)
        df["KI-Vorhersage"] = df["KI-Vorhersage"].map({1: "📈 Steigt", 0: "📉 Fällt"})

        # Sortieren mit vollständig neuer Spalte, explizit float
        sort_col = pd.to_numeric(df["Trend 5d (%)"], errors="coerce")
        df["Trend 5d (%) (float)"] = sort_col.values  # gleicher Index

        df = df.sort_values(by="Trend 5d (%) (float)", ascending=False)
        df = df.drop(columns=["Ziel (Demo)", "Trend 5d (%) (float)"])

        st.success("Analyse abgeschlossen – Ergebnisse unten:")
        st.dataframe(df.set_index("Ticker"))

else:
    st.warning("Keine gültigen Aktien gefunden oder fehlerhafte Ticker.")
