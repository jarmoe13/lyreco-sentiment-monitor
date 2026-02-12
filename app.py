import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from transformers import pipeline

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Big Data Monitor", layout="wide", page_icon="📈")

# --- API KEY ---
try:
    API_KEY = st.secrets["SERPER_API_KEY"]
except:
    API_KEY = None

# --- AI MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Ładowanie silnika AI..."):
        sentiment_pipeline = load_model()
except: pass

# --- LOGIC ---
def map_sentiment(label):
    if 'label_0' in str(label).lower() or 'neg' in str(label).lower(): return 'Negative'
    if 'label_2' in str(label).lower() or 'pos' in str(label).lower(): return 'Positive'
    return 'Neutral'

def analyze_sentiment(df):
    if df.empty: return df
    try:
        results = sentiment_pipeline(df['Title'].tolist(), truncation=True, max_length=512)
        df['sentiment'] = [map_sentiment(r['label']) for r in results]
    except:
        df['sentiment'] = "Neutral"
    return df

def fetch_massive_data(market_code, lang, api_key):
    """
    Strategia 'Verticals': Pobieramy dane tematami, żeby zwiększyć wolumen.
    """
    url = "https://google.serper.dev/search"
    all_items = []
    
    # Lista tematów - to nam pomnoży wyniki x4
    topics = {
        "General": "Lyreco",
        "HR & Career": f"Lyreco {lang == 'pl' and 'praca opinie' or 'careers reviews'}",
        "Logistics": f"Lyreco {lang == 'pl' and 'dostawa problem' or 'delivery issues'}",
        "CSR & Sustainability": f"Lyreco {lang == 'pl' and 'ekologia' or 'sustainability'}"
    }

    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    for category, query in topics.items():
        payload = json.dumps({
            "q": query,
            "gl": market_code,
            "hl": lang,
            "num": 40,      # Zwiększamy do 40 wyników na strzał!
            "tbs": "qdr:m12" # Szukamy 12 miesięcy wstecz (więcej danych)
        })

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            results = response.json()
            
            # Organic
            if 'organic' in results:
                for r in results['organic']:
                    all_items.append({
                        'Category': category,
                        'Title': r.get('title', ''),
                        'Link': r.get('link', ''),
                        'Snippet': r.get('snippet', ''),
                        'Source': 'Web'
                    })
            # News (jeśli są)
            if 'news' in results:
                for r in results['news']:
                    all_items.append({
                        'Category': 'News',
                        'Title': r.get('title', ''),
                        'Link': r.get('link', ''),
                        'Snippet': r.get('snippet', ''),
                        'Source': 'News'
                    })
        except: pass
        
    return all_items

# --- UI ---
st.title("📈 Lyreco Global: Big Data Volume Monitor")
st.markdown("Tryb: **Massive Fetch** (General + HR + Logistics + CSR)")

with st.sidebar:
    if not API_KEY:
        API_KEY = st.text_input("API Key:", type="password")
    
    MARKETS = {
        "France 🇫🇷": {"code": "fr", "lang": "fr"},
        "Poland 🇵🇱": {"code": "pl", "lang": "pl"},
        "UK 🇬🇧": {"code": "gb", "lang": "en"},
        "Italy 🇮🇹": {"code": "it", "lang": "it"},
        "Germany 🇩🇪": {"code": "de", "lang": "de"},
    }
    
    selected_markets = st.multiselect("Rynki:", list(MARKETS.keys()), default=["France 🇫🇷", "Poland 🇵🇱"])
    st.warning("⚠️ Uwaga: Ten tryb zużywa więcej kredytów API (4 zapytania na kraj).")
    run_btn = st.button("🚀 POBIERZ DUŻO DANYCH", type="primary")

if run_btn and API_KEY:
    full_data = []
    progress = st.progress(0)
    
    for i, market in enumerate(selected_markets):
        config = MARKETS[market]
        st.toast(f"Pobieranie: {market} (4 kategorie)...")
        
        data = fetch_massive_data(config['code'], config['lang'], API_KEY)
        
        for item in data:
            item['Market'] = market
            full_data.append(item)
            
        progress.progress((i + 1) / len(selected_markets))
        
    if full_data:
        df = pd.DataFrame(full_data)
        # Usuwamy duplikaty (bo w różnych kategoriach mogło znaleźć to samo)
        df = df.drop_duplicates(subset=['Title'])
        
        with st.spinner(f"Analiza AI dla {len(df)} rekordów..."):
            df = analyze_sentiment(df)
            
        # --- DASHBOARD ---
        k1, k2, k3 = st.columns(3)
        k1.metric("Zebrane Dane (Total)", len(df))
        k2.metric("Rynki", len(selected_markets))
        k3.metric("Dominujący Temat", df['Category'].mode()[0])
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Rozkład wg Kategorii")
            fig = px.treemap(df, path=['Market', 'Category'], color='Category')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Sentyment Globalny")
            fig2 = px.pie(df, names='sentiment', color='sentiment', 
                          color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#cccccc'})
            st.plotly_chart(fig2, use_container_width=True)
            
        st.subheader("Baza Danych")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.error("Brak danych.")
