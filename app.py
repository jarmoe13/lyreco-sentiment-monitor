import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from transformers import pipeline
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Intel [PRO]", layout="wide", page_icon="💎")

# --- API SETUP ---
# Próbujemy pobrać klucz z sekretów Streamlit Cloud
try:
    API_KEY = st.secrets["SERPER_API_KEY"]
except:
    # Fallback dla testów lokalnych lub gdy sekret nie jest ustawiony
    API_KEY = None

# --- AI MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Initializing AI Engine..."):
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
        df['score'] = [r['score'] for r in results]
    except Exception as e:
        st.error(f"AI Error: {e}")
        df['sentiment'] = "Neutral"
        df['score'] = 0.5
    return df

def fetch_serper_data(query, country_code, api_key):
    """Profesjonalne pobieranie danych z Google via Serper.dev"""
    url = "https://google.serper.dev/search"
    
    # Parametry zapytania (gl = geo location, hl = host language)
    payload = json.dumps({
        "q": query,
        "gl": country_code.lower(),
        "num": 20, # Pobieramy 20 wyników na kraj
        "tbs": "qdr:m6" # Ostatnie 6 miesięcy (qdr:m6)
    })
    
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()
        
        parsed_data = []
        
        # 1. Wyniki organiczne (Organic Search)
        if 'organic' in results:
            for r in results['organic']:
                parsed_data.append({
                    'Source': 'Web/Organic',
                    'Title': r.get('title', 'No Title'),
                    'Link': r.get('link', '#'),
                    'Snippet': r.get('snippet', ''),
                    'Date': r.get('date', 'Recent') # Serper czasem daje datę tekstową "2 days ago"
                })
                
        # 2. Wyniki News (Top Stories) - jeśli są
        if 'news' in results:
             for r in results['news']:
                parsed_data.append({
                    'Source': 'Google News',
                    'Title': r.get('title', 'No Title'),
                    'Link': r.get('link', '#'),
                    'Snippet': r.get('snippet', ''),
                    'Date': r.get('date', 'Recent')
                })
                
        return parsed_data
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return []

# --- UI LAYOUT ---
st.title("💎 Lyreco Strategic Intel [Premium API]")
st.markdown("**Powered by Serper.dev (Google Engine)** - No more blocks, just data.")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    
    # Sprawdzenie klucza
    if not API_KEY:
        st.warning("⚠️ Brak klucza API w Secrets!")
        user_key = st.text_input("Podaj klucz Serper.dev ręcznie:", type="password")
        if user_key:
            API_KEY = user_key
    else:
        st.success("✅ API Key Loaded from Secrets")
    
    st.divider()
    
    # Konfiguracja rynków
    MARKETS = {
        "France 🇫🇷": {"code": "fr", "query": "Lyreco e-commerce avis"},
        "Poland 🇵🇱": {"code": "pl", "query": "Lyreco platforma opinie"},
        "UK 🇬🇧": {"code": "gb", "query": "Lyreco webshop reviews"},
        "Italy 🇮🇹": {"code": "it", "query": "Lyreco recensioni servizio"},
    }
    
    selected_markets = st.multiselect("Markets to Scan:", list(MARKETS.keys()), default=list(MARKETS.keys()))
    run_btn = st.button("🚀 LAUNCH PREMIUM SCAN", type="primary")

# Main Logic
if run_btn:
    if not API_KEY:
        st.error("Stop! Musisz podać klucz API, aby uruchomić tryb Premium.")
    else:
        all_data = []
        progress = st.progress(0)
        
        for i, market in enumerate(selected_markets):
            config = MARKETS[market]
            st.toast(f"Scanning {market} via Google API...")
            
            # Pobieranie danych
            raw_data = fetch_serper_data(config['query'], config['code'], API_KEY)
            
            # Dodanie etykiety rynku
            for item in raw_data:
                item['Market'] = market
                all_data.append(item)
            
            progress.progress((i + 1) / len(selected_markets))
            
        progress.empty()
        
        # Przetwarzanie
        if all_data:
            df = pd.DataFrame(all_data)
            df = analyze_sentiment(df)
            
            # --- DASHBOARD ---
            k1, k2, k3 = st.columns(3)
            k1.metric("Premium Data Points", len(df))
            k2.metric("Market Coverage", len(selected_markets))
            positive_share = len(df[df['sentiment'] == 'Positive']) / len(df) * 100
            k3.metric("Positive Sentiment", f"{positive_share:.1f}%")
            
            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🌍 Mentions by Market")
                fig_bar = px.bar(df['Market'].value_counts().reset_index(), x='count', y='Market', 
                                 orientation='h', text_auto=True, color='count')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with c2:
                st.subheader("🧠 AI Sentiment Analysis")
                fig_pie = px.pie(df, names='sentiment', color='sentiment', 
                                 color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.subheader("📑 Verifiable Sources (Clickable)")
            st.dataframe(
                df[['Market', 'Title', 'sentiment', 'Link']],
                column_config={"Link": st.column_config.LinkColumn("Source URL")},
                use_container_width=True
            )
            
        else:
            st.warning("API nie zwróciło wyników. Sprawdź limity na serper.dev.")
