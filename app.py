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
    except:
        df['sentiment'] = "Neutral"
    return df

def fetch_massive_data(market_code, lang, api_key):
    """
    'Verticals' Strategy: Fetching data by distinct topics to maximize volume.
    Excluding internal domains to prevent false negative sentiment from FAQ/Support pages.
    """
    url = "https://google.serper.dev/search"
    all_items = []
    
    # Topic list - multiplies results x4 per country
    # We added "-site:lyreco.com" to EXCLUDE the official domain and subdomains!
    topics = {
        "General Brand": "Lyreco -site:lyreco.com",
        "HR & Careers": f"Lyreco {lang == 'pl' and 'praca opinie' or 'careers reviews'} -site:lyreco.com",
        "Logistics & Ops": f"Lyreco {lang == 'pl' and 'dostawa problem' or 'delivery issues'} -site:lyreco.com",
        "CSR & Sustainability": f"Lyreco {lang == 'pl' and 'ekologia' or 'sustainability'} -site:lyreco.com"
    }

    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    for category, query in topics.items():
        payload = json.dumps({
            "q": query,
            "gl": market_code,
            "hl": lang,
            "num": 40,       # Maximize fetch size
            "tbs": "qdr:m12" # Lookback 12 months
        })

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            results = response.json()
            
            # Organic Results
            if 'organic' in results:
                for r in results['organic']:
                    all_items.append({
                        'Category': category,
                        'Title': r.get('title', ''),
                        'Link': r.get('link', ''),
                        'Snippet': r.get('snippet', ''),
                        'Source': 'Web'
                    })
            # News Results
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

# --- UI LAYOUT ---
st.title("📈 Lyreco Global: Big Data Volume Monitor")
st.markdown("""
This dashboard uses a **Multi-Vertical Scanning Strategy** to gather maximum intelligence volume. 
Instead of a simple keyword search, it penetrates 4 key operational pillars for each market: 
**General Brand, HR/Careers, Logistics, and Sustainability**.
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not API_KEY:
        API_KEY = st.text_input("Enter Serper API Key:", type="password")
    
    st.markdown("---")
    
    MARKETS = {
        "France 🇫🇷": {"code": "fr", "lang": "fr"},
        "Poland 🇵🇱": {"code": "pl", "lang": "pl"},
        "UK 🇬🇧": {"code": "gb", "lang": "en"},
        "Italy 🇮🇹": {"code": "it", "lang": "it"},
        "Germany 🇩🇪": {"code": "de", "lang": "de"},
        "Spain 🇪🇸": {"code": "es", "lang": "es"},
        "Benelux 🇧🇪🇳🇱": {"code": "be", "lang": "nl"},
    }
    
    selected_markets = st.multiselect("Select Markets:", list(MARKETS.keys()), default=["France 🇫🇷", "Poland 🇵🇱"])
    
    st.markdown("---")
    
    # --- DATA SOURCES SECTION ---
    st.markdown("### 📡 Data Sources")
    st.info("""
    The system aggregates intelligence from:
    * **Professional:** LinkedIn, Official Press
    * **Reviews:** Trustpilot, Glassdoor, GoWork, Indeed
    * **Social:** Reddit, Forums, Quora
    * **Media:** Google News, Industry Blogs
    """)
    
    st.warning("⚠️ Warning: This mode consumes 4 API credits per selected market.")
    
    run_btn = st.button("🚀 FETCH MASSIVE DATA", type="primary")

if run_btn and API_KEY:
    full_data = []
    progress = st.progress(0)
    status_text = st.empty()
    
    for i, market in enumerate(selected_markets):
        config = MARKETS[market]
        status_text.text(f"📡 Scanning ecosystem: {market}...")
        
        data = fetch_massive_data(config['code'], config['lang'], API_KEY)
        
        for item in data:
            item['Market'] = market
            full_data.append(item)
            
        progress.progress((i + 1) / len(selected_markets))
        
    status_text.empty()
    progress.empty()
        
    if full_data:
        df = pd.DataFrame(full_data)
        df = df.drop_duplicates(subset=['Title'])
        
        with st.spinner(f"🧠 AI analyzing sentiment for {len(df)} records..."):
            df = analyze_sentiment(df)
            
        # --- KPI BOARD ---
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Data Points", len(df))
        k2.metric("Active Markets", len(selected_markets))
        k3.metric("Dominant Topic", df['Category'].mode()[0])
        
        st.divider()
        
        # --- CHARTS ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Volume by Category")
            st.caption("Visualizes the volume of discussion across operational pillars. Colors represent categories, not sentiment.")
            
            # Using a distinct color sequence for categories to avoid confusion with sentiment
            fig = px.treemap(
                df, 
                path=['Market', 'Category'], 
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Bold # Distinct, non-sentiment colors
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("❤️ Global Sentiment")
            st.caption("AI-driven emotional analysis of all collected data points.")
            
            # Ensuring Neutral is GRAY as requested
            fig2 = px.pie(
                df, 
                names='sentiment', 
                color='sentiment', 
                color_discrete_map={
                    'Positive':'#00CC96', 
                    'Negative':'#EF553B', 
                    'Neutral':'#cccccc' # Gray
                }
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        # --- DATA TABLE ---
        st.subheader("🗄️ Intelligence Database")
        st.dataframe(
            df[['Market', 'Category', 'Title', 'sentiment', 'Link']],
            column_config={
                "Link": st.column_config.LinkColumn("Source URL")
            },
            use_container_width=True
        )
        
    else:
        st.error("No data found. Please check your API limits or try different markets.")

elif run_btn and not API_KEY:
    st.error("Please provide an API Key in the sidebar.")
