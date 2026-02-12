import streamlit as st
import pandas as pd
import plotly.express as px
from pygooglenews import GoogleNews
from duckduckgo_search import DDGS
from transformers import pipeline
import time

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Sentinel AI", layout="wide", page_icon="🌍")

# --- CUSTOM STYLES ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px;}
</style>
""", unsafe_allow_html=True)

# --- AI MODEL LOADING ---
@st.cache_resource
def load_model():
    # Multilingual model (XLM-RoBERTa) - works for EN, PL, FR, DE, etc.
    return pipeline("sentiment-analysis", 
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment", 
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment")

try:
    with st.spinner("Initializing AI Engine..."):
        sentiment_pipeline = load_model()
except Exception as e:
    st.error(f"Critical Error loading model: {e}")

# --- CORE LOGIC ---
def map_label(label):
    label = str(label).lower()
    if 'label_0' in label or 'neg' in label: return 'Negative'
    if 'label_1' in label or 'neu' in label: return 'Neutral'
    if 'label_2' in label or 'pos' in label: return 'Positive'
    return 'Neutral'

def analyze_data(df):
    if df.empty: return df
    texts = df['title'].tolist()
    # Batch processing
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    df['sentiment'] = [map_label(res['label']) for res in results]
    df['score'] = [res['score'] for res in results]
    return df

def fetch_data_boosted(base_query):
    all_data = []
    
    # 1. GOOGLE NEWS (Multi-Region Strategy)
    # Scanning multiple markets for broader context
    regions = [('pl', 'PL'), ('en', 'GB'), ('fr', 'FR'), ('en', 'US')]
    
    status_text = st.empty()
    
    for lang, country in regions:
        status_text.text(f"📡 Scanning Google News ({country})...")
        try:
            gn = GoogleNews(lang=lang, country=country)
            search = gn.search(base_query, when="90d") # 90 days lookback
            for entry in search['entries']:
                all_data.append({
                    'source': f'Google News ({country})', 
                    'title': entry.title, 
                    'date': entry.published,
                    'link': entry.link
                })
        except: continue

    # 2. DUCKDUCKGO - "Deep Dive" (Reviews, Forums, Social)
    extra_queries = [
        f'{base_query} site:reddit.com',
        f'{base_query} site:gowork.pl',
        f'{base_query} site:trustpilot.com',
        f'{base_query} reviews',
        f'{base_query} opinions'
    ]
    
    with DDGS() as ddgs:
        for q in extra_queries:
            status_text.text(f"🕵️ Deep Search: {q}...")
            try:
                # max_results limited to avoid timeouts
                results = list(ddgs.text(q, max_results=10)) 
                for r in results:
                    all_data.append({
                        'source': 'Social/Web', 
                        'title': r['title'] + " - " + r['body'][:50], 
                        'date': None, 
                        'link': r['href']
                    })
            except: continue
            time.sleep(0.5)

    status_text.empty()
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['title'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    return df

# --- UI LAYOUT (ENGLISH) ---
st.title("🌍 Lyreco Global Sentiment AI [Deep Dive]")

with st.sidebar:
    st.header("Analysis Parameters")
    query = st.text_input("Brand / Topic:", "Lyreco")
    st.caption("Mode: Deep Search (News + Reddit + Employee Reviews + Trustpilot)")
    run_btn = st.button("🚀 LAUNCH ENGINE", type="primary")
    st.markdown("---")
    st.markdown("Powered by **Streamlit & HuggingFace**")

if run_btn:
    df = fetch_data_boosted(query)
    
    if df.empty:
        st.error("No data found. Try a different keyword.")
    else:
        df = analyze_data(df)
        
        # --- TOP KPI ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Mentions", len(df))
        
        pos_count = len(df[df['sentiment'] == 'Positive'])
        neg_count = len(df[df['sentiment'] == 'Negative'])
        
        kpi2.metric("Positive", pos_count, delta=None)
        kpi3.metric("Negative", neg_count, delta_color="inverse")
        
        try:
            top_sent = df['sentiment'].mode()[0]
        except: top_sent = "N/A"
        kpi4.metric("Dominant Sentiment", top_sent)
        
        st.divider()

        # --- CHARTS ---
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(df, names='sentiment', color='sentiment', hole=0.4,
                             color_discrete_map={'Positive':'#00CC96', 'Neutral':'#AB63FA', 'Negative':'#EF553B'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.subheader("Source Analysis")
            source_counts = df['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig_bar = px.bar(source_counts, x='Count', y='Source', orientation='h', text_auto=True)
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- DATA TABLE ---
        st.subheader("📢 Live Mention Log")
        
        st.dataframe(
            df[['source', 'title', 'sentiment', 'score', 'link']],
            column_config={
                "link": st.column_config.LinkColumn("URL"),
                "score": st.column_config.ProgressColumn("AI Confidence", format="%.2f", min_value=0, max_value=1),
                "title": "Headline / Snippet",
                "source": "Origin"
            },
            use_container_width=True
        )

else:
    st.info("👈 Click the **'LAUNCH ENGINE'** button in the sidebar to start the global scan.")
