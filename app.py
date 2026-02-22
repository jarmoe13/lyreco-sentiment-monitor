import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import tempfile
import unicodedata
from transformers import pipeline
from anthropic import Anthropic
from fpdf import FPDF

# --- CONFIG ---
st.set_page_config(page_title="Lyreco Big Data Monitor", layout="wide", page_icon="📈")

# --- API KEYS ---
try:
    API_KEY = st.secrets.get("SERPER_API_KEY", "")
    ANTHROPIC_KEY = st.secrets.get("CLAUDE_KEY", "")
except:
    API_KEY = None
    ANTHROPIC_KEY = None

# --- AI MODEL (Sentiment) ---
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
    url = "https://google.serper.dev/search"
    all_items = []
    
    # We keep local search terms to fetch relevant regional data, but UI/Reports are in English
    topics = {
        "General Brand": "Lyreco",
        "HR & Careers": f"Lyreco {lang == 'pl' and 'praca opinie' or 'careers reviews'}",
        "Logistics & Ops": f"Lyreco {lang == 'pl' and 'dostawa problem' or 'delivery issues'}",
        "CSR & Sustainability": f"Lyreco {lang == 'pl' and 'ekologia' or 'sustainability'}"
    }

    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    for category, query in topics.items():
        payload = json.dumps({
            "q": query, "gl": market_code, "hl": lang, "num": 40, "tbs": "qdr:m12"
        })
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code != 200: continue
            results = response.json()
            if 'organic' in results:
                for r in results['organic']:
                    all_items.append({'Category': category, 'Title': r.get('title', ''), 'Link': r.get('link', ''), 'Source': 'Web'})
            if 'news' in results:
                for r in results['news']:
                    all_items.append({'Category': 'News', 'Title': r.get('title', ''), 'Link': r.get('link', ''), 'Source': 'News'})
        except Exception as e: pass
        
    return all_items

# --- CLAUDE 3 HAIKU INTEGRATION ---
def generate_executive_summary(df, api_key):
    try:
        client = Anthropic(api_key=api_key)
        
        negatives = df[df['sentiment'] == 'Negative']['Title'].tolist()[:15]
        positives = df[df['sentiment'] == 'Positive']['Title'].tolist()[:10]
        
        prompt = f"""
        You are the Chief Data Analyst at Lyreco. 
        Here is the latest data from our web scraping and social listening tool.
        
        Negative signals ({len(negatives)}):
        {negatives}
        
        Positive signals ({len(positives)}):
        {positives}
        
        Write a short, professional 'Executive Summary' (about 4-5 sentences) for the Board of Directors.
        Highlight the 2 most frequent problems (if applicable based on data) and 1 positive finding.
        Write in English, maintaining a highly professional, business-oriented, and advisory tone.
        """
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating AI summary: {str(e)}"

# --- PDF GENERATOR ---
def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def create_pdf_report(summary_text, total_records):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Lyreco AI Monitor - Executive Summary", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Total records analyzed: {total_records}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    safe_text = strip_accents(summary_text).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name

# --- UI LAYOUT ---
st.title("📈 Lyreco Global: Big Data Volume Monitor")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not API_KEY:
        API_KEY = st.text_input("Enter Serper API Key:", type="password")
        
    if not ANTHROPIC_KEY:
        ANTHROPIC_KEY = st.text_input("Enter Anthropic API Key (Claude 3):", type="password")
        st.caption("Press ENTER after pasting your keys!")
    
    st.markdown("---")
    
    MARKETS = {
        "France 🇫🇷": {"code": "fr", "lang": "fr"},
        "Poland 🇵🇱": {"code": "pl", "lang": "pl"},
        "UK 🇬🇧": {"code": "gb", "lang": "en"}
    }
    
    selected_markets = st.multiselect("Select Markets:", list(MARKETS.keys()), default=["France 🇫🇷", "Poland 🇵🇱"])
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
        
        # Filter out Lyreco domains
        df = df[~df['Link'].str.contains('lyreco.com', na=False, case=False)]
        
        if df.empty:
            st.warning("No data found after filtering out Lyreco domains.")
        else:
            with st.spinner(f"🧠 AI analyzing sentiment for {len(df)} records..."):
                df = analyze_sentiment(df)
            
            # --- AI EXECUTIVE SUMMARY ---
            st.subheader("🤖 AI Executive Summary")
            if ANTHROPIC_KEY:
                with st.spinner("Claude 3 Haiku is generating insights for the Board..."):
                    ai_summary = generate_executive_summary(df, ANTHROPIC_KEY)
                    st.success(ai_summary)
                    
                    # Generate and download PDF
                    pdf_path = create_pdf_report(ai_summary, len(df))
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_file,
                            file_name="Lyreco_Executive_Summary.pdf",
                            mime="application/pdf"
                        )
            else:
                st.info("Enter your Anthropic API Key in the sidebar to unlock AI summaries and PDF reports!")
                
            st.divider()
                
            # --- CHARTS ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Volume by Category")
                fig = px.treemap(df, path=['Market', 'Category'], color='Category', color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("❤️ Global Sentiment")
                fig2 = px.pie(df, names='sentiment', color='sentiment', color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#cccccc'})
                st.plotly_chart(fig2, use_container_width=True)
                
            # --- DATA TABLE ---
            st.subheader("🗄️ Intelligence Database")
            st.dataframe(df[['Market', 'Category', 'Title', 'sentiment', 'Link']], use_container_width=True)
            
    else:
        st.error("No data found.")
elif run_btn and not API_KEY:
    st.error("Please provide a Serper API Key in the sidebar.")
