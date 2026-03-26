"""
TemporalGuard-RAG - Professional Financial Analysis Interface
Clean, modern design inspired by Bloomberg Terminal & ChatGPT
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import requests
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="TemporalGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0f0f0f;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #252525;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --accent: #10a37f;
        --accent-hover: #1a7f64;
        --border: #333333;
        --success: #10a37f;
        --warning: #f59e0b;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        max-width: 900px;
        padding: 2rem 1rem;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    .app-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Input container */
    .input-section {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Labels */
    .stSelectbox label, .stTextArea label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    
    /* Selectbox - FORCE dark background */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div {
        background-color: #252525 !important;
        background: #252525 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    /* Selected value - white text */
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div,
    [data-baseweb="select"] span {
        color: #ffffff !important;
    }
    
    /* Dropdown menu - white bg, dark text */
    .stSelectbox [data-baseweb="menu"],
    [data-baseweb="menu"],
    [data-baseweb="popover"] > div {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li,
    .stSelectbox [data-baseweb="menu"] li span,
    .stSelectbox [data-baseweb="menu"] li div,
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] li span {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li:hover {
        background-color: #e0e0e0 !important;
    }
    
    /* Dropdown arrow icon */
    .stSelectbox svg {
        fill: #a0a0a0 !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #252525 !important;
        background: #252525 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #666 !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
    }
    
    .stButton > button:disabled {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-secondary) !important;
    }
    
    /* Response box */
    .response-box {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .response-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    .response-icon {
        width: 28px;
        height: 28px;
        background: var(--accent);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    
    .response-label {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .response-time {
        margin-left: auto;
        color: var(--text-secondary);
        font-size: 0.75rem;
    }
    
    .response-content {
        color: var(--text-primary);
        line-height: 1.75;
        font-size: 0.95rem;
    }
    
    /* Metrics */
    .stMetric {
        background: var(--bg-tertiary);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    /* Toggle - fix text visibility */
    .stToggle label span {
        color: #ffffff !important;
    }
    
    .stToggle p {
        color: #ffffff !important;
    }
    
    .stToggle [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* Text input */
    .stTextInput input {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.6rem 1rem !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent) !important;
    }
    
    .stTextInput input::placeholder {
        color: #666 !important;
    }
    
    /* Checkbox */
    .stCheckbox label span {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-tertiary);
        padding: 0.25rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        background: transparent !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Code */
    .stCodeBlock, code {
        background: var(--bg-tertiary) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }
    
    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
    }
    
    /* Alerts */
    .stAlert {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-color: var(--accent) transparent transparent transparent !important;
    }
    
    /* Progress */
    .stProgress > div > div {
        background-color: var(--accent) !important;
    }
    
    /* JSON viewer */
    [data-testid="stJson"] {
        background: var(--bg-tertiary) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
    }
    
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
    
    .footer-items {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 0.75rem;
    }
    
    /* FORCE ALL TEXT WHITE - except dropdown menu items */
    .stApp, .stApp *:not([data-baseweb="menu"] *) {
        color: #ffffff !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #ffffff !important;
    }
    
    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    .stTextInput label, .stSelectbox label, .stTextArea label, .stCheckbox label {
        color: #ffffff !important;
    }
    
    .stRadio label, .stMultiSelect label, .stDateInput label, .stTimeInput label {
        color: #ffffff !important;
    }
    
    /* Info/Warning/Success boxes */
    .stInfo, .stWarning, .stSuccess, .stError {
        color: #ffffff !important;
    }
    
    [data-testid="stNotification"] {
        color: #ffffff !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader p, .streamlit-expanderContent p {
        color: #ffffff !important;
    }
    
    /* Subheader */
    [data-testid="stSubheader"] {
        color: #ffffff !important;
    }
    
    /* Caption and small text */
    .stCaption, small {
        color: #b0b0b0 !important;
    }
    
    /* DROPDOWN FIXES - Menu items must be dark */
    .stSelectbox [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li,
    .stSelectbox [data-baseweb="menu"] li span,
    .stSelectbox [data-baseweb="menu"] li div,
    .stSelectbox [data-baseweb="menu"] ul li {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li:hover,
    .stSelectbox [data-baseweb="menu"] li:hover span {
        background-color: #e0e0e0 !important;
        color: #1a1a1a !important;
    }
    
    /* Input box dark background, white text */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #252525 !important;
    }
    
    .stSelectbox [data-baseweb="select"] input {
        color: #ffffff !important;
        caret-color: #ffffff !important;
        cursor: text !important;
    }
    
    /* Text area cursor */
    .stTextArea textarea {
        caret-color: #ffffff !important;
        cursor: text !important;
    }
    
    .stTextArea textarea:focus {
        outline: none !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def load_company_list():
    """Load company list with tickers and names."""
    companies = {}
    
    # Load local data
    for dir_path in [Path("data/raw/xbrl_structured"), Path("data/raw/yahoo_finance")]:
        if dir_path.exists():
            for f in dir_path.glob("*.json"):
                try:
                    with open(f, 'r') as file:
                        data = json.load(file)
                        if 'company_info' in data:
                            ticker = data.get('ticker', f.stem.replace('_', '.')).upper()
                            name = data['company_info'].get('name', ticker)
                        else:
                            ticker = f.stem.upper()
                            name = data.get('entity_name', ticker)
                        companies[ticker] = name
                except:
                    pass
    
    # Fetch SEC list
    try:
        sec_cache = Path("data/sec_companies_cache.json")
        if sec_cache.exists():
            with open(sec_cache, 'r') as f:
                companies.update(json.load(f))
        else:
            resp = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": "TemporalGuardRAG research@example.com"},
                timeout=10
            )
            if resp.status_code == 200:
                sec_companies = {e['ticker']: e['title'] for e in resp.json().values()}
                companies.update(sec_companies)
                sec_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(sec_cache, 'w') as f:
                    json.dump(sec_companies, f)
    except:
        pass
    
    # Global tickers
    companies.update({
        "RELIANCE.NS": "Reliance Industries (India)",
        "TCS.NS": "Tata Consultancy Services (India)",
        "INFY.NS": "Infosys (India)",
        "HDFCBANK.NS": "HDFC Bank (India)",
        "7203.T": "Toyota Motor (Japan)",
        "6758.T": "Sony Group (Japan)",
        "0700.HK": "Tencent Holdings (Hong Kong)",
        "9988.HK": "Alibaba Group (Hong Kong)",
        "HSBA.L": "HSBC Holdings (UK)",
        "BP.L": "BP plc (UK)",
        "SAP.DE": "SAP SE (Germany)",
        "MC.PA": "LVMH (France)",
    })
    
    return companies


def init_state():
    """Initialize session state."""
    for key, val in {'history': [], 'result': None, 'orch': None, 'provider': 'ollama', 'model': 'llama3.2'}.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_orchestrator():
    """Load orchestrator."""
    if st.session_state.orch is None:
        try:
            from src.rag_system.vector_store import TemporalVectorStore
            from src.agents.orchestrator import MultiAgentOrchestrator
            
            vs = TemporalVectorStore()
            st.session_state.orch = MultiAgentOrchestrator(
                vector_store=vs,
                provider=st.session_state.provider,
                model_name=st.session_state.model
            )
            return True
        except Exception as e:
            st.error(f"Load failed: {e}")
            return False
    return True


def main():
    """Main app."""
    init_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        st.session_state.provider = st.selectbox("Provider", ["ollama", "openai"])
        if st.session_state.provider == "ollama":
            st.session_state.model = st.selectbox("Model", ["llama3.2", "llama3.1", "mistral"])
        else:
            key = st.text_input("API Key", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
            st.session_state.model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
    
    # Header
    st.markdown("## 🛡️ TemporalGuard")
    st.caption("Financial Analysis with Look-Ahead Bias Prevention")
    st.divider()
    
    # Load companies
    companies = load_company_list()
    
    # Build full list of options for dropdown
    all_options = [""] + [f"{t} — {n}" for t, n in companies.items()]
    
    # Single searchable dropdown (Streamlit selectbox)
    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox(
            "🔍 Select Company (type to search)",
            all_options,
            index=0,
            placeholder="Choose a company..."
        )
        ticker = sel.split("—")[0].strip() if sel else ""
    
    with c2:
        st.write("")  # Spacing
        auto = st.checkbox("📅 Auto-detect date", value=True, help="Infer date from query")
        date = None if auto else st.date_input("Analysis Date", datetime.now() - timedelta(30))
    
    # Show selected ticker
    if ticker:
        st.success(f"Selected: **{ticker}** — {companies.get(ticker, '')}")
    
    # Question input
    query = st.text_area(
        "💬 Your Question",
        placeholder="What was Apple's revenue growth in FY 2023?\nCompare profit margins to last year\nWhat's the current debt-to-equity ratio?",
        height=100
    )
    
    # Analyze button
    st.write("")  # Spacing
    if st.button("✨ Analyze", disabled=not (query and ticker), use_container_width=True, type="primary"):
        run_analysis(query, ticker, date)
    
    # Results
    if st.session_state.result:
        show_results(st.session_state.result)
    
    # Footer
    st.divider()
    cols = st.columns(4)
    cols[0].caption("🔒 Temporal Safety")
    cols[1].caption("📊 SEC Filings")
    cols[2].caption("🌍 Global Coverage")
    cols[3].caption("⚡ Multi-Agent")


def run_analysis(query, ticker, date):
    """Run analysis."""
    with st.spinner("Analyzing..."):
        if not load_orchestrator():
            return
        try:
            dp = date.strftime('%Y%m%d') if date else None
            result = st.session_state.orch.process_query(query=query, ticker=ticker, analysis_date=dp, verbose=False)
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error: {e}")


def show_results(result):
    """Display results."""
    st.divider()
    
    ticker = result.get('ticker', 'N/A')
    answer = result.get('final_answer', '')
    clean = clean_text(answer)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"🛡️ {ticker} Analysis")
    with col2:
        st.caption(f"⏱️ {datetime.now().strftime('%H:%M')}")
    
    # Answer box
    st.info(clean if clean else "No response generated.")
    
    # Metrics
    stages = result.get('stages', {})
    valid = stages.get('temporal', {}).get('is_valid', True)
    docs = stages.get('document_retrieval', {}).get('document_count', 0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "✓ Valid" if valid else "⚠ Check")
    c2.metric("Sources", f"{docs}")
    c3.metric("Date", str(result.get('analysis_date', 'Auto'))[:10])
    c4.metric("Model", st.session_state.model)
    
    # Details
    with st.expander("📋 Technical Details"):
        tabs = st.tabs(["Temporal", "Calculations", "Documents"])
        with tabs[0]:
            st.json(stages.get('temporal', {}))
        with tabs[1]:
            calc_out = stages.get('calculations', {}).get('output', 'N/A')
            st.code(str(calc_out)[:800] if calc_out else "No calculations")
        with tabs[2]:
            doc_out = stages.get('document_retrieval', {}).get('output', 'N/A')
            st.text(str(doc_out)[:500] if doc_out else "No documents")
    
    st.download_button("📥 Export Results", json.dumps(result, indent=2, default=str), f"{ticker}.json", "application/json")


def clean_text(text):
    """Clean response text."""
    if not text:
        return "No response"
    
    skip = ['TEMPORAL AGENT:', 'DOCUMENT AGENT:', 'CALCULATION AGENT:', 'VERIFICATION AGENT:', '━', '═', '─']
    lines = [l for l in text.split('\n') if not any(s in l for s in skip)]
    lines = [l for l in lines if not l.strip().startswith('- Valid:') and not l.strip().startswith('- Documents')]
    
    result = '\n'.join(lines).strip()
    return result if result else "No response"


if __name__ == "__main__":
    main()
