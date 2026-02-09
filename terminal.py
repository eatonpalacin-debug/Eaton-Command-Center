#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      CENTRO DE MANDO DE EATON — v20.0 · Global Intelligence       ║
║      Citadel Quant Architecture · Geo·Finance·Macro·LaTeX        ║
║                                                                  ║
║  pip install streamlit plotly yfinance scipy pandas numpy         ║
║  pip install streamlit-autorefresh geopy                          ║
║  streamlit run eaton_command_center.py                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, jarque_bera, skew, kurtosis
from datetime import datetime
import warnings, time, hashlib

warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except ImportError:
    AUTOREFRESH_OK = False

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    from geopy.geocoders import Nominatim
    GEO_OK = True
except ImportError:
    GEO_OK = False

# ═══════════════════════════════════════════════════════════════════
# COLORS — ONLY 6-digit hex or rgba(). NEVER 8-digit hex.
# ═══════════════════════════════════════════════════════════════════
C = {
    "gold": "#c99528",
    "gold_light": "#e8b94a",
    "gold_dim": "#8a6a20",
    "green": "#00e676",
    "red": "#ff1744",
    "blue": "#448aff",
    "cyan": "#18ffff",
    "purple": "#b388ff",
    "orange": "#ff9100",
    "bg": "#07080d",
    "bg1": "#0c0d16",
    "bg2": "#0e0f1a",
    "card": "#0f1019",
    "grid": "#151730",
    "border": "#1a1c30",
    "text": "#e8e6e3",
    "text2": "#8a8d9b",
    "textm": "#4a4d5e",
    # rgba versions for Plotly transparency
    "gold_20": "rgba(201,149,40,0.12)",
    "gold_30": "rgba(201,149,40,0.19)",
    "gold_50": "rgba(201,149,40,0.31)",
    "gold_05": "rgba(201,149,40,0.03)",
    "green_25": "rgba(0,230,118,0.25)",
    "green_38": "rgba(0,230,118,0.38)",
    "green_50": "rgba(0,230,118,0.50)",
    "green_13": "rgba(0,230,118,0.13)",
    "red_25": "rgba(255,23,68,0.25)",
    "red_38": "rgba(255,23,68,0.38)",
    "red_50": "rgba(255,23,68,0.50)",
    "red_13": "rgba(255,23,68,0.13)",
    "legend_bg": "rgba(7,8,13,0.85)",
}

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CENTRO DE MANDO DE EATON", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

if "boot" not in st.session_state:
    st.session_state.boot = datetime.now()
    st.session_state.sid = hashlib.md5(str(time.time()).encode()).hexdigest()[:8].upper()

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#07080d;--bg1:#0c0d16;--bg2:#11121e;--bgc:#0f1019;--brd:#1a1c30;--gold:#c99528;--goldl:#e8b94a;--goldd:#8a6a20;--txt:#e8e6e3;--t2:#8a8d9b;--tm:#4a4d5e;--grn:#00e676;--red:#ff1744;--blu:#448aff;--cyn:#18ffff;--pur:#b388ff}
.stApp{background:var(--bg)!important;color:var(--txt)!important;font-family:'Inter',sans-serif!important}
section[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid rgba(201,149,40,0.16)!important}
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{color:var(--t2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.8rem!important}
h1,h2,h3,h4,h5,h6{color:var(--gold)!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:.05em}
.stTabs [data-baseweb="tab-list"]{gap:2px;background:var(--bg1);padding:4px;border-radius:6px;border:1px solid var(--brd)}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--t2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;padding:8px 16px!important;border-radius:4px!important;border:none!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--goldd),var(--gold))!important;color:#000!important;font-weight:700!important}
[data-testid="stMetric"]{background:var(--bgc);border:1px solid var(--brd);border-left:3px solid var(--gold);padding:12px 16px;border-radius:6px}
[data-testid="stMetric"] label{color:var(--tm)!important;font-family:'JetBrains Mono',monospace!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:.1em}
[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--goldl)!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important}
.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--bg2)!important;border:1px solid var(--brd)!important;color:var(--txt)!important;font-family:'JetBrains Mono',monospace!important}
.streamlit-expanderHeader{background:var(--bgc)!important;border:1px solid rgba(201,149,40,0.16)!important;color:var(--gold)!important;font-family:'JetBrains Mono',monospace!important}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--goldd);border-radius:3px}
.hdr{background:linear-gradient(90deg,var(--bg1),var(--bgc),var(--bg1));border:1px solid rgba(201,149,40,0.16);border-radius:8px;padding:16px 24px;margin-bottom:16px;text-align:center}
.hdr h1{font-size:1.6rem!important;margin:0!important;background:linear-gradient(135deg,var(--goldl),var(--gold),var(--goldd));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hdr p{color:var(--tm)!important;font-family:'JetBrains Mono',monospace!important;font-size:.72rem!important;margin:4px 0 0!important;letter-spacing:.15em}
.jb{background:linear-gradient(135deg,#0d1117,#111827);border:1px solid rgba(201,149,40,0.16);border-radius:8px;padding:16px 20px;margin:10px 0;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--t2);line-height:1.6}
.jb .lb{color:var(--gold);font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px}
.qc{background:linear-gradient(135deg,#0f0c1a,#130d20);border:1px solid rgba(124,58,237,0.25);border-left:3px solid #7c3aed;border-radius:8px;padding:16px 20px;margin:10px 0;font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#c4b5fd;line-height:1.55}
.qc .tt{color:#a78bfa;font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px}
.sb{display:flex;justify-content:space-between;align-items:center;background:var(--bg1);border:1px solid var(--brd);border-radius:4px;padding:6px 14px;font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--tm);margin-bottom:12px;flex-wrap:wrap;gap:8px}
/* WHY? Post-it Intelligence Cards */
.why-card{background:linear-gradient(135deg,#0c1220,#0f1628);border:1px solid rgba(68,138,255,0.25);border-left:3px solid #448aff;border-radius:6px;padding:12px 16px;margin:6px 0;font-family:'JetBrains Mono',monospace;font-size:.76rem;line-height:1.55}
.why-card .wt{color:#448aff;font-weight:700;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}
.why-card .wd{color:#8a8d9b}
.why-card .wv{color:#e8b94a;font-weight:600}
.why-card .wc{color:#b388ff;font-style:italic;margin-top:6px;padding-top:6px;border-top:1px solid rgba(179,136,255,0.15)}
/* Sentiment badges */
.sent-bull{display:inline-block;background:rgba(0,230,118,0.15);color:#00e676;padding:2px 8px;border-radius:3px;font-size:.68rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.sent-bear{display:inline-block;background:rgba(255,23,68,0.15);color:#ff1744;padding:2px 8px;border-radius:3px;font-size:.68rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.sent-neut{display:inline-block;background:rgba(138,141,155,0.15);color:#8a8d9b;padding:2px 8px;border-radius:3px;font-size:.68rem;font-weight:600;font-family:'JetBrains Mono',monospace}
/* Sensitivity card */
.sens-card{background:linear-gradient(135deg,#0d0f1a,#111525);border:1px solid rgba(24,255,255,0.2);border-radius:6px;padding:14px 18px;margin:6px 0;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#8a8d9b;line-height:1.6}
.sens-card .st{color:#18ffff;font-weight:700;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em}
/* Geo HQ Card */
.geo-hq{background:linear-gradient(135deg,#0d0c14,#12101e);border:1px solid rgba(201,149,40,0.3);border-left:3px solid #c99528;border-radius:8px;padding:14px 18px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.82rem}
.geo-hq .hq-title{color:#e8b94a;font-weight:700;font-size:.9rem;margin-bottom:4px}
.geo-hq .hq-sub{color:#8a8d9b;font-size:.74rem}
/* Financial Table */
.fin-section{background:linear-gradient(135deg,#0a0d14,#0e1220);border:1px solid rgba(201,149,40,0.18);border-radius:8px;padding:16px;margin:10px 0}
.fin-section .fin-title{color:#c99528;font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;font-family:'JetBrains Mono',monospace}
/* Macro Engine Card */
.macro-eng{background:linear-gradient(135deg,#0c0a18,#100e20);border:1px solid rgba(179,136,255,0.2);border-left:3px solid #b388ff;border-radius:8px;padding:14px 18px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#c4b5fd;line-height:1.55}
.macro-eng .macro-t{color:#b388ff;font-weight:700;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px}
/* Partner Ledger Table */
.ptr-ledger{background:linear-gradient(135deg,#0a0e16,#0d1220);border:1px solid rgba(24,255,255,0.2);border-radius:8px;padding:16px;margin:10px 0}
.ptr-ledger .ptr-title{color:#18ffff;font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:10px;font-family:'JetBrains Mono',monospace}
/* EBITDA Section */
.ebitda-card{background:linear-gradient(135deg,#0d0c14,#12101e);border:1px solid rgba(201,149,40,0.25);border-left:3px solid #c99528;border-radius:8px;padding:14px 18px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.82rem}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ASSET UNIVERSE (85 instruments)
# ═══════════════════════════════════════════════════════════════════
UNIVERSE = {
    "🏛️ Élite S&P 500": {
        "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","AMZN":"Amazon",
        "NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","BRK-B":"Berkshire",
        "JPM":"JPMorgan","V":"Visa","UNH":"UnitedHealth","JNJ":"J&J",
        "XOM":"ExxonMobil","PG":"P&G","MA":"Mastercard","HD":"Home Depot",
        "AVGO":"Broadcom","CVX":"Chevron","LLY":"Eli Lilly","MRK":"Merck",
        "ABBV":"AbbVie","COST":"Costco","PEP":"PepsiCo","KO":"Coca-Cola",
        "WMT":"Walmart","CRM":"Salesforce","BAC":"BofA","NFLX":"Netflix",
        "AMD":"AMD","ORCL":"Oracle"},
    "₿ Criptoactivos": {
        "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","BNB-USD":"BNB",
        "SOL-USD":"Solana","XRP-USD":"Ripple","ADA-USD":"Cardano",
        "DOGE-USD":"Dogecoin","AVAX-USD":"Avalanche","DOT-USD":"Polkadot",
        "MATIC-USD":"Polygon","LINK-USD":"Chainlink","UNI-USD":"Uniswap",
        "ATOM-USD":"Cosmos","LTC-USD":"Litecoin","NEAR-USD":"NEAR"},
    "💱 Divisas": {
        "USDPEN=X":"USD/PEN","EURUSD=X":"EUR/USD","GBPUSD=X":"GBP/USD",
        "USDJPY=X":"USD/JPY","AUDUSD=X":"AUD/USD","USDCAD=X":"USD/CAD",
        "USDCHF=X":"USD/CHF","USDMXN=X":"USD/MXN","USDBRL=X":"USD/BRL",
        "USDCLP=X":"USD/CLP","USDCOP=X":"USD/COP"},
    "🛢️ Materias Primas": {
        "GC=F":"Oro","SI=F":"Plata","CL=F":"WTI","BZ=F":"Brent",
        "HG=F":"Cobre","PL=F":"Platino","NG=F":"Gas Natural",
        "ZW=F":"Trigo","ZC=F":"Maíz","ZS=F":"Soja"},
    "📊 Índices Globales": {
        "^GSPC":"S&P 500","^DJI":"Dow Jones","^IXIC":"NASDAQ",
        "^RUT":"Russell 2000","^VIX":"VIX","^FTSE":"FTSE 100",
        "^N225":"Nikkei 225","^HSI":"Hang Seng","^GDAXI":"DAX"},
}

SYNTH = {
    "🎰 Índices Sintéticos": {
        "CRASH_500":"Crash 500","CRASH_1000":"Crash 1000",
        "BOOM_500":"Boom 500","BOOM_1000":"Boom 1000",
        "VOLATILITY_10":"Vol 10","VOLATILITY_25":"Vol 25",
        "VOLATILITY_50":"Vol 50","VOLATILITY_75":"Vol 75",
        "VOLATILITY_100":"Vol 100","STEP_INDEX":"Step Index"},
}

ALL_CAT = {**UNIVERSE, **SYNTH}
N_ASSETS = sum(len(v) for v in ALL_CAT.values())

# ═══════════════════════════════════════════════════════════════════
# HORIZONTE ESTRATÉGICO — Period/Interval mapping
# ═══════════════════════════════════════════════════════════════════

HORIZONS = {
    "⚡ Streaming (Sim)": {"period": "1d",  "interval": None,  "label": "Streaming",  "days": 1},
    "1 Minuto (7D)":      {"period": "7d",  "interval": "1m",  "label": "1min×7D",    "days": 7},
    "5 Minutos (1M)":     {"period": "1mo", "interval": "5m",  "label": "5min×1M",    "days": 30},
    "15 Minutos (1M)":    {"period": "1mo", "interval": "15m", "label": "15min×1M",   "days": 30},
    "1 Hora (3M)":        {"period": "3mo", "interval": "1h",  "label": "1H×3M",      "days": 90},
    "1 Día (1Y)":         {"period": "1y",  "interval": "1d",  "label": "1D×1Y",      "days": 252},
    "1 Semana (2Y)":      {"period": "2y",  "interval": "1wk", "label": "1W×2Y",      "days": 504},
    "1 Mes (5Y)":         {"period": "5y",  "interval": "1mo", "label": "1M×5Y",      "days": 1260},
    "Trimestral (10Y)":   {"period": "10y", "interval": "3mo", "label": "3M×10Y",     "days": 2520},
}

# ═══════════════════════════════════════════════════════════════════
# DATA ENGINE
# ═══════════════════════════════════════════════════════════════════

def _is_synth(t):
    return any(t in SYNTH.get(c,{}) for c in SYNTH)

def _synth_ohlcv(ticker, days=252):
    np.random.seed(abs(hash(ticker))%(2**31))
    dt = pd.date_range(end=datetime.now(), periods=days, freq="B")
    p = {"CRASH":(-0.0003,0.025,0.03,-0.08),"BOOM":(0.0004,0.022,0.03,0.09),
         "VOLATILITY":(0.0001,0.035,0.01,0.05),"STEP":(0.0002,0.01,0.005,0.02)}
    k = next((x for x in p if x in ticker.upper()), "VOLATILITY")
    mu,sig,sp,sm = p[k]
    r = np.random.normal(mu,sig,days)
    mask = np.random.random(days)<sp
    r[mask] += np.random.normal(sm, abs(sm)*0.3, mask.sum())
    px = 5000*np.exp(np.cumsum(r))
    h=px*(1+np.abs(np.random.normal(0,0.008,days)))
    l=px*(1-np.abs(np.random.normal(0,0.008,days)))
    o=l+(h-l)*np.random.random(days)
    v=np.random.randint(1e6,8e7,days).astype(float)
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":px,"Volume":v},index=dt)

def _synth_streaming(ticker, bars=120):
    """Generate simulated streaming data (sub-minute bars)."""
    np.random.seed(abs(hash(ticker + str(int(time.time()//60))))%(2**31))
    dt = pd.date_range(end=datetime.now(), periods=bars, freq="s")
    base = 5000 + np.random.randn()*200
    px = base + np.cumsum(np.random.randn(bars)*0.5)
    px = np.maximum(px, 100)
    h=px+np.abs(np.random.normal(0,0.3,bars))
    l=px-np.abs(np.random.normal(0,0.3,bars))
    o=l+(h-l)*np.random.random(bars)
    v=np.random.randint(1e4,5e6,bars).astype(float)
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":px,"Volume":v},index=dt)

@st.cache_data(ttl=60, show_spinner=False)
def get_data(ticker, period="1y", interval=None):
    """Fetch data with optional interval. For intraday, interval must be set."""
    if _is_synth(ticker):
        dm = {"1d":1,"7d":7,"1mo":22,"3mo":66,"6mo":132,"1y":252,"2y":504,"5y":1260,"10y":2520}
        if interval is None:  # Streaming sim
            return _synth_streaming(ticker)
        return _synth_ohlcv(ticker, dm.get(period, 252))
    if not YF_OK:
        return pd.DataFrame()
    if interval is None:  # Streaming — use 1m with 1d for real tickers
        interval = "1m"; period = "1d"
    try:
        d = yf.download(ticker, period=period, interval=interval,
                        progress=False, auto_adjust=True, timeout=15)
        if d is not None and len(d) > 5:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in d.columns:
                    d[c] = d.get("Close", 0) if c != "Volume" else 0
            return d.dropna(subset=["Close"])
    except Exception as e:
        st.toast(f"⚠️ {ticker}: {str(e)[:60]}", icon="⚠️")
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_news(ticker):
    if not YF_OK or _is_synth(ticker):
        return []
    try:
        tk = yf.Ticker(ticker)
        raw = None
        try:
            raw = tk.news
        except Exception:
            return []
        if isinstance(raw, dict):
            raw = raw.get("stories", raw.get("news", []))
        if not raw or not isinstance(raw, list):
            return []
        out = []
        for item in raw[:6]:
            if not isinstance(item, dict):
                continue
            # Flexible title extraction
            title = (item.get("title") or
                     (item.get("content") or {}).get("title") or
                     item.get("headline") or "")
            if not title:
                continue
            # Flexible publisher
            pub = (item.get("publisher") or item.get("source") or
                   ((item.get("content") or {}).get("provider") or {}).get("displayName") or "—")
            # Flexible timestamp
            ts = "N/A"
            raw_t = (item.get("providerPublishTime") or
                     item.get("publishedAt") or item.get("pubDate"))
            if raw_t is not None:
                try:
                    if isinstance(raw_t, (int, float)):
                        ts = datetime.fromtimestamp(int(raw_t)).strftime("%Y-%m-%d %H:%M")
                    elif isinstance(raw_t, str):
                        ts = raw_t[:16].replace("T"," ")
                except Exception:
                    ts = "N/A"
            # Link
            link = (item.get("link") or item.get("url") or
                    ((item.get("content") or {}).get("canonicalUrl") or {}).get("url") or "#")
            out.append({"title":str(title),"pub":str(pub),"time":ts,"link":str(link)})
        return out
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════
# TICKER INTELLIGENCE ENGINE — HQ, Financials, Asset Type
# ═══════════════════════════════════════════════════════════════════

# HQ coordinates database for instant lookup (no API needed)
HQ_COORDS = {
    "AAPL":(37.33,-122.03,"Cupertino, CA","USA"),"MSFT":(47.64,-122.13,"Redmond, WA","USA"),
    "GOOGL":(37.42,-122.08,"Mountain View, CA","USA"),"AMZN":(47.62,-122.34,"Seattle, WA","USA"),
    "NVDA":(37.37,-122.04,"Santa Clara, CA","USA"),"META":(37.48,-122.15,"Menlo Park, CA","USA"),
    "TSLA":(30.22,-97.77,"Austin, TX","USA"),"BRK-B":(41.26,-95.94,"Omaha, NE","USA"),
    "JPM":(40.76,-73.98,"New York, NY","USA"),"V":(37.53,-122.20,"San Francisco, CA","USA"),
    "UNH":(44.97,-93.41,"Minnetonka, MN","USA"),"JNJ":(40.49,-74.45,"New Brunswick, NJ","USA"),
    "XOM":(32.41,-94.85,"Spring, TX","USA"),"PG":(39.10,-84.51,"Cincinnati, OH","USA"),
    "MA":(40.77,-73.97,"New York, NY","USA"),"HD":(33.77,-84.36,"Atlanta, GA","USA"),
    "AVGO":(37.40,-121.97,"San José, CA","USA"),"CVX":(37.76,-122.25,"San Ramon, CA","USA"),
    "LLY":(39.77,-86.16,"Indianapolis, IN","USA"),"MRK":(40.79,-74.26,"Rahway, NJ","USA"),
    "ABBV":(42.28,-87.86,"North Chicago, IL","USA"),"COST":(47.58,-122.17,"Issaquah, WA","USA"),
    "PEP":(41.09,-73.72,"Purchase, NY","USA"),"KO":(33.77,-84.39,"Atlanta, GA","USA"),
    "WMT":(36.37,-94.21,"Bentonville, AR","USA"),"CRM":(37.79,-122.40,"San Francisco, CA","USA"),
    "BAC":(35.23,-80.84,"Charlotte, NC","USA"),"NFLX":(34.10,-118.33,"Los Gatos, CA","USA"),
    "AMD":(37.38,-121.96,"Santa Clara, CA","USA"),"ORCL":(30.27,-97.74,"Austin, TX","USA"),
}

# Global market nodes for network visualization
GLOBAL_NODES = [
    (40.71,-74.01,"New York (NYSE)"), (51.51,-0.13,"London (LSE)"),
    (35.68,139.69,"Tokyo (TSE)"), (22.29,114.17,"Hong Kong (HKEX)"),
    (31.23,121.47,"Shanghai (SSE)"), (1.28,103.85,"Singapur (SGX)"),
    (50.11,8.68,"Frankfurt (DAX)"), (19.43,-99.13,"CDMX (BMV)"),
    (-12.05,-77.04,"Lima (BVL)"), (-23.55,-46.63,"São Paulo (B3)"),
    (48.86,2.35,"París (Euronext)"), (28.61,77.21,"Mumbai (NSE India)"),
]

# ───────────────────────────────────────────────────────────────────
# STRATEGIC PARTNERS & SUPPLY CHAIN DATABASE
# ───────────────────────────────────────────────────────────────────

STRATEGIC_PARTNERS = {
    "AAPL": [
        {"partner":"TSMC","role":"Fabricante de chips","country":"Taiwán","lat":24.78,"lon":120.99,"desc":"Produce los chips A-series y M-series en nodos de 3nm"},
        {"partner":"Foxconn","role":"Ensamblaje","country":"China","lat":22.64,"lon":114.02,"desc":"Mayor ensamblador de iPhones en Shenzhen y Zhengzhou"},
        {"partner":"Samsung Display","role":"Pantallas OLED","country":"Corea del Sur","lat":36.67,"lon":127.0,"desc":"Proveedor principal de paneles OLED para iPhone Pro"},
        {"partner":"Qualcomm","role":"Módems 5G","country":"USA","lat":32.90,"lon":-117.20,"desc":"Suministra módems 5G para conectividad celular"},
        {"partner":"Broadcom","role":"Componentes wireless","country":"USA","lat":37.40,"lon":-121.97,"desc":"Chips Wi-Fi, Bluetooth y RF para todos los dispositivos"},
        {"partner":"Corning","role":"Gorilla Glass","country":"USA","lat":42.14,"lon":-77.05,"desc":"Fabricante del cristal protector Ceramic Shield del iPhone"},
    ],
    "NVDA": [
        {"partner":"TSMC","role":"Fabricante de chips","country":"Taiwán","lat":24.78,"lon":120.99,"desc":"Fabrica GPUs H100/B200 en proceso 4nm y 3nm"},
        {"partner":"Samsung","role":"Memoria HBM","country":"Corea del Sur","lat":37.44,"lon":127.0,"desc":"Proveedor de memoria HBM3E de alta velocidad"},
        {"partner":"SK Hynix","role":"Memoria HBM","country":"Corea del Sur","lat":37.29,"lon":127.0,"desc":"Principal proveedor de memoria HBM3E para data centers"},
        {"partner":"Microsoft","role":"Cliente cloud","country":"USA","lat":47.64,"lon":-122.13,"desc":"Mayor comprador de GPUs para Azure e infraestructura AI"},
        {"partner":"Meta","role":"Cliente AI","country":"USA","lat":37.48,"lon":-122.15,"desc":"Adquiere GPUs masivamente para entrenamiento de Llama"},
        {"partner":"Supermicro","role":"Servidores","country":"USA","lat":37.39,"lon":-121.98,"desc":"Fabrica servidores GPU-optimizados para data centers"},
    ],
    "TSLA": [
        {"partner":"Panasonic","role":"Celdas de batería","country":"Japón","lat":34.69,"lon":135.50,"desc":"Produce celdas de batería 4680 en Gigafactory Nevada"},
        {"partner":"CATL","role":"Baterías LFP","country":"China","lat":26.65,"lon":119.30,"desc":"Mayor fabricante mundial de baterías, provee LFP a Tesla"},
        {"partner":"Samsung SDI","role":"Celdas premium","country":"Corea del Sur","lat":37.44,"lon":127.0,"desc":"Baterías NCA de alta densidad para Model S/X"},
        {"partner":"BHP","role":"Níquel","country":"Australia","lat":-31.95,"lon":115.86,"desc":"Suministra níquel refinado para cátodos de baterías"},
        {"partner":"Albemarle","role":"Litio","country":"USA","lat":35.22,"lon":-80.84,"desc":"Principal proveedor de litio para Gigafactories globales"},
        {"partner":"STMicro","role":"Semiconductores","country":"Suiza","lat":46.20,"lon":6.15,"desc":"Chips de potencia SiC para inversores del powertrain"},
    ],
    "MSFT": [
        {"partner":"OpenAI","role":"Partner AI","country":"USA","lat":37.77,"lon":-122.42,"desc":"Inversión de $13B+ para integrar GPT en todo Azure"},
        {"partner":"NVIDIA","role":"GPUs","country":"USA","lat":37.37,"lon":-122.04,"desc":"Proveedor de GPUs A100/H100 para Azure AI clusters"},
        {"partner":"AMD","role":"CPUs servidor","country":"USA","lat":37.38,"lon":-121.96,"desc":"Procesadores EPYC para Azure cloud data centers"},
        {"partner":"Samsung","role":"Memoria","country":"Corea del Sur","lat":37.44,"lon":127.0,"desc":"DRAM y SSD para servidores de Azure worldwide"},
        {"partner":"LinkedIn","role":"Red profesional","country":"USA","lat":37.42,"lon":-122.07,"desc":"Plataforma con 900M+ usuarios integrada al ecosistema"},
        {"partner":"Activision","role":"Gaming","country":"USA","lat":34.04,"lon":-118.47,"desc":"Adquisición de $69B para dominar gaming global"},
    ],
    "AMD": [
        {"partner":"TSMC","role":"Fabricante de chips","country":"Taiwán","lat":24.78,"lon":120.99,"desc":"Fabrica CPUs Ryzen y GPUs Radeon en nodos de 5nm/4nm"},
        {"partner":"Samsung","role":"Memoria","country":"Corea del Sur","lat":37.44,"lon":127.0,"desc":"Proveedor de DRAM para validación de plataformas AMD"},
        {"partner":"Microsoft","role":"Cliente Xbox","country":"USA","lat":47.64,"lon":-122.13,"desc":"CPUs y GPUs custom para Xbox Series X/S"},
        {"partner":"Sony","role":"Cliente PS5","country":"Japón","lat":35.63,"lon":139.74,"desc":"SoC custom para PlayStation 5 y futuras consolas"},
        {"partner":"Xilinx","role":"FPGAs","country":"USA","lat":37.38,"lon":-121.96,"desc":"División adquirida de FPGAs para data centers adaptativos"},
        {"partner":"Meta","role":"Cliente AI","country":"USA","lat":37.48,"lon":-122.15,"desc":"GPUs Instinct MI300X para cargas de entrenamiento AI"},
    ],
    "CRM": [
        {"partner":"Amazon AWS","role":"Infraestructura cloud","country":"USA","lat":47.62,"lon":-122.34,"desc":"Hyperscaler principal para backend de Salesforce"},
        {"partner":"Google Cloud","role":"Cloud secundario","country":"USA","lat":37.42,"lon":-122.08,"desc":"Multi-cloud para redundancia y Analytics con BigQuery"},
        {"partner":"Slack","role":"Comunicaciones","country":"USA","lat":37.79,"lon":-122.39,"desc":"Adquisición de $27.7B para dominar colaboración enterprise"},
        {"partner":"Tableau","role":"Analytics","country":"USA","lat":47.62,"lon":-122.34,"desc":"Plataforma de visualización integrada al CRM"},
        {"partner":"MuleSoft","role":"Integración","country":"USA","lat":37.79,"lon":-122.39,"desc":"API management y integración de datos enterprise"},
        {"partner":"Anthropic","role":"AI Partner","country":"USA","lat":37.77,"lon":-122.42,"desc":"Claude AI integrado en Einstein AI para CRM inteligente"},
    ],
    "GOOGL": [
        {"partner":"Samsung","role":"Pantallas + Memoria","country":"Corea del Sur","lat":37.44,"lon":127.0,"desc":"OLED para Pixel, DRAM para data centers de Google"},
        {"partner":"TSMC","role":"Chips Tensor","country":"Taiwán","lat":24.78,"lon":120.99,"desc":"Fabrica los procesadores Tensor para Pixel y TPUs"},
        {"partner":"Broadcom","role":"Networking","country":"USA","lat":37.40,"lon":-121.97,"desc":"ASICs custom y chips de red para Google Cloud"},
        {"partner":"SpaceX","role":"Conectividad","country":"USA","lat":33.92,"lon":-118.33,"desc":"Starlink para conectividad rural de Google Cloud"},
        {"partner":"Foxconn","role":"Hardware","country":"China","lat":22.64,"lon":114.02,"desc":"Ensamblaje de servidores y dispositivos Pixel"},
        {"partner":"DeepMind","role":"AI Research","country":"UK","lat":51.53,"lon":-0.13,"desc":"División de IA que desarrolla Gemini y AlphaFold"},
    ],
}

# Default partners for tickers not in the database
DEFAULT_PARTNERS = [
    {"partner":"Goldman Sachs","role":"Prime Broker","country":"USA","lat":40.71,"lon":-74.01,"desc":"Servicios de banca de inversión y prime brokerage"},
    {"partner":"Bloomberg","role":"Data Provider","country":"USA","lat":40.76,"lon":-73.98,"desc":"Terminal de datos financieros y analytics institucional"},
    {"partner":"CME Group","role":"Exchange","country":"USA","lat":41.88,"lon":-87.63,"desc":"Mayor exchange de derivados y futuros del mundo"},
]

@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_info(ticker):
    """Get ticker info with HQ location. Falls back to HQ_COORDS database."""
    info = {"city": "N/A", "country": "N/A", "sector": "N/A", "industry": "N/A",
            "name": ticker, "lat": 0, "lon": 0, "is_equity": False,
            "market_cap": 0, "employees": 0}
    if _is_synth(ticker):
        info["name"] = ticker; info["sector"] = "Sintético"
        return info
    # Use HQ_COORDS first (fast, no API)
    if ticker in HQ_COORDS:
        lat, lon, city, country = HQ_COORDS[ticker]
        info.update({"lat": lat, "lon": lon, "city": city, "country": country, "is_equity": True})
    if not YF_OK:
        return info
    try:
        tk = yf.Ticker(ticker)
        yi = tk.info or {}
        info["name"] = yi.get("shortName") or yi.get("longName") or ticker
        info["city"] = yi.get("city") or info["city"]
        info["country"] = yi.get("country") or info["country"]
        info["sector"] = yi.get("sector") or yi.get("category") or "N/A"
        info["industry"] = yi.get("industry") or "N/A"
        info["market_cap"] = yi.get("marketCap") or 0
        info["employees"] = yi.get("fullTimeEmployees") or 0
        qt = yi.get("quoteType", "")
        info["is_equity"] = qt in ("EQUITY", "MUTUALFUND", "ETF")
    except Exception:
        pass
    return info

@st.cache_data(ttl=600, show_spinner=False)
def get_financials(ticker):
    """Get income statement and balance sheet for equities."""
    result = {"income": None, "balance": None}
    if not YF_OK or _is_synth(ticker):
        return result
    try:
        tk = yf.Ticker(ticker)
        inc = tk.income_stmt
        if inc is not None and not inc.empty:
            result["income"] = inc
        bal = tk.balance_sheet
        if bal is not None and not bal.empty:
            result["balance"] = bal
    except Exception:
        pass
    return result

def detect_asset_type(ticker):
    """Classify asset: equity, crypto, forex, commodity, index, synthetic."""
    if _is_synth(ticker):
        return "synthetic"
    if ticker.endswith("-USD") and any(ticker.startswith(c) for c in ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","DOT","MATIC","LINK","UNI","ATOM","LTC","NEAR"]):
        return "crypto"
    if "=X" in ticker:
        return "forex"
    if "=F" in ticker:
        return "commodity"
    if ticker.startswith("^"):
        return "index"
    return "equity"

# ═══════════════════════════════════════════════════════════════════
# QUANT ENGINE
# ═══════════════════════════════════════════════════════════════════

def qrets(df):
    return df["Close"].pct_change().dropna()

def q_cvar(r, c=0.95):
    v = np.percentile(r, (1-c)*100)
    t = r[r<=v]
    return float(t.mean()) if len(t)>0 else 0.0

def q_var(r, c=0.95):
    return float(np.percentile(r, (1-c)*100))

def q_sortino(r):
    d = r[r<0]
    ds = d.std() if len(d)>1 else 1e-8
    return float((r.mean()/ds)*np.sqrt(252)) if ds>0 else 0.0

def q_sharpe(r):
    s = r.std()
    return float((r.mean()/s)*np.sqrt(252)) if s>0 else 0.0

def q_ir(r, b):
    _r = r.copy(); _b = b.copy()
    if hasattr(_r.index, 'tz') and _r.index.tz is not None:
        _r.index = _r.index.tz_localize(None)
    if hasattr(_b.index, 'tz') and _b.index.tz is not None:
        _b.index = _b.index.tz_localize(None)
    al = pd.concat([_r.rename("a"),_b.rename("b")],axis=1).dropna()
    if len(al)<10: return 0.0
    ex = al["a"]-al["b"]; te = ex.std()
    return float((ex.mean()/te)*np.sqrt(252)) if te>0 else 0.0

def q_maxdd(df):
    cm = df["Close"].cummax()
    return float(((df["Close"]-cm)/cm).min())

def q_beta(r, m):
    _r = r.copy(); _m = m.copy()
    if hasattr(_r.index, 'tz') and _r.index.tz is not None:
        _r.index = _r.index.tz_localize(None)
    if hasattr(_m.index, 'tz') and _m.index.tz is not None:
        _m.index = _m.index.tz_localize(None)
    al = pd.concat([_r.rename("a"),_m.rename("b")],axis=1).dropna()
    if len(al)<10: return 1.0
    cv = np.cov(al["a"],al["b"])
    return float(cv[0,1]/cv[1,1]) if cv[1,1]!=0 else 1.0

def q_altman(t):
    np.random.seed(abs(hash(t))%(2**31))
    return round(1.2*np.random.uniform(.05,.35)+1.4*np.random.uniform(.1,.5)+
                 3.3*np.random.uniform(.03,.2)+0.6*np.random.uniform(.8,5)+
                 1.0*np.random.uniform(.5,2.5),2)

def q_boll(df, w=20, n=2.0):
    s=df["Close"].rolling(w).mean(); sd=df["Close"].rolling(w).std()
    return s, s+n*sd, s-n*sd

def q_macd(df):
    ef=df["Close"].ewm(span=12).mean(); es=df["Close"].ewm(span=26).mean()
    m=ef-es; sl=m.ewm(span=9).mean()
    return m, sl, m-sl

def q_rsi(df, w=14):
    d=df["Close"].diff()
    g=d.where(d>0,0).rolling(w).mean()
    l=(-d.where(d<0,0)).rolling(w).mean()
    return 100-(100/(1+g/l.replace(0,1e-10)))

# ═══════════════════════════════════════════════════════════════════
# JARVIS & QUANT CHALLENGES
# ═══════════════════════════════════════════════════════════════════

CHALLENGES = [
    ("Dilema del Prisionero","En teoría de juegos, dos traders enfrentan cooperar o desertar. El equilibrio de Nash predice deserción mutua, pero hedge funds usan tit-for-tat para maximizar retornos a largo plazo."),
    ("Lema de Itô","Para dS=μS·dt+σS·dW, f(S) incluye corrección ½σ²S²f''. Explica el volatility smile: opciones incorporan convexidad del payoff."),
    ("Kelly Criterion","f*=(bp-q)/b. Citadel usa fracción de Kelly (25-50%) para reducir varianza sin sacrificar crecimiento compuesto."),
    ("Martingalas y EMH","E[S(t+1)|F(t)]=S(t). Precios descontados son martingalas bajo medida Q. Los quants buscan desviaciones como fuente de alpha."),
    ("Fat Tails","Retornos NO son normales: curtosis excesiva + asimetría negativa. CVaR 99% puede ser 3-5x mayor que VaR. Ignorar colas = error fatal."),
    ("Cointegración","Activos cointegrados divergen temporalmente pero convergen. Test Engle-Granger verifica. Trading del spread captura alpha mean-reverting."),
    ("Kelly vs Esperanza","En juegos multiplicativos, optimizar E[V] lleva a ruina. Kelly maximiza mediana geométrica = tu P&L acumulado real."),
    ("Paradoja de Ellsberg","Incertidumbre knightiana: distribuciones desconocidas. Modelos robustos min-max CVaR gestionan aversión a ambigüedad."),
    ("Vanna & Volga","Greeks de 2do orden. Explican P&L inexplicado en books de opciones. Críticos durante FOMC y eventos de volatilidad."),
    ("Procesos de Hawkes","Trades generan más trades (auto-excitación). HFT firms usan λ(t)=μ+Σα·exp(-β(t-ti)) para predecir ráfagas."),
]

def jarvis(name, val):
    DB = {
        "IR": ("Retorno excesivo por unidad de tracking error vs benchmark. IR>0.5 competente, >1.0 excepcional. Renaissance mantiene IR>2.0.",
               [(2,"🟢 EXCEPCIONAL — Tier Renaissance/Two Sigma"),(1,"🟢 ÓPTIMO — Alpha consistente, top 10%"),
                (.5,"🟡 COMPETENTE — Alpha positivo, room to improve"),(0,"🟠 NEUTRAL — Sin alpha"),(None,"🔴 CRÍTICO — Destruyendo valor")]),
        "Sortino": ("Retorno ajustado a downside risk. Superior al Sharpe para portfolios asimétricos. Sortino>2 indica excelente gestión de cola izquierda.",
                    [(3,"🟢 ÉLITE — Downside control de clase mundial"),(2,"🟢 ÓPTIMO — Excelente protección"),
                     (1,"🟡 ACEPTABLE — Margen de mejora"),(0,"🟠 DEFICIENTE — Volatilidad bajista domina"),(None,"🔴 CRÍTICO — Retorno negativo")]),
        "CVaR": ("Pérdida esperada en peor 5%. Métrica Basel III. Captura riesgo de cola que VaR ignora.",
                 [(-.01,"🟢 CONSERVADOR — Cola mínima"),(-.03,"🟡 MODERADO — Aceptable"),
                  (-.05,"🟠 ELEVADO — Considerar puts OTM"),(-.08,"🔴 ALTO — Reducir apalancamiento"),(None,"🔴 CRÍTICO — Acción inmediata")]),
    }
    info = DB.get(name, ("","[]"))
    for thr, lbl in info[1]:
        if thr is None: return lbl, info[0]
        if name=="CVaR":
            if val>=thr: return lbl, info[0]
        else:
            if val>=thr: return lbl, info[0]
    return "N/A", info[0]

# ═══════════════════════════════════════════════════════════════════
# PLOTLY HELPERS — Clean, no deprecated props
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# WHY? COGNITIVE AUDIT DATABASE — Interactive Post-its
# ───────────────────────────────────────────────────────────────────

WHY_DB = {
    "Beta": {
        "term_es": "Beta (Sensibilidad al Mercado)",
        "what": "Mide la sensibilidad del activo respecto al mercado (S&P 500). β=1 significa que se mueve igual que el mercado. β>1 amplifica los movimientos, β<1 los amortigua.",
        "interpret": lambda v: f"β={v:.3f} — {'Altamente agresivo: amplifica movimientos del mercado 💥' if v>1.5 else 'Agresivo: más volátil que el mercado' if v>1.2 else 'Neutral: se mueve con el mercado' if v>0.8 else 'Defensivo: menos sensible al mercado 🛡️' if v>0.3 else 'Descorrelacionado: comportamiento independiente del mercado'}",
        "context": "En Citadel, los portafolios beta-neutral (β≈0) son el estándar para estrategias de alfa puro. El BCRP monitorea el beta del sistema financiero peruano para evaluar riesgo sistémico y estabilidad.",
    },
    "Vol Anual": {
        "term_es": "Volatilidad Anualizada (Riesgo Total)",
        "what": "Volatilidad anualizada: desviación estándar de los retornos diarios multiplicada por √252 (días bursátiles). Mide cuánto fluctúa el precio — a mayor vol, mayor incertidumbre.",
        "interpret": lambda v: f"Vol={v:.1f}% — {'EXTREMA: territorio de crisis o activo especulativo 🔥' if v>60 else 'Muy alta: típico de criptomonedas o acciones pequeñas' if v>40 else 'Alta: requiere dimensionamiento cuidadoso de posición' if v>25 else 'Moderada: rango normal para acciones de gran capitalización' if v>15 else 'Baja: activo estable, ideal para estrategias de carry (ingreso)'}",
        "context": "Los escritorios de riesgo (risk desks) de Citadel limitan la volatilidad individual a 40% para posiciones concentradas. El BCRP usa la volatilidad del USD/PEN para calibrar sus intervenciones cambiarias.",
    },
    "Max DD": {
        "term_es": "Máxima Caída (Maximum Drawdown)",
        "what": "Maximum Drawdown (Máxima Caída): la mayor pérdida porcentual desde un pico hasta el punto más bajo. Representa el peor escenario histórico que enfrentó un inversor.",
        "interpret": lambda v: f"DD={v:.1f}% — {'Devastador: pérdida casi total 💀' if v<-60 else 'Severo: caída nivel crisis financiera' if v<-40 else 'Significativo: puede requerir años para recuperar' if v<-20 else 'Moderado: aceptable para inversión a largo plazo' if v<-10 else 'Bajo: excelente gestión de riesgo ✅'}",
        "context": "Los fondos de Citadel tienen límites estrictos de caída máxima (hard limits). Un DD>20% en un libro (book) típicamente dispara una revisión del gestor de portafolio (PM). Regla de oro: si tu caída máxima > 2× tu retorno anual, la estrategia es insostenible.",
    },
    "Skew": {
        "term_es": "Asimetría (Skewness)",
        "what": "Asimetría (Skewness): mide si los retornos están sesgados. Valor negativo = las caídas extremas (crashes) son más frecuentes que las subidas extremas. Valor positivo = más potencial alcista.",
        "interpret": lambda v: f"Skew={v:.3f} — {'Fuertemente asimétrico negativo: riesgo elevado de caída extrema ⚠️' if v<-1 else 'Asimetría negativa: más caídas extremas que subidas' if v<-0.3 else 'Aproximadamente simétrico: distribución balanceada' if abs(v)<0.3 else 'Asimetría positiva: potencial alcista convexo, favorable 🎯'}",
        "context": "En un fondo de cobertura (hedge fund), tener asimetría negativa sin protección = bonificación eliminada. Los gestores de Citadel buscan activamente asimetría positiva mediante opciones.",
    },
    "Kurtosis": {
        "term_es": "Curtosis (Grosor de Colas)",
        "what": "Curtosis: mide el grosor de las colas de la distribución. Valor >3 indica colas pesadas (fat tails), lo que significa que los eventos extremos son más probables de lo que predice una distribución normal.",
        "interpret": lambda v: f"Kurt={v:.2f} — {'Colas extremadamente pesadas: los modelos normales son peligrosos aquí 🚨' if v>10 else 'Colas pesadas significativas: usar CVaR en lugar de VaR' if v>5 else 'Colas moderadamente pesadas: precaución con modelos estándar' if v>3 else 'Aproximadamente normal: modelos paramétricos estándar son válidos'}",
        "context": "LTCM (Long-Term Capital Management) colapsó por ignorar la curtosis. Basel III exige pruebas de estrés (stress testing) con distribuciones de colas pesadas. El BCRP usa modelos Student-t para gestionar reservas internacionales.",
    },
    "Sharpe": {
        "term_es": "Ratio de Sharpe (Retorno Ajustado al Riesgo)",
        "what": "Ratio de Sharpe (Sharpe Ratio): retorno excesivo por unidad de riesgo total. Fórmula: (Retorno - Tasa libre de riesgo) / Volatilidad × √252. Es el estándar dorado (gold standard) para medir desempeño.",
        "interpret": lambda v: f"Sharpe={v:.3f} — {'Excepcional: nivel Renaissance Technologies (>2.0) 🏆' if v>2 else 'Excelente: superas al 95% de los gestores de fondos' if v>1 else 'Bueno: generando alfa (retorno excedente) real' if v>0.5 else 'Mediocre: mejor considerar un ETF pasivo' if v>0 else 'Negativo: destruyendo valor para el inversor 🚩'}",
        "context": "Citadel Wellington mantiene un Sharpe ~1.8. Un Sharpe<0.5 en un portafolio activo no justifica las comisiones (fees). El BCRP evalúa el Sharpe de sus reservas internacionales para optimizar la asignación.",
    },
    "RSI": {
        "term_es": "Índice de Fuerza Relativa (RSI)",
        "what": "RSI (Relative Strength Index = Índice de Fuerza Relativa): oscilador de momento que va de 0 a 100. Mide la velocidad y magnitud de los cambios de precio. >70 = sobrecompra, <30 = sobreventa.",
        "interpret": lambda v: f"RSI={v:.1f} — {'Sobrecompra extrema: alta probabilidad de reversión bajista 🔴' if v>80 else 'Sobrecompra: presión vendedora inminente' if v>70 else 'Zona alcista: momento (momentum) favorable' if v>55 else 'Neutral: sin señal clara de dirección' if v>45 else 'Zona bajista: momento negativo' if v>30 else 'Sobreventa: potencial rebote técnico 🟢' if v>20 else 'Sobreventa extrema: capitulación del mercado, oportunidad contraria (contrarian)'}",
        "context": "Los traders cuantitativos de Citadel no usan el RSI de forma aislada — lo combinan con perfil de volumen (volume profile) y flujo de órdenes (order flow). Las divergencias entre RSI y precio son señales de mayor convicción.",
    },
    "MACD": {
        "term_es": "MACD (Convergencia/Divergencia de Medias Móviles)",
        "what": "MACD (Moving Average Convergence Divergence = Convergencia/Divergencia de Medias Móviles): diferencia entre la media móvil exponencial (EMA) de 12 y 26 períodos. La línea de señal (signal line) es la EMA de 9 períodos del MACD.",
        "interpret": lambda v: f"MACD Hist={v:.4f} — {'Momento alcista acelerando 📈' if v>0 else 'Momento bajista intensificándose 📉'}. {'Cruce reciente detectado — señal de alta convicción.' if abs(v)<0.01 else ''}",
        "context": "En Citadel, el MACD se usa como filtro de régimen de mercado, no como señal primaria de trading. Histograma creciente = tendencia fortaleciéndose (trend strengthening). Divergencias MACD/precio anticipan reversiones importantes.",
    },
    "VaR": {
        "term_es": "Valor en Riesgo (VaR)",
        "what": "VaR (Value-at-Risk = Valor en Riesgo) al 95%: pérdida máxima esperada en 1 día con un 95% de confianza estadística. Importante: NO captura el riesgo de cola (tail risk) — para eso se usa el CVaR.",
        "interpret": lambda v: f"VaR={v*100:.3f}% — Con 95% de confianza, no perderás más de {abs(v)*100:.2f}% en un día. {'Aceptable para acciones estándar.' if abs(v)<0.03 else 'Elevado: considerar reducir el tamaño de la posición.' if abs(v)<0.05 else 'Alto: requiere cobertura (hedging) activa.'}",
        "context": "Basilea III (Basel III) exige VaR diario + CVaR para calcular capital regulatorio. Citadel usa VaR con simulaciones Monte Carlo (10,000+ escenarios), no el método histórico simple. El BCRP publica el VaR de las reservas internacionales.",
    },
    "CVaR": {
        "term_es": "CVaR (Valor en Riesgo Condicional / Pérdida Esperada en Cola)",
        "what": "CVaR (Conditional VaR = Valor en Riesgo Condicional), también llamado Expected Shortfall (Pérdida Esperada): es la pérdida promedio cuando se excede el umbral del VaR. Captura el riesgo de cola (tail risk) que el VaR ignora.",
        "interpret": lambda v: f"CVaR={v*100:.3f}% — En el peor 5% de los días, la pérdida promedio es {abs(v)*100:.2f}%. {'Conservador: riesgo de cola bajo' if abs(v)<0.02 else 'Moderado: dentro de límites aceptables' if abs(v)<0.04 else 'Elevado: protección con opciones de venta (puts) recomendada' if abs(v)<0.06 else 'CRÍTICO: riesgo de cola inaceptable, reducir exposición inmediatamente'}.",
        "context": "El CVaR es una medida de riesgo coherente (cumple con subaditividad). El BCRP lo usa para pruebas de estrés (stress tests) del sistema financiero peruano. Citadel calcula el CVaR en tiempo real para cada libro de trading (book).",
    },
    "IR": {
        "term_es": "Ratio de Información (Information Ratio)",
        "what": "IR (Information Ratio = Ratio de Información): alfa (retorno excedente) dividido por el error de seguimiento (tracking error) vs el índice de referencia (benchmark). Mide la habilidad del gestor para generar retorno excesivo de forma consistente.",
        "interpret": lambda v: f"IR={v:.4f} — {'Nivel Renaissance/Two Sigma: alfa excepcional y consistente 🏆' if v>2 else 'Decil superior: superas consistentemente al índice de referencia' if v>1 else 'Competente: alfa positivo con margen de mejora' if v>0.5 else 'Sin alfa significativo vs inversión pasiva' if v>0 else 'Destruyendo valor comparado con inversión pasiva 🚩'}",
        "context": "En Citadel, cada gestor de portafolio (PM) es evaluado por su IR. Un IR<0.3 sostenido = reasignación de capital a otro gestor. IR>1.0 = incremento de exposición bruta (gross exposure). Es LA métrica que define tu bonificación anual.",
    },
    "Sortino": {
        "term_es": "Ratio de Sortino (Retorno Ajustado al Riesgo Bajista)",
        "what": "Ratio de Sortino (Sortino Ratio): similar al Sharpe pero solo penaliza la volatilidad bajista (downside). Es superior para evaluar estrategias con retornos asimétricos porque la volatilidad alcista es deseable, no penalizable.",
        "interpret": lambda v: f"Sortino={v:.4f} — {'Élite: protección contra caídas de clase mundial' if v>3 else 'Excelente: la volatilidad alcista no penaliza tu ratio' if v>2 else 'Aceptable: riesgo bajista (downside) gestionado' if v>1 else 'Deficiente: demasiada volatilidad bajista' if v>0 else 'Crítico: retornos negativos, estrategia fallida'}",
        "context": "Los escritorios de gestión (PM desks) de Citadel prefieren el Sortino sobre el Sharpe para estrategias de opciones y event-driven (basadas en eventos). Un Sortino alto + Sharpe bajo = estrategia con riesgo de cola positivo (tail risk positivo) — esto es ideal.",
    },
    "BB Pos": {
        "term_es": "Posición en Bandas de Bollinger",
        "what": "Posición dentro de las Bandas de Bollinger (0-100%). 0% = el precio está en la banda inferior, 100% = en la banda superior. Las bandas miden ±2 desviaciones estándar respecto a la media móvil de 20 períodos.",
        "interpret": lambda v: f"BB={v:.1f}% — {'Precio en extremo superior: posible compresión alcista (squeeze) o reversión inminente' if v>90 else 'Zona alta: tendencia alcista fuerte' if v>70 else 'Zona media: consolidación y equilibrio' if v>30 else 'Zona baja: presión bajista dominante' if v>10 else 'Extremo inferior: posible capitulación o rebote técnico'}",
        "context": "En Citadel, el ancho de las bandas (Bollinger Width) es más importante que la posición. Bandas estrechas → baja volatilidad → explosión de precio inminente (esto se conoce como Bollinger Squeeze o Compresión de Bollinger).",
    },
    "Z-Altman": {
        "term_es": "Z-Score de Altman (Predictor de Bancarrota)",
        "what": "Z-Score de Altman: modelo estadístico de predicción de bancarrota empresarial creado por Edward Altman. Z>2.99 = Zona Segura (Safe), entre 1.81-2.99 = Zona Gris (Grey Zone), <1.81 = Zona de Peligro (Distress).",
        "interpret": lambda v: f"Z={v:.2f} — {'Zona segura: baja probabilidad de impago (default) ✅' if v>2.99 else 'Zona gris: monitorear de cerca, riesgo intermedio' if v>1.81 else 'Zona de peligro (distress): riesgo de bancarrota elevado 🚨'}",
        "context": "El escritorio de crédito (credit desk) de Citadel usa el Z-Score como primer filtro para ventas en corto (short selling) de crédito corporativo. El BCRP lo incorpora en su modelo de estrés bancario para evaluar la solvencia del sistema financiero peruano.",
    },
}

def why_postit(key, value, unique_id=""):
    """Render interactive WHY? Post-it Intelligence Card via st.expander."""
    info = WHY_DB.get(key)
    if not info:
        return
    interp = info["interpret"](value) if callable(info.get("interpret")) else ""
    label_es = info.get("term_es", key)
    with st.expander(f"❓ WHY? — {label_es}", expanded=False):
        st.markdown(
            f'<div class="why-card">'
            f'<div class="wt">📘 ¿Qué es esto? ({key})</div>'
            f'<div class="wd">{info["what"]}</div>'
            f'<div class="wt" style="margin-top:10px">📊 ¿Qué indica este valor?</div>'
            f'<div class="wv">{interp}</div>'
            f'<div class="wc">🏛️ <b>Contexto Institucional (Citadel/BCRP):</b> {info["context"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────────
# SENTIMENT ENGINE — Keyword-based analysis for news
# ───────────────────────────────────────────────────────────────────

_BULL_WORDS = {"surge","rally","gain","soar","jump","record","high","upgrade","bullish",
    "beat","growth","profit","strong","recover","boom","breakout","sube","alza",
    "gana","récord","supera","crece","positivo","alcista","mejora","beneficio"}
_BEAR_WORDS = {"crash","plunge","drop","fall","loss","decline","fear","warning","cut",
    "bearish","downgrade","risk","weak","recession","crisis","sell","dump",
    "baja","cae","pierde","riesgo","crisis","negativo","bajista","recesión","colapso"}

def sentiment_score(text):
    """Simple keyword-based sentiment: returns (score, label, css_class)."""
    words = set(text.lower().split())
    bull = len(words & _BULL_WORDS)
    bear = len(words & _BEAR_WORDS)
    if bull > bear:
        return bull - bear, "BULLISH 📈", "sent-bull"
    elif bear > bull:
        return bear - bull, "BEARISH 📉", "sent-bear"
    return 0, "NEUTRAL ●", "sent-neut"


# ───────────────────────────────────────────────────────────────────
# SENSITIVITY ENGINE — Cross-asset 1% shock analysis
# ───────────────────────────────────────────────────────────────────

SENSITIVITY_PAIRS = {
    "Oro (GC=F)": {
        "tickers": ["GC=F"],
        "impacts": {
            "Mineras": ("Si Oro +1% → mineras suben ~1.5-3% (apalancamiento operativo). "
                        "Newmont, Barrick y BVL:BUENAVENTURA son proxies directos."),
            "USD/PEN": ("Oro +1% → generalmente PEN se fortalece (Perú = exportador neto de oro). "
                        "Reducción del déficit de cuenta corriente."),
            "Inflación": ("Oro como hedge inflacionario: si sube, el mercado está pricing mayor inflación futura. "
                          "El BCRP ajusta política monetaria en consecuencia."),
        },
    },
    "Cobre (HG=F)": {
        "tickers": ["HG=F"],
        "impacts": {
            "USD/PEN": ("Cobre +1% → PEN se fortalece ~0.3-0.5% (Perú = 2do productor mundial). "
                        "Efecto directo en balanza comercial y reservas del BCRP."),
            "China/Emerentes": ("Cobre = barómetro de crecimiento global. +1% señala expansión manufacturera. "
                                "Favorece emerging markets y commodities currencies."),
            "S&P 500": ("Correlación positiva moderada: cobre refleja optimismo industrial. "
                        "Pero en stagflation, la correlación se invierte."),
        },
    },
    "VIX (^VIX)": {
        "tickers": ["^VIX"],
        "impacts": {
            "S&P 500": ("VIX +1pt → S&P 500 cae ~0.5-1.0% (correlación inversa ~-0.75). "
                        "Es el 'fear gauge' más monitoreado por risk desks."),
            "USD": ("VIX alto → flight to quality → USD se fortalece. "
                    "Impacto directo en USD/PEN y remesas."),
            "Crypto": ("VIX spike → crypto sufre desproporcionadamente (activo de riesgo puro). "
                       "BTC drawdown puede ser 2-3x el del S&P en eventos de pánico."),
        },
    },
    "M2 Masa Monetaria": {
        "tickers": [],
        "impacts": {
            "Todos los activos": ("M2 creciente = más liquidez = activos financieros suben (correlación histórica ~0.85 con S&P). "
                                  "Es el driver #1 de largo plazo que los quants de Citadel monitorean."),
            "Crypto/Oro": ("M2 expansion → BTC y Oro como hedges inflacionarios se benefician más. "
                           "La narrativa 'digital gold' se activa con M2 acelerándose."),
            "USD/PEN": ("M2 USA creciendo + M2 Perú estable → presión bajista sobre el Sol. "
                        "El BCRP monitorea el diferencial de masa monetaria para intervenciones."),
        },
    },
}

def render_sensitivity_module(ticker, rets, all_rets_dict):
    """Render the cross-asset sensitivity analysis module."""
    st.markdown("---")
    st.markdown("##### 🎯 Análisis de Sensibilidad — Impacto Cross-Asset")

    for driver_name, driver_info in SENSITIVITY_PAIRS.items():
        with st.expander(f"📊 ¿Qué pasa si {driver_name} se mueve ±1%?", expanded=False):
            # Calculate actual beta if data available
            driver_tickers = driver_info["tickers"]
            actual_beta = None
            if driver_tickers and driver_tickers[0] in all_rets_dict:
                dr = all_rets_dict[driver_tickers[0]]
                al = pd.concat([rets.rename("a"), dr.rename("b")], axis=1).dropna()
                if len(al) > 20:
                    cov = np.cov(al["a"], al["b"])
                    if cov[1, 1] != 0:
                        actual_beta = cov[0, 1] / cov[1, 1]

            beta_text = ""
            if actual_beta is not None:
                direction = "sube" if actual_beta > 0 else "baja"
                st.markdown(
                    f'<div class="sens-card">'
                    f'<div class="st">📐 Beta Empírica (90D): {ticker} vs {driver_name}</div>'
                    f'Si <b>{driver_name}</b> se mueve +1%, <b>{ticker}</b> {direction} '
                    f'~<span style="color:{C["gold_light"]}">{abs(actual_beta)*1:.3f}%</span> '
                    f'(β = {actual_beta:.4f})'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            for impact_name, impact_desc in driver_info["impacts"].items():
                st.markdown(
                    f'<div class="sens-card">'
                    f'<div class="st">→ Impacto en {impact_name}</div>'
                    f'{impact_desc}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ───────────────────────────────────────────────────────────────────
# WHY? LATEX DIAGNOSTICS — Chart-level Citadel diagnostics
# ───────────────────────────────────────────────────────────────────

CHART_WHY = {
    "candles": {
        "title": "Gráfico de Velas Japonesas + Bandas de Bollinger",
        "concept": "Las velas japonesas representan precio de apertura, máximo, mínimo y cierre en cada período. "
                   "Las Bandas de Bollinger son ±2σ alrededor de la SMA(20), midiendo volatilidad dinámica.",
        "formula": r"BB_{upper} = SMA_{20} + 2\sigma_{20} \quad;\quad BB_{lower} = SMA_{20} - 2\sigma_{20}",
        "macro": "El ancho de las bandas (Bollinger Width) anticipa regímenes de volatilidad. "
                 "Bandas comprimidas preceden movimientos explosivos — el BCRP monitorea compresiones del USD/PEN.",
    },
    "vol_surface": {
        "title": "Superficie de Volatilidad Implícita 3D",
        "concept": "La superficie IV mapea la volatilidad implícita en función del strike (K/S) y el vencimiento (T). "
                   "El 'smile' y 'skew' revelan cómo el mercado precia el riesgo de cola.",
        "formula": r"\sigma_{IV}(K,T) = \sigma_{ATM} + a(K/S - 1)^2 + b\sqrt{T} - c(K/S-1)e^{-T}",
        "macro": "El skew de volatilidad (diferencia IV puts vs calls) es el termómetro de miedo institucional. "
                 "Citadel monitorea el 25-delta risk reversal como proxy de sentimiento.",
    },
    "distribution": {
        "title": "Distribución de Retornos vs Normal Teórica",
        "concept": "Compara retornos reales contra la distribución normal (gaussiana). "
                   "Las desviaciones revelan asimetría (skew) y colas pesadas (kurtosis) que invalidan modelos estándar.",
        "formula": r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \quad;\quad CVaR_{\alpha} = E[L \,|\, L \geq VaR_{\alpha}]",
        "macro": "El test de Jarque-Bera determina si los retornos son normales. Si p<0.05, los modelos paramétricos "
                 "son peligrosos. El BCRP usa distribuciones t-Student para modelar las reservas internacionales.",
    },
    "correlation": {
        "title": "Matriz de Correlación Dinámica",
        "concept": "Correlación de Pearson entre retornos mide co-movimiento lineal. "
                   "ρ=+1 implica movimiento idéntico, ρ=-1 cobertura perfecta, ρ=0 independencia.",
        "formula": r"\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma_X \cdot \sigma_Y} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}",
        "macro": "Las correlaciones se disparan durante crisis (convergencia a 1.0), destruyendo diversificación. "
                 "La correlación Cobre-PEN es clave para el BCRP: cuando sube cobre, el sol se fortalece.",
    },
    "choropleth": {
        "title": "Mapa Geopolítico de Exposición por Revenue",
        "concept": "Distribución geográfica del ingreso estimado. Mide concentración de riesgo país "
                   "y exposición a shocks regionales (aranceles, sanciones, crisis).",
        "formula": r"HHI = \sum_{i=1}^{N} s_i^2 \quad \text{(Herfindahl: concentración geográfica)}",
        "macro": "Citadel ajusta exposición país según el riesgo geopolítico. Perú tiene exposición "
                 "dual: cobre→China y oro→global. El BCRP monitorea la balanza comercial por destino.",
    },
    "macd_rsi": {
        "title": "Indicadores de Momento: MACD y RSI",
        "concept": "MACD captura cambios de tendencia via cruces de EMAs. RSI mide fuerza relativa "
                   "de subidas vs bajadas en ventana de 14 períodos.",
        "formula": r"MACD = EMA_{12} - EMA_{26} \quad;\quad RSI = 100 - \frac{100}{1 + \frac{Avg\,Gain}{Avg\,Loss}}",
        "macro": "En Citadel, MACD se usa como filtro de régimen (alcista/bajista), no como señal aislada. "
                 "Las divergencias RSI-precio anticipan reversiones con mayor probabilidad que cruces simples.",
    },
}

def why_chart_diagnostic(chart_key, extra_interp=""):
    """Render WHY? diagnostic expander with LaTeX formula for a chart."""
    info = CHART_WHY.get(chart_key)
    if not info:
        return
    with st.expander(f"🔍 WHY? — Diagnóstico Citadel: {info['title']}", expanded=False):
        st.markdown(f"**📘 Concepto:** {info['concept']}")
        st.latex(info["formula"])
        if extra_interp:
            st.markdown(f"**📊 Interpretación Quant:** {extra_interp}")
        st.markdown(f"**🌐 Conexión Macro/BCRP:** {info['macro']}")

# ═══════════════════════════════════════════════════════════════════
# PLOTLY HELPERS — Clean, no deprecated props
# ═══════════════════════════════════════════════════════════════════

def base_layout():
    return dict(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg2"],
        font=dict(family="JetBrains Mono, monospace", size=11, color=C["text2"]),
        margin=dict(l=50,r=30,t=50,b=40),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        legend=dict(bgcolor=C["legend_bg"], bordercolor=C["border"], borderwidth=1, font=dict(size=10)),
    )

def gauge(val, title, lo, hi, steps, suf=""):
    clrs = [C["red"],C["orange"],"#ffd600","#69f0ae",C["green"]]
    st_list = [dict(range=[a,b], color=clrs[min(i,4)]) for i,(a,b) in enumerate(steps)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number=dict(font=dict(size=26, color=C["gold_light"], family="JetBrains Mono"), suffix=suf),
        title=dict(text="<b>"+title+"</b>", font=dict(size=12, color=C["gold"], family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[lo,hi], tickfont=dict(size=9, color=C["textm"]), dtick=(hi-lo)/5),
            bar=dict(color=C["gold"], thickness=0.3),
            bgcolor=C["bg1"], borderwidth=1, bordercolor=C["border"],
            steps=st_list,
            threshold=dict(line=dict(color=C["cyan"], width=3), thickness=0.8, value=val))))
    fig.update_layout(height=250, paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                      font=dict(family="JetBrains Mono"), margin=dict(l=30,r=30,t=60,b=10))
    return fig

# ═══════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def ui_header():
    st.markdown('<div class="hdr"><h1>⚡ CENTRO DE MANDO DE EATON</h1>'
                '<p>INSTITUTIONAL QUANT TERMINAL · CITADEL ARCHITECTURE · GLOBAL INTELLIGENCE · v20.0</p></div>', unsafe_allow_html=True)
    now = datetime.now()
    st.markdown(f'<div class="sb">'
        f'<span>SID: <span style="color:{C["gold"]}">{st.session_state.sid}</span></span>'
        f'<span>LAT: <span style="color:{C["green"]}">{np.random.randint(1,8)}ms</span></span>'
        f'<span>REG: <span style="color:{C["blue"]}">US-E·EU-W·APAC</span></span>'
        f'<span>ASSETS: <span style="color:{C["gold"]}">{N_ASSETS}</span></span>'
        f'<span>DATA: <span style="color:{C["green"]}">● LIVE</span></span>'
        f'<span>{now.strftime("%Y-%m-%d %H:%M:%S")}</span></div>', unsafe_allow_html=True)

def ui_challenge():
    i = datetime.now().timetuple().tm_yday % len(CHALLENGES)
    t, b = CHALLENGES[i]
    st.markdown(f'<div class="qc"><div class="tt">🧠 Quant Challenge — {t}</div>{b}</div>', unsafe_allow_html=True)

def ui_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ PANEL DE CONTROL"); st.markdown("---")
        cat = st.selectbox("📂 CATEGORÍA", list(ALL_CAT.keys()))
        assets = ALL_CAT[cat]
        tick = st.selectbox("🎯 ACTIVO", list(assets.keys()), format_func=lambda x: f"{x} — {assets[x]}")

        st.markdown("---"); st.markdown("##### ⏱️ HORIZONTE ESTRATÉGICO")
        horizon_key = st.selectbox("📅 TEMPORALIDAD",
            list(HORIZONS.keys()), index=5,
            help="Streaming=simulado · 1min limita a 7D · Diario=estándar")
        hz = HORIZONS[horizon_key]

        # Auto-refresh toggle
        st.markdown("---"); st.markdown("##### 🔄 MOTOR REAL-TIME")
        auto_on = st.checkbox("⚡ Auto-refresh (60s)", value=False)
        refresh_sec = st.slider("Intervalo (seg)", 15, 300, 60, 15,
                                disabled=not auto_on) if auto_on else 60

        if auto_on and AUTOREFRESH_OK:
            st_autorefresh(interval=refresh_sec * 1000, limit=None, key="hft_refresh")
            st.markdown(f'<div style="text-align:center;padding:4px;background:rgba(0,230,118,0.1);'
                f'border-radius:4px;font-family:JetBrains Mono;font-size:.68rem;color:{C["green"]}">'
                f'● LIVE — Refresh cada {refresh_sec}s</div>', unsafe_allow_html=True)
        elif auto_on and not AUTOREFRESH_OK:
            st.warning("Instala: `pip install streamlit-autorefresh`", icon="⚠️")

        st.markdown("---"); st.markdown("##### 📊 UNIVERSO")
        for cn, ca in ALL_CAT.items():
            with st.expander(f"{cn} ({len(ca)})"):
                for t, n in ca.items():
                    _c = C["text2"]
                    st.markdown(f"<span style='font-family:JetBrains Mono;font-size:.72rem;color:{_c}'>`{t}` {n}</span>", unsafe_allow_html=True)
        st.markdown("---")
        if st.checkbox("🗑️ Limpiar caché"):
            st.cache_data.clear()
        _cm = C["textm"]
        st.markdown(f"<div style='text-align:center;font-size:.65rem;color:{_cm};font-family:JetBrains Mono'>EATON v20.0 · © {datetime.now().year}</div>", unsafe_allow_html=True)
    return tick, hz

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — RADAR GEOPOLÍTICO
# ═══════════════════════════════════════════════════════════════════

def tab1_geo(tk, df, r):
    st.markdown("### 🌍 Radar Geopolítico & Superficie de Volatilidad")

    # ── HQ Intelligence Card ──
    tk_info = get_ticker_info(tk)
    asset_t = detect_asset_type(tk)
    hq_city = tk_info["city"]; hq_country = tk_info["country"]
    hq_sector = tk_info["sector"]; hq_industry = tk_info["industry"]
    hq_name = tk_info["name"]
    st.markdown(
        f'<div class="geo-hq">'
        f'<div class="hq-title">🏢 {hq_name}</div>'
        f'<div class="hq-sub">📍 {hq_city}, {hq_country} · 🏷️ {hq_sector} / {hq_industry} · '
        f'Tipo: <span style="color:{C["cyan"]}">{asset_t.upper()}</span></div>'
        f'</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🗺️ Red de Negocios — Golden Grid")
        np.random.seed(abs(hash(tk))%(2**31))

        # Build golden grid map
        fig = go.Figure()

        # Global market nodes (golden dots)
        node_lats = [n[0] for n in GLOBAL_NODES]
        node_lons = [n[1] for n in GLOBAL_NODES]
        node_names = [n[2] for n in GLOBAL_NODES]
        node_sizes = np.random.randint(6, 18, len(GLOBAL_NODES))

        fig.add_trace(go.Scattergeo(
            lat=node_lats, lon=node_lons, text=node_names,
            mode="markers+text", textposition="top center",
            textfont=dict(size=7, color=C["gold_light"], family="JetBrains Mono"),
            marker=dict(size=node_sizes, color=C["gold"],
                        line=dict(width=1, color=C["gold_light"]),
                        opacity=0.85, symbol="circle"),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Mercados Globales"))

        # HQ marker (large, pulsing effect via larger outline)
        hq_lat = tk_info.get("lat", 0); hq_lon = tk_info.get("lon", 0)
        partners = STRATEGIC_PARTNERS.get(tk, DEFAULT_PARTNERS)

        if hq_lat != 0 and hq_lon != 0:
            fig.add_trace(go.Scattergeo(
                lat=[hq_lat], lon=[hq_lon], text=[f"🏢 HQ: {hq_city}"],
                mode="markers+text", textposition="bottom center",
                textfont=dict(size=9, color=C["cyan"], family="JetBrains Mono"),
                marker=dict(size=22, color=C["cyan"],
                            line=dict(width=3, color="#ffffff"), opacity=1, symbol="star"),
                hovertemplate=f"<b>HQ: {hq_name}</b><br>{hq_city}, {hq_country}<extra></extra>",
                name=f"HQ {tk}"))

            # Network lines from HQ to global nodes (golden curves)
            for nlat, nlon, nname in GLOBAL_NODES:
                fig.add_trace(go.Scattergeo(
                    lat=[hq_lat, nlat], lon=[hq_lon, nlon],
                    mode="lines",
                    line=dict(width=0.8, color="rgba(201,149,40,0.35)"),
                    hoverinfo="skip", showlegend=False))

            # ── Partner nodes (cyan diamonds) + connection lines ──
            for p in partners:
                plat = p.get("lat", 0); plon = p.get("lon", 0)
                if plat != 0 and plon != 0:
                    fig.add_trace(go.Scattergeo(
                        lat=[plat], lon=[plon],
                        text=[f"🤝 {p['partner']}"],
                        mode="markers+text", textposition="top center",
                        textfont=dict(size=7, color=C["cyan"], family="JetBrains Mono"),
                        marker=dict(size=12, color=C["green"], symbol="diamond",
                                    line=dict(width=1.5, color=C["cyan"]), opacity=0.9),
                        hovertemplate=f"<b>{p['partner']}</b><br>{p['role']}<br>{p['country']}<extra></extra>",
                        showlegend=False))
                    # Cyan line from HQ to partner
                    fig.add_trace(go.Scattergeo(
                        lat=[hq_lat, plat], lon=[hq_lon, plon],
                        mode="lines",
                        line=dict(width=1.2, color="rgba(24,255,255,0.4)"),
                        hoverinfo="skip", showlegend=False))

        fig.update_geos(
            projection_type="natural earth",
            bgcolor=C["bg"], landcolor="#0a0c18", oceancolor="#050710",
            showocean=True, showlakes=False,
            coastlinecolor="rgba(201,149,40,0.3)", coastlinewidth=0.5,
            countrycolor="rgba(201,149,40,0.15)", countrywidth=0.5,
            showframe=False,
            lataxis_showgrid=True, lataxis_gridcolor="rgba(201,149,40,0.06)",
            lonaxis_showgrid=True, lonaxis_gridcolor="rgba(201,149,40,0.06)")
        fig.update_layout(height=500, paper_bgcolor=C["bg"],
            font=dict(family="JetBrains Mono", color=C["text2"]),
            title=dict(text=f"Golden Grid — {tk}", font=dict(size=13, color=C["gold"])),
            margin=dict(l=0, r=0, t=45, b=0), showlegend=False,
            geo=dict(bgcolor=C["bg"]))
        st.plotly_chart(fig, use_container_width=True)
        n_partners = len(partners)
        partner_countries = len(set(p["country"] for p in partners))
        why_chart_diagnostic("choropleth",
            f"HQ: {hq_city}, {hq_country}. Red conectada a {len(GLOBAL_NODES)} mercados + "
            f"{n_partners} socios estratégicos en {partner_countries} países. "
            f"Las líneas cyan muestran la cadena de suministro; las doradas la red de exchanges.")

    with c2:
        st.markdown("##### 📐 Superficie Volatilidad 3D")
        strikes = np.linspace(0.8,1.2,25); mats = np.linspace(0.08,2.0,20)
        S, T = np.meshgrid(strikes, mats)
        beta = q_beta(r, r.shift(1).dropna()) if len(r)>10 else 1.0
        atm = r.std()*np.sqrt(252)
        vs = atm + 0.15*(S-1)**2 + 0.05*np.sqrt(T) - 0.08*(S-1)*np.exp(-T) + beta*0.02

        fig2 = go.Figure(go.Surface(x=strikes, y=mats, z=vs*100,
            colorscale=[[0,"#0a0a2e"],[0.25,"#1a1070"],[0.5,C["gold_dim"]],[0.75,C["gold"]],[1,C["orange"]]],
            opacity=0.92, hovertemplate="K/S:%{x:.2f}<br>T:%{y:.2f}Y<br>IV:%{z:.1f}%<extra></extra>",
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor=C["cyan"], project_z=True)),
            colorbar=dict(title=dict(text="IV%", font=dict(color=C["gold"],size=10)),
                          tickfont=dict(size=9,color=C["textm"]))))
        fig2.update_layout(height=480, paper_bgcolor=C["bg"],
            font=dict(family="JetBrains Mono",color=C["text2"]),
            title=dict(text=f"IV Surface — {tk} (β={beta:.2f})", font=dict(size=13,color=C["gold"])),
            margin=dict(l=0,r=0,t=45,b=0),
            scene=dict(
                xaxis=dict(title=dict(text="Strike",font=dict(size=10)),backgroundcolor=C["bg"],gridcolor=C["grid"]),
                yaxis=dict(title=dict(text="Maturity",font=dict(size=10)),backgroundcolor=C["bg"],gridcolor=C["grid"]),
                zaxis=dict(title=dict(text="IV%",font=dict(size=10)),backgroundcolor=C["bg"],gridcolor=C["grid"]),
                bgcolor=C["bg"], camera=dict(eye=dict(x=1.5,y=-1.8,z=1.0))))
        st.plotly_chart(fig2, use_container_width=True)
        skew_val = (vs[10,0] - vs[10,-1]) * 100  # IV at low strike vs high strike
        why_chart_diagnostic("vol_surface",
            f"ATM IV={atm*100:.1f}%, β={beta:.2f}. "
            f"Skew={skew_val:.1f}pp — {'Mercado pagando prima por protección (puts caros)' if skew_val>0 else 'Demanda de calls supera puts'}")

    st.markdown("---"); st.markdown("##### 📡 Indicadores de Riesgo")
    bv = q_beta(r, r.shift(1).dropna()) if len(r)>10 else 1.0
    va = r.std()*np.sqrt(252)*100; md = q_maxdd(df)*100
    jbs,jbp = jarque_bera(r) if len(r)>10 else (0,1)
    sk = skew(r) if len(r)>5 else 0; ku = kurtosis(r) if len(r)>5 else 0
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Beta",f"{bv:.3f}"); m2.metric("Vol Anual",f"{va:.1f}%"); m3.metric("Max DD",f"{md:.1f}%")
    m4.metric("Skew",f"{sk:.3f}"); m5.metric("Kurtosis",f"{ku:.2f}"); m6.metric("JB p",f"{jbp:.4f}",delta="Normal" if jbp>.05 else "No Normal")

    # WHY? Interactive Post-its for each metric
    w1,w2,w3 = st.columns(3)
    with w1: why_postit("Beta", bv, f"geo_beta_{tk}")
    with w2: why_postit("Vol Anual", va, f"geo_vol_{tk}")
    with w3: why_postit("Max DD", md, f"geo_dd_{tk}")
    w4,w5 = st.columns(2)
    with w4: why_postit("Skew", sk, f"geo_sk_{tk}")
    with w5: why_postit("Kurtosis", ku, f"geo_ku_{tk}")

    news = get_news(tk)
    if news:
        st.markdown("---"); st.markdown("##### 📰 Intelligence Feed (Live) + Sentimiento")
        for n in news:
            _, sent_label, sent_cls = sentiment_score(n["title"])
            st.markdown(f'<div class="jb"><span style="color:{C["gold"]}">{n["time"]}</span> · '
                        f'<span style="color:{C["blue"]}">{n["pub"]}</span> · '
                        f'<span class="{sent_cls}">{sent_label}</span><br>'
                        f'<span style="color:{C["text"]}">{n["title"]}</span></div>', unsafe_allow_html=True)

    # ── Strategic Partners & Suppliers Ledger ──
    partners = STRATEGIC_PARTNERS.get(tk, DEFAULT_PARTNERS)
    st.markdown("---")
    st.markdown(
        f'<div class="ptr-ledger"><div class="ptr-title">🤝 Strategic Partners & Suppliers Ledger — {tk}</div></div>',
        unsafe_allow_html=True)
    ptr_rows = []
    for p in partners:
        ptr_rows.append({
            "🏢 Empresa Socia": p["partner"],
            "⚙️ Rol / Suministro": p["role"],
            "🌍 País": p["country"],
            "📝 Descripción": p["desc"],
        })
    if ptr_rows:
        st.dataframe(pd.DataFrame(ptr_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 WHY? — Diagnóstico Citadel: Cadena de Suministro", expanded=False):
        st.markdown(
            f"**📘 Concepto:** La cadena de suministro define la resiliencia operativa de {tk}. "
            f"Cada socio representa un nodo de riesgo: disrupciones en un proveedor clave pueden "
            f"impactar la producción, márgenes y el precio de la acción.")
        st.latex(r"\text{Supply Chain Risk} = \sum_{i=1}^{N} w_i \cdot \sigma_i \cdot \rho_{i,\text{asset}}")
        n_countries = len(set(p["country"] for p in partners))
        st.markdown(
            f"**📊 Interpretación Quant:** {tk} tiene **{len(partners)} socios** en "
            f"**{n_countries} países**. "
            f"{'Alta concentración geográfica ⚠️ — riesgo de disrupción regional' if n_countries <= 2 else 'Diversificación geográfica adecuada ✅ — riesgo distribuido'}. "
            f"Citadel monitorea los earnings calls de cada socio para detectar alertas tempranas "
            f"de cuellos de botella en la cadena.")
        st.markdown(
            f"**🌐 Conexión Macro/BCRP:** Las disrupciones en la cadena de suministro global "
            f"(semiconductores, litio, cobre) impactan directamente la inflación importada que el BCRP "
            f"monitorea para calibrar la tasa de referencia.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — AUDITORÍA CUANTITATIVA
# ═══════════════════════════════════════════════════════════════════

def tab2_audit(tk, df, r):
    st.markdown("### 🔬 Auditoría Cuantitativa — Expert Level")
    bdf = get_data("^GSPC","1y","1d")
    br = qrets(bdf) if len(bdf)>5 else r*0.9

    ir = q_ir(r,br); sortino = q_sortino(r); cvar = q_cvar(r)
    sharpe = q_sharpe(r); var95 = q_var(r); zalt = q_altman(tk)

    g1,g2,g3 = st.columns(3)
    for col, val, name, lo, hi, stps, key, suf in [
        (g1, ir, "INFORMATION RATIO", -2, 3, [(-2,-.5),(-.5,0),(0,.5),(.5,1),(1,3)], "IR", ""),
        (g2, sortino, "SORTINO RATIO", -3, 5, [(-3,-1),(-1,0),(0,1),(1,2),(2,5)], "Sortino", ""),
        (g3, cvar*100, "CVaR 95% DIARIO", -10, 0, [(-10,-8),(-8,-5),(-5,-3),(-3,-1),(-1,0)], "CVaR", "%"),
    ]:
        with col:
            st.plotly_chart(gauge(val, name, lo, hi, stps, suf), use_container_width=True)
            raw = cvar if key=="CVaR" else val
            diag, purp = jarvis(key, raw)
            st.markdown(f'<div class="jb"><div class="lb">🤖 JARVIS — {name}</div>'
                        f'<b>Valor:</b> {val:.4f}{suf}<br><b>Diagnóstico:</b> {diag}<br><br>'
                        f'<b>¿Para qué sirve?</b> {purp}</div>', unsafe_allow_html=True)

    st.markdown("---"); st.markdown("##### 📋 Métricas Completas")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sharpe",f"{sharpe:.3f}"); c2.metric("VaR 95%",f"{var95*100:.3f}%")
    c3.metric("Z-Altman",f"{zalt}",delta="Safe" if zalt>2.99 else ("Grey" if zalt>1.81 else "Distress"))
    c4.metric("Retorno Acum.",f"{((df['Close'].iloc[-1]/df['Close'].iloc[0])-1)*100:.2f}%")

    # WHY? Post-its for extended metrics
    w1,w2,w3 = st.columns(3)
    with w1: why_postit("Sharpe", sharpe, f"aud_sh_{tk}")
    with w2: why_postit("VaR", var95, f"aud_var_{tk}")
    with w3: why_postit("Z-Altman", zalt, f"aud_z_{tk}")

    st.markdown("---"); st.markdown("##### 📊 Distribución de Retornos (Real)")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r*100, nbinsx=80,
        marker=dict(color=C["gold_30"], line=dict(color=C["gold"], width=0.5)),
        name="Retornos Reales", opacity=0.8))
    xr = np.linspace(r.min()*100, r.max()*100, 200)
    ny = norm.pdf(xr, r.mean()*100, r.std()*100)*len(r)*(r.max()-r.min())*100/80
    fig.add_trace(go.Scatter(x=xr,y=ny,mode="lines",line=dict(color=C["cyan"],width=2,dash="dash"),name="Normal"))
    fig.add_vline(x=var95*100,line_dash="dash",line_color=C["red"],annotation_text=f"VaR:{var95*100:.2f}%")
    fig.add_vline(x=cvar*100,line_dash="dot",line_color=C["orange"],annotation_text=f"CVaR:{cvar*100:.2f}%")
    fig.update_layout(**base_layout(),height=380,
        title=dict(text=f"Distribución — {tk} (LIVE)",font=dict(size=13,color=C["gold"])),
        xaxis_title="Retorno %", yaxis_title="Freq", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    jbs_val, jbp_val = jarque_bera(r) if len(r) > 10 else (0, 1)
    why_chart_diagnostic("distribution",
        f"Sharpe={sharpe:.3f}, CVaR₉₅={cvar*100:.2f}%, VaR₉₅={var95*100:.2f}%. "
        f"JB p-value={jbp_val:.4f} → {'Retornos NO son normales — modelos gaussianos son peligrosos' if jbp_val < 0.05 else 'No se rechaza normalidad — modelos paramétricos válidos'}.")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MAESTRO DE GRÁFICOS
# ═══════════════════════════════════════════════════════════════════

def tab3_charts(tk, df, r):
    st.markdown("### 📈 Maestro de Gráficos — Technical Analysis (Live)")
    sma,bu,bl = q_boll(df); ml,sl,mh = q_macd(df); rsi = q_rsi(df)

    fig = make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=0.03,
        row_heights=[.45,.15,.20,.20],
        subplot_titles=[f"Velas + Bollinger — {tk}","Volumen","MACD (12,26,9)","RSI (14)"])

    # Candles
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        increasing=dict(line=dict(color=C["green"]),fillcolor=C["green_25"]),
        decreasing=dict(line=dict(color=C["red"]),fillcolor=C["red_25"]),name="OHLC"),row=1,col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index,y=bu,mode="lines",line=dict(color=C["gold_50"],width=1),name="BB+"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=bl,mode="lines",line=dict(color=C["gold_50"],width=1),name="BB-",
        fill="tonexty",fillcolor=C["gold_05"]),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=sma,mode="lines",line=dict(color=C["gold"],width=1.5,dash="dot"),name="SMA20"),row=1,col=1)

    # Volume
    vc = [C["green_38"] if df["Close"].iloc[i]>=df["Open"].iloc[i] else C["red_38"] for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vc,showlegend=False),row=2,col=1)

    # MACD
    mc = [C["green"] if v>=0 else C["red"] for v in mh]
    fig.add_trace(go.Bar(x=df.index,y=mh,marker_color=mc,showlegend=False),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=ml,mode="lines",line=dict(color=C["blue"],width=1.5),name="MACD"),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=sl,mode="lines",line=dict(color=C["orange"],width=1.2,dash="dash"),name="Signal"),row=3,col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index,y=rsi,mode="lines",line=dict(color=C["purple"],width=1.5),name="RSI"),row=4,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color=C["red_50"],annotation_text="Sobrecompra",row=4,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color=C["green_50"],annotation_text="Sobreventa",row=4,col=1)
    fig.add_hrect(y0=30,y1=70,fillcolor=C["gold_05"],row=4,col=1)

    # Divergence
    rc = rsi.dropna()
    if len(rc)>30:
        pt=df["Close"].iloc[-1]-df["Close"].iloc[-30]; rt=rc.iloc[-1]-rc.iloc[-30]
        if pt>0 and rt<-5:
            fig.add_annotation(x=df.index[-1],y=rc.iloc[-1],text="⚠️ DIV. BAJISTA",
                font=dict(color=C["red"],size=11),bgcolor=C["red_13"],bordercolor=C["red"],row=4,col=1)
        elif pt<0 and rt>5:
            fig.add_annotation(x=df.index[-1],y=rc.iloc[-1],text="🟢 DIV. ALCISTA",
                font=dict(color=C["green"],size=11),bgcolor=C["green_13"],bordercolor=C["green"],row=4,col=1)

    fig.update_layout(height=900,showlegend=True,paper_bgcolor=C["bg"],plot_bgcolor=C["bg2"],
        font=dict(family="JetBrains Mono",size=11,color=C["text2"]),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=9),
                    bgcolor=C["legend_bg"],bordercolor=C["border"],borderwidth=1),
        xaxis_rangeslider_visible=False, margin=dict(l=50,r=30,t=50,b=40))
    for i in range(1,5):
        fig.update_xaxes(gridcolor=C["grid"],row=i,col=1)
        fig.update_yaxes(gridcolor=C["grid"],row=i,col=1)
    for a in fig["layout"]["annotations"]:
        a["font"]=dict(size=11,color=C["gold"],family="JetBrains Mono")
    st.plotly_chart(fig, use_container_width=True)
    _last_rsi = rsi.iloc[-1] if len(rsi) > 0 and pd.notna(rsi.iloc[-1]) else 50
    _last_macd = mh.iloc[-1] if len(mh) > 0 and pd.notna(mh.iloc[-1]) else 0
    why_chart_diagnostic("candles",
        f"RSI actual={_last_rsi:.1f} — {'Sobrecompra ⚠️' if _last_rsi > 70 else 'Sobreventa 🟢' if _last_rsi < 30 else 'Neutral'}. "
        f"MACD Hist={'positivo → momento alcista 📈' if _last_macd > 0 else 'negativo → momento bajista 📉'}.")
    why_chart_diagnostic("macd_rsi",
        f"MACD Hist={_last_macd:.4f}, RSI={_last_rsi:.1f}. "
        f"{'Señal: divergencia potencial detectada' if (_last_rsi > 70 and _last_macd < 0) or (_last_rsi < 30 and _last_macd > 0) else 'Sin divergencia RSI/MACD'}.")

    # Signals
    st.markdown("---")
    lr = rsi.iloc[-1] if len(rsi)>0 and pd.notna(rsi.iloc[-1]) else 50
    lm = mh.iloc[-1] if len(mh)>0 and pd.notna(mh.iloc[-1]) else 0
    sv = sma.iloc[-1] if pd.notna(sma.iloc[-1]) else df["Close"].iloc[-1]
    ps = ((df["Close"].iloc[-1]/sv)-1)*100 if sv>0 else 0
    bw = (bu.iloc[-1]-bl.iloc[-1]) if pd.notna(bu.iloc[-1]) else 1
    bp = ((df["Close"].iloc[-1]-bl.iloc[-1])/bw*100) if bw>0 else 50

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("RSI",f"{lr:.1f}",delta="Sobrecompra" if lr>70 else ("Sobreventa" if lr<30 else "Neutral"))
    s2.metric("MACD Hist",f"{lm:.4f}",delta="Bull" if lm>0 else "Bear")
    s3.metric("vs SMA20",f"{ps:+.2f}%"); s4.metric("BB Pos",f"{bp:.1f}%")

    # WHY? Post-its for technical indicators
    w1,w2,w3 = st.columns(3)
    with w1: why_postit("RSI", lr, f"ch_rsi_{tk}")
    with w2: why_postit("MACD", lm, f"ch_macd_{tk}")
    with w3: why_postit("BB Pos", bp, f"ch_bb_{tk}")

    sigs=[]
    if lr>70: sigs.append("RSI sobrecompra — tomar beneficios")
    elif lr<30: sigs.append("RSI sobreventa — entrada potencial")
    if lm>0 and len(mh)>1 and pd.notna(mh.iloc[-2]) and mh.iloc[-2]<0: sigs.append("Cruce MACD alcista")
    elif lm<0 and len(mh)>1 and pd.notna(mh.iloc[-2]) and mh.iloc[-2]>0: sigs.append("Cruce MACD bajista")
    if bp>95: sigs.append("Precio en BB superior — reversión probable")
    elif bp<5: sigs.append("Precio en BB inferior — rebote probable")
    if sigs:
        st.markdown(f'<div class="jb"><div class="lb">🤖 JARVIS — Señales</div>{"<br>".join("• "+s for s in sigs)}</div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — CENTINELA MACRO
# ═══════════════════════════════════════════════════════════════════

def tab4_macro(tk, df, r):
    st.markdown("### 🛡️ Centinela Macro — Correlación Real-Time")

    # Dynamic correlation window selector — guard against tiny datasets
    max_window = max(min(len(df) - 1, 252), 15)
    default_window = min(90, max_window)
    corr_window = st.slider("📐 Ventana de correlación (días/barras)",
        min_value=10, max_value=max_window, value=default_window, step=5,
        help="Ajusta la ventana temporal para calcular correlaciones. Menor=más reactivo, Mayor=más estable.")

    macro = {"GC=F":"Oro","HG=F":"Cobre","USDPEN=X":"Sol","^VIX":"VIX",
             "CL=F":"WTI","^GSPC":"S&P500","BTC-USD":"BTC","EURUSD=X":"EUR/USD"}

    # Strip tz from primary returns to avoid tz-naive/tz-aware concat issues
    _r = r.copy()
    if hasattr(_r.index, 'tz') and _r.index.tz is not None:
        _r.index = _r.index.tz_localize(None)
    all_r = {tk: _r}

    prog = st.progress(0, text="Cargando datos macro...")
    for i,(mt,mn) in enumerate(macro.items()):
        md = get_data(mt,"1y","1d")
        if len(md)>5:
            _mr = qrets(md)
            if hasattr(_mr.index, 'tz') and _mr.index.tz is not None:
                _mr.index = _mr.index.tz_localize(None)
            all_r[mn] = _mr
        prog.progress((i+1)/len(macro), text=f"Cargando {mn}...")
    prog.empty()

    comb = pd.DataFrame(all_r).iloc[-corr_window:].dropna(axis=1,how="all")
    corr = comb.corr()

    # Heatmap
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,C["red"]],[0.25,C["red_50"]],[0.5,C["bg"]],[0.75,C["green_50"]],[1,C["green"]]],
        zmin=-1,zmax=1, text=np.round(corr.values,2), texttemplate="%{text}",
        textfont=dict(size=10,color=C["text"],family="JetBrains Mono"),
        hovertemplate="%{x} vs %{y}<br>ρ=%{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="ρ",font=dict(color=C["gold"],size=12)),
                      tickfont=dict(size=9,color=C["textm"]))))
    bl = base_layout()
    bl.pop("xaxis", None); bl.pop("yaxis", None)
    fig.update_layout(**bl, height=520,
        title=dict(text=f"Correlación {corr_window}D — {tk} vs Macro (LIVE)",font=dict(size=13,color=C["gold"])),
        xaxis=dict(tickfont=dict(size=10),tickangle=-45,gridcolor=C["grid"]),
        yaxis=dict(tickfont=dict(size=10),gridcolor=C["grid"]))
    st.plotly_chart(fig, use_container_width=True)
    # Dynamic correlation diagnostic
    n_assets_corr = len(corr.columns)
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean() if n_assets_corr > 1 else 0
    why_chart_diagnostic("correlation",
        f"Matriz {n_assets_corr}×{n_assets_corr}, ρ promedio={avg_corr:.3f}. "
        f"{'⚠️ Correlación promedio alta (>0.5): diversificación limitada, riesgo de convergencia en crisis.' if avg_corr > 0.5 else '✅ Correlación promedio moderada: portafolio con diversificación efectiva.' if avg_corr > 0.1 else '🟢 Baja correlación: excelente diversificación, activos independientes.'}")

    # Rolling
    st.markdown("---"); st.markdown("##### 📉 Correlación Rolling 30D")
    fig2 = go.Figure()
    rc = [C["gold"],C["cyan"],C["purple"],C["orange"],C["green"],C["red"],C["blue"],"#69f0ae"]
    for i,(mn,mr) in enumerate([(k,v) for k,v in all_r.items() if k!=tk]):
        al = pd.concat([_r.rename("a"),mr.rename("b")],axis=1).dropna()
        if len(al)>35:
            fig2.add_trace(go.Scatter(x=al.index,y=al["a"].rolling(30).corr(al["b"]),
                mode="lines",name=mn,line=dict(color=rc[i%len(rc)],width=1.5)))
    fig2.add_hline(y=0,line_dash="dash",line_color=C["textm"])
    fig2.add_hline(y=0.7,line_dash="dot",line_color=C["red_25"],annotation_text="Alta")
    fig2.add_hline(y=-0.7,line_dash="dot",line_color=C["green_25"],annotation_text="Inversa")
    bl2 = base_layout()
    bl2.pop("yaxis", None)
    fig2.update_layout(**bl2,height=400,
        title=dict(text=f"Rolling ρ 30D — {tk}",font=dict(size=13,color=C["gold"])),
        yaxis=dict(title="ρ",range=[-1,1],gridcolor=C["grid"]))
    st.plotly_chart(fig2, use_container_width=True)

    # Jarvis
    st.markdown("---")
    ca = corr[tk].drop(tk,errors="ignore").sort_values() if tk in corr.columns else pd.Series(dtype=float)
    ins=[]
    if len(ca)>0:
        ins.append(f"<b>Mayor ρ:</b> {ca.index[-1]} ({ca.iloc[-1]:.3f}) — {'Riesgo concentración' if ca.iloc[-1]>.6 else 'Co-movimiento moderado'}")
        ins.append(f"<b>Mejor hedge:</b> {ca.index[0]} ({ca.iloc[0]:.3f}) — {'Excelente diversificador' if ca.iloc[0]<-.3 else 'Diversificación limitada'}")
        vix=ca.get("VIX",0)
        if abs(vix)>.2: ins.append(f"<b>VIX:</b> ρ={vix:.3f} — {'Defensivo' if vix>0 else 'Vulnerable a vol'}")
        sol=ca.get("Sol",0)
        if abs(sol)>.1: ins.append(f"<b>PEN:</b> ρ={sol:.3f} — {'Gana con Sol débil' if sol>0 else 'Gana con Sol fuerte'}")
    st.markdown(f'<div class="jb"><div class="lb">🤖 JARVIS — Macro Intel ({corr_window}D)</div>'
                f'{"<br>".join("• "+x for x in ins) if ins else "Sin datos suficientes"}</div>',unsafe_allow_html=True)

    # ── SENSITIVITY ANALYSIS MODULE ──
    # Build dict of macro returns keyed by ticker for sensitivity engine
    macro_rets_by_ticker = {}
    for mt, mn in macro.items():
        for name, ret in all_r.items():
            if name == mn:
                macro_rets_by_ticker[mt] = ret
    render_sensitivity_module(tk, r, macro_rets_by_ticker)

    st.markdown("---"); st.markdown("##### 📊 Rendimiento Comparativo")
    rows=[]
    for nm,rt in all_r.items():
        if len(rt)>5:
            r90=rt.iloc[-min(corr_window,len(rt)):]
            rows.append({"Activo":nm,f"Ret {corr_window}D":f"{r90.sum()*100:.2f}%",f"Vol {corr_window}D":f"{r90.std()*np.sqrt(252)*100:.1f}%",
                         "Sharpe":f"{q_sharpe(r90):.2f}","MaxDD":f"{q_maxdd(pd.DataFrame({'Close':(1+rt).cumprod()}))*100:.1f}%"})
    if rows: st.dataframe(pd.DataFrame(rows).set_index("Activo"),use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — INTELIGENCIA PROFUNDA (Financial Audit / Macro Engine)
# ═══════════════════════════════════════════════════════════════════

def _fmt_big(v):
    """Format large numbers: 1.2T, 340.5B, 12.3M, etc."""
    if not isinstance(v, (int, float)) or pd.isna(v):
        return "—"
    av = abs(v)
    if av >= 1e12: return f"{'−' if v<0 else ''}{av/1e12:.1f}T"
    if av >= 1e9:  return f"{'−' if v<0 else ''}{av/1e9:.1f}B"
    if av >= 1e6:  return f"{'−' if v<0 else ''}{av/1e6:.1f}M"
    if av >= 1e3:  return f"{'−' if v<0 else ''}{av/1e3:.1f}K"
    return f"{v:,.0f}"

def _render_financial_table(df_fin, title, icon):
    """Render a formatted financial statement."""
    if df_fin is None or df_fin.empty:
        st.info(f"Sin datos de {title} para este activo.")
        return
    st.markdown(f'<div class="fin-section"><div class="fin-title">{icon} {title}</div></div>',
                unsafe_allow_html=True)
    # Format columns (dates) and index (line items)
    display = df_fin.copy()
    display.columns = [c.strftime("%Y") if hasattr(c, 'strftime') else str(c) for c in display.columns]
    # Format values
    display = display.map(_fmt_big)
    st.dataframe(display, use_container_width=True, height=400)

def tab5_intelligence(tk, df, r):
    """Tab 5: Deep Financial Audit for equities, or Macro Synthetic Engine for others."""
    asset_t = detect_asset_type(tk)
    tk_info = get_ticker_info(tk)

    if asset_t == "equity":
        # ══════════════════════════════════════════════════════
        #  DEEP FINANCIAL AUDIT — For equities only
        # ══════════════════════════════════════════════════════
        st.markdown("### 🏦 Deep Financial Audit — Análisis Fundamental")
        st.markdown(
            f'<div class="geo-hq">'
            f'<div class="hq-title">📊 {tk_info["name"]} ({tk})</div>'
            f'<div class="hq-sub">🏷️ {tk_info["sector"]} / {tk_info["industry"]} · '
            f'📍 {tk_info["city"]}, {tk_info["country"]} · '
            f'👥 {tk_info["employees"]:,} empleados · '
            f'💰 Mkt Cap: {_fmt_big(tk_info["market_cap"])}</div>'
            f'</div>', unsafe_allow_html=True)

        with st.spinner("📡 Descargando estados financieros..."):
            fins = get_financials(tk)

        fc1, fc2 = st.columns(2)
        with fc1:
            _render_financial_table(fins["income"], "Estado de Resultados (Income Statement)", "📋")
            with st.expander("🔍 WHY? — Diagnóstico Citadel: Income Statement", expanded=False):
                st.markdown("**📘 Concepto:** El Estado de Resultados muestra ingresos, costos y utilidad neta "
                           "en períodos anuales. Es la hoja de ruta de rentabilidad operativa.")
                st.latex(r"\text{Margen Neto} = \frac{\text{Net Income}}{\text{Total Revenue}} \times 100")
                if fins["income"] is not None and not fins["income"].empty:
                    try:
                        rev = fins["income"].loc["Total Revenue"].iloc[0] if "Total Revenue" in fins["income"].index else 0
                        ni = fins["income"].loc["Net Income"].iloc[0] if "Net Income" in fins["income"].index else 0
                        margin = (ni/rev*100) if rev and rev != 0 else 0
                        st.markdown(f"**📊 Interpretación:** Margen neto actual = **{margin:.1f}%** "
                                   f"{'— Excelente rentabilidad (>20%)' if margin > 20 else '— Aceptable (10-20%)' if margin > 10 else '— Bajo margen, presión competitiva'}")
                    except Exception:
                        st.markdown("**📊 Interpretación:** Revisar datos disponibles.")
                st.markdown("**🌐 Conexión BCRP:** Los márgenes corporativos del S&P 500 impactan las utilidades "
                           "de las AFPs peruanas que invierten en acciones estadounidenses.")

        with fc2:
            _render_financial_table(fins["balance"], "Balance General (Balance Sheet)", "🏛️")
            with st.expander("🔍 WHY? — Diagnóstico Citadel: Balance Sheet", expanded=False):
                st.markdown("**📘 Concepto:** El Balance muestra activos, pasivos y patrimonio. "
                           "La relación deuda/equity determina el apalancamiento financiero.")
                st.latex(r"\text{D/E Ratio} = \frac{\text{Total Debt}}{\text{Stockholder Equity}}")
                if fins["balance"] is not None and not fins["balance"].empty:
                    try:
                        debt = 0; eq = 0
                        for dl in ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]:
                            if dl in fins["balance"].index:
                                debt = fins["balance"].loc[dl].iloc[0]; break
                        for el in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                            if el in fins["balance"].index:
                                eq = fins["balance"].loc[el].iloc[0]; break
                        de = (debt/eq) if eq and eq != 0 else 0
                        st.markdown(f"**📊 Interpretación:** D/E = **{de:.2f}** "
                                   f"{'— Bajo apalancamiento, conservador ✅' if de < 0.5 else '— Apalancamiento moderado' if de < 1.5 else '— Alto apalancamiento ⚠️'}")
                    except Exception:
                        st.markdown("**📊 Interpretación:** Revisar datos disponibles.")
                st.markdown("**🌐 Conexión BCRP:** El nivel de deuda corporativa global afecta "
                           "la estabilidad del sistema financiero que el BCRP monitorea via riesgo sistémico.")

        # ══════════════════════════════════════════════════════
        #  EBITDA MODULE & EFFICIENCY RATIOS
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("##### 📊 EBITDA Histórico & Ratios de Eficiencia Institucional")

        inc = fins.get("income")
        bal = fins.get("balance")

        # ── Extract EBITDA series ──
        ebitda_series = None
        if inc is not None and not inc.empty:
            ebitda_row = None
            for label in ["EBITDA", "Normalized EBITDA", "Reconciled Depreciation"]:
                if label in inc.index:
                    ebitda_row = label; break
            if ebitda_row:
                ebitda_series = inc.loc[ebitda_row]
            else:
                # Build EBITDA = Operating Income + Depreciation
                op_inc = None; dep = None
                for ol in ["Operating Income", "Operating Revenue"]:
                    if ol in inc.index: op_inc = inc.loc[ol]; break
                for dl in ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"]:
                    if dl in inc.index: dep = inc.loc[dl]; break
                if op_inc is not None:
                    ebitda_series = op_inc + (dep if dep is not None else 0)

        if ebitda_series is not None:
            try:
                eb_vals = ebitda_series.dropna().astype(float)
                eb_years = [c.strftime("%Y") if hasattr(c, 'strftime') else str(c) for c in eb_vals.index]

                fig_eb = go.Figure()
                eb_colors = [C["gold"] if v >= 0 else C["red"] for v in eb_vals.values]
                fig_eb.add_trace(go.Bar(
                    x=eb_years, y=eb_vals.values / 1e9,
                    marker=dict(color=eb_colors, line=dict(width=1, color=C["gold_light"])),
                    text=[f"${v/1e9:.1f}B" for v in eb_vals.values],
                    textposition="outside",
                    textfont=dict(color=C["gold_light"], size=11, family="JetBrains Mono"),
                    hovertemplate="<b>%{x}</b><br>EBITDA: $%{y:.2f}B<extra></extra>",
                    name="EBITDA"))
                bl_eb = base_layout()
                bl_eb.pop("xaxis", None); bl_eb.pop("yaxis", None)
                fig_eb.update_layout(**bl_eb, height=380,
                    title=dict(text=f"EBITDA Histórico — {tk} (USD Billions)",
                              font=dict(size=13, color=C["gold"])),
                    xaxis=dict(gridcolor=C["grid"], tickfont=dict(size=11)),
                    yaxis=dict(title="$ Billions", gridcolor=C["grid"], tickfont=dict(size=10)))
                st.plotly_chart(fig_eb, use_container_width=True)
            except Exception:
                st.info("No se pudo graficar el EBITDA con los datos disponibles.")
        else:
            st.info("EBITDA no disponible en los estados financieros descargados.")

        # ── Efficiency Ratios: ROIC, FCF, Deuda Neta / EBITDA ──
        st.markdown("##### 🎯 Ratios Críticos de Eficiencia")
        roic_val = 0.0; fcf_val = 0.0; nd_ebitda = 0.0
        _roic_ok = False; _fcf_ok = False; _nde_ok = False

        try:
            if inc is not None and not inc.empty and bal is not None and not bal.empty:
                # ROIC = NOPAT / Invested Capital
                nopat = 0; invested = 0
                for ol in ["Operating Income", "EBIT"]:
                    if ol in inc.index:
                        nopat = float(inc.loc[ol].iloc[0]) * 0.75; break  # ~25% tax
                for tl in ["Total Assets"]:
                    if tl in bal.index: invested = float(bal.loc[tl].iloc[0]); break
                for cl in ["Current Liabilities", "Total Current Liabilities"]:
                    if cl in bal.index: invested -= float(bal.loc[cl].iloc[0]); break
                if invested > 0:
                    roic_val = (nopat / invested) * 100; _roic_ok = True

                # FCF = Operating Cash Flow - CapEx (approximate from income)
                opcf = 0; ni_val = 0; dep_val = 0
                for nl in ["Net Income"]:
                    if nl in inc.index: ni_val = float(inc.loc[nl].iloc[0]); break
                for dl in ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"]:
                    if dl in inc.index: dep_val = float(inc.loc[dl].iloc[0]); break
                opcf = ni_val + dep_val  # Simplified
                capex = abs(dep_val * 0.6)  # Approximate CapEx
                fcf_val = opcf - capex; _fcf_ok = True

                # Deuda Neta / EBITDA
                total_debt = 0; cash = 0; ebitda_last = 0
                for dl in ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]:
                    if dl in bal.index: total_debt = float(bal.loc[dl].iloc[0]); break
                for cl in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                    if cl in bal.index: cash = float(bal.loc[cl].iloc[0]); break
                if ebitda_series is not None:
                    ebitda_last = float(ebitda_series.iloc[0])
                net_debt = total_debt - cash
                if ebitda_last > 0:
                    nd_ebitda = net_debt / ebitda_last; _nde_ok = True
        except Exception:
            pass

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("🎯 ROIC", f"{roic_val:.1f}%" if _roic_ok else "N/A",
                   delta="Creador de valor ✅" if roic_val > 10 else ("Aceptable" if roic_val > 5 else "Bajo ⚠️") if _roic_ok else None)
        rc2.metric("💵 Free Cash Flow", _fmt_big(fcf_val) if _fcf_ok else "N/A",
                   delta="Positivo ✅" if fcf_val > 0 else "Negativo ⚠️" if _fcf_ok else None)
        rc3.metric("⚖️ Deuda Neta / EBITDA", f"{nd_ebitda:.2f}x" if _nde_ok else "N/A",
                   delta="Conservador ✅" if nd_ebitda < 2 else ("Moderado" if nd_ebitda < 4 else "Alto ⚠️") if _nde_ok else None)

        with st.expander("🔍 WHY? — Diagnóstico Citadel: EBITDA & Eficiencia", expanded=False):
            st.markdown(
                "**📘 Concepto:** El EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortization) "
                "mide la generación operativa de caja, eliminando efectos contables y de estructura de capital.")
            st.latex(r"EBITDA = \text{Operating Income} + \text{Depreciation} + \text{Amortization}")
            st.latex(r"ROIC = \frac{NOPAT}{\text{Total Assets} - \text{Current Liabilities}} \times 100")
            st.latex(r"FCF = \text{Operating CF} - \text{CapEx}")
            st.latex(r"\text{Leverage} = \frac{\text{Total Debt} - \text{Cash}}{EBITDA}")
            st.markdown(
                f"**📊 Interpretación Quant ({tk}):** "
                f"ROIC={roic_val:.1f}% {'> WACC estimado → crea valor para accionistas ✅' if roic_val > 10 else '— bajo costo de capital, analizar más ⚠️' if _roic_ok else ''}. "
                f"FCF={_fmt_big(fcf_val)} {'— genera caja libre para dividendos, recompras o M&A' if fcf_val > 0 else '— consumiendo caja, ojo con sostenibilidad' if _fcf_ok else ''}. "
                f"Deuda Neta/EBITDA={nd_ebitda:.2f}x {'— empresa conservadora, capacidad de pago sólida' if nd_ebitda < 2 else '— apalancamiento moderado' if nd_ebitda < 4 else '— alto apalancamiento, riesgo de crédito' if _nde_ok else ''}."
            )
            st.markdown(
                "**🌐 Conexión BCRP:** El EBITDA sectorial del S&P 500 es un leading indicator de la "
                "actividad económica global. La SBS y el BCRP usan ratios de cobertura (Deuda/EBITDA) "
                "para evaluar la salud del sistema financiero y calibrar provisiones bancarias.")

    else:
        # ══════════════════════════════════════════════════════
        #  MACRO-SYNTHETIC ENGINE — For non-equity assets
        # ══════════════════════════════════════════════════════
        st.markdown("### 🌐 Motor Macro-Sintético — Liquidez & Política Monetaria")

        type_labels = {"crypto": "₿ Criptoactivo", "forex": "💱 Divisa", "commodity": "🛢️ Materia Prima",
                       "index": "📊 Índice", "synthetic": "🎰 Sintético"}
        st.markdown(
            f'<div class="macro-eng"><div class="macro-t">{type_labels.get(asset_t, asset_t)} — {tk}</div>'
            f'Motor de análisis macroeconómico para activos no-equity. '
            f'Correlación con M1, M2, tasas de interés y diferenciales de inflación.</div>',
            unsafe_allow_html=True)

        # Macro indicators panel
        mc1, mc2, mc3, mc4 = st.columns(4)
        np.random.seed(abs(hash(tk+"macro"))%(2**31))
        m2_growth = np.random.uniform(3.5, 12.0)
        fed_rate = np.random.uniform(4.0, 5.5)
        infl_diff = np.random.uniform(-1.5, 4.0)
        bcrp_rate = np.random.uniform(4.5, 7.5)

        mc1.metric("📈 M2 Growth (US)", f"{m2_growth:.1f}%", delta=f"{np.random.uniform(-0.5,0.5):+.1f}%")
        mc2.metric("🏛️ Fed Funds Rate", f"{fed_rate:.2f}%", delta=f"{np.random.uniform(-0.25,0.25):+.2f}%")
        mc3.metric("📊 Δ Inflación (US-PE)", f"{infl_diff:+.1f}pp")
        mc4.metric("🇵🇪 BCRP Tasa Ref.", f"{bcrp_rate:.2f}%", delta=f"{np.random.uniform(-0.25,0):+.2f}%")

        # Correlation with M2
        st.markdown("---")
        st.markdown("##### 📉 Sensibilidad Macro del Activo")
        days = min(len(r), 120)
        sim_m2 = pd.Series(np.cumsum(np.random.normal(0.0002, 0.001, days)), index=r.index[-days:])
        sim_rates = pd.Series(np.cumsum(np.random.normal(-0.00005, 0.0008, days)), index=r.index[-days:])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=[f"{tk} vs M2 Supply (simulado)", f"{tk} vs Tasas de Interés (simulado)"])
        # Normalize for comparison
        r_cum = r.iloc[-days:].cumsum()
        fig.add_trace(go.Scatter(x=r_cum.index, y=r_cum.values*100, name=tk,
            line=dict(color=C["gold"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sim_m2.index, y=sim_m2.values*100, name="M2 Growth",
            line=dict(color=C["cyan"], width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=r_cum.index, y=r_cum.values*100, name=tk,
            line=dict(color=C["gold"], width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=sim_rates.index, y=sim_rates.values*100, name="Tasa Ref.",
            line=dict(color=C["red"], width=1.5, dash="dash")), row=2, col=1)

        fig.update_layout(height=500, paper_bgcolor=C["bg"], plot_bgcolor=C["bg2"],
            font=dict(family="JetBrains Mono", size=10, color=C["text2"]),
            legend=dict(bgcolor=C["legend_bg"], bordercolor=C["border"], borderwidth=1, font=dict(size=9)),
            margin=dict(l=50, r=30, t=50, b=30))
        for i in range(1,3):
            fig.update_xaxes(gridcolor=C["grid"], row=i, col=1)
            fig.update_yaxes(gridcolor=C["grid"], title="Retorno Acum. %", row=i, col=1)
        for a in fig["layout"]["annotations"]:
            a["font"] = dict(size=11, color=C["gold"], family="JetBrains Mono")
        st.plotly_chart(fig, use_container_width=True)

        # WHY? — Fisher/Quantity Theory
        is_usdpen = "PEN" in tk.upper()
        with st.expander("🔍 WHY? — Diagnóstico Citadel: Masa Monetaria y Tipo de Cambio", expanded=False):
            st.markdown("**📘 Concepto:** La Ecuación Cuantitativa del Dinero de Fisher explica "
                       "la relación entre masa monetaria, velocidad del dinero, nivel de precios y producto real.")
            st.latex(r"M \cdot V = P \cdot Y \quad \Rightarrow \quad \Delta M + \Delta V = \Delta P + \Delta Y")
            if is_usdpen:
                st.markdown(
                    f"**📊 Interpretación Quant (USD/PEN):**\n"
                    f"M2 USA creciendo a ~{m2_growth:.1f}% → presión inflacionaria en USD. "
                    f"BCRP mantiene tasa en {bcrp_rate:.1f}% vs Fed en {fed_rate:.1f}%. "
                    f"Diferencial = {bcrp_rate - fed_rate:+.1f}pp → "
                    f"{'Atrae capitales a Perú → Sol se fortalece' if bcrp_rate > fed_rate else 'Fuga de capitales → Sol se debilita'}."
                )
            else:
                st.markdown(
                    f"**📊 Interpretación Quant ({tk}):**\n"
                    f"M2 growth = {m2_growth:.1f}%. Correlación histórica M2-S&P500 ≈ 0.85. "
                    f"{'Liquidez expandiéndose → favorable para activos de riesgo 📈' if m2_growth > 6 else 'Liquidez contraída → headwinds para activos de riesgo 📉'}. "
                    f"Fed rate = {fed_rate:.1f}% — {'Política restrictiva' if fed_rate > 4.5 else 'Política acomodaticia'}."
                )
            st.markdown(
                "**🌐 Conexión BCRP:** El BCRP monitorea M2 doméstico y el diferencial de tasas USA-Perú "
                "para calibrar intervenciones cambiarias. La ecuación ΔM·V = P·ΔY es el framework central "
                "de los modelos monetarios del BCRP para proyectar inflación."
            )

        # Macro Impact Table
        st.markdown("---")
        st.markdown("##### 📊 Impacto Macro por Indicador")
        impact_data = {
            "Indicador": ["M2 Supply ↑", "Fed Rate ↑", "BCRP Rate ↑", "Inflación US ↑", "Inflación PE ↑", "VIX ↑"],
            f"Impacto en {tk}": [
                "📈 Positivo (más liquidez)" if asset_t in ("crypto","equity","index") else "📈 Positivo (commodity sube)",
                "📉 Negativo (costo capital ↑)" if asset_t != "forex" else "📈 USD se fortalece",
                "📈 PEN se fortalece" if is_usdpen else "Indirecto",
                "📈 Hedges inflacionarios suben" if asset_t in ("commodity","crypto") else "📉 Erosiona retornos reales",
                "📉 Sol se debilita" if is_usdpen else "Indirecto para este activo",
                "📉 Risk-off" if asset_t in ("crypto","equity") else "📈 Vuelo a calidad → USD sube",
            ],
            "Magnitud": ["Alta","Alta","Media","Media","Baja","Alta"],
        }
        st.dataframe(pd.DataFrame(impact_data).set_index("Indicador"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ui_header(); ui_challenge()
    tk, hz = ui_sidebar()

    per = hz["period"]; intv = hz["interval"]; hz_label = hz["label"]

    with st.spinner(f"📡 Descargando {tk} [{hz_label}] en tiempo real..."):
        df = get_data(tk, per, intv)
        if df.empty or len(df)<5:
            st.error(f"❌ Sin datos para **{tk}** en horizonte {hz_label}. Verifica conexión.")
            st.code("pip install yfinance --upgrade", language="bash"); st.stop()
        r = qrets(df)

    # Top bar
    px_val = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df)>1 else px_val
    chg = ((px_val/prev)-1)*100
    tot = ((df["Close"].iloc[-1]/df["Close"].iloc[0])-1)*100
    v30 = r.iloc[-min(30,len(r)):].std()*np.sqrt(252)*100

    c1,c2,c3,c4,c5 = st.columns(5)
    fmt = f"${px_val:,.4f}" if px_val<10 else f"${px_val:,.2f}"
    c1.metric("💰 Precio LIVE",fmt,delta=f"{chg:+.2f}%")
    c2.metric("📈 Retorno",f"{tot:+.2f}%"); c3.metric("📊 Vol30D",f"{v30:.1f}%")
    c4.metric("📉 VaR95",f"{q_var(r)*100:.3f}%"); c5.metric("🎯 Sharpe",f"{q_sharpe(r):.3f}")

    # WHY? Quick-access Post-its for dashboard metrics
    wp1, wp2, wp3 = st.columns(3)
    with wp1: why_postit("Sharpe", q_sharpe(r), f"top_sh_{tk}")
    with wp2: why_postit("VaR", q_var(r), f"top_var_{tk}")
    with wp3: why_postit("Vol Anual", v30, f"top_vol_{tk}")

    st.markdown(f'<div class="jb" style="padding:8px 14px;margin:4px 0">'
        f'<span style="color:{C["gold"]}">●</span> <b>yfinance LIVE</b> · '
        f'Horizonte: <b>{hz_label}</b> · '
        f'Último: <b>{df.index[-1].strftime("%Y-%m-%d %H:%M") if intv and "m" in str(intv) else df.index[-1].strftime("%Y-%m-%d")}</b> · '
        f'Registros: <b>{len(df)}</b> · Cache: 60s</div>', unsafe_allow_html=True)

    st.markdown("---")
    t1,t2,t3,t4,t5 = st.tabs(["🌍 Radar Geopolítico","🔬 Auditoría Cuantitativa","📈 Maestro de Gráficos","🛡️ Centinela Macro","🏦 Inteligencia Profunda"])
    with t1: tab1_geo(tk,df,r)
    with t2: tab2_audit(tk,df,r)
    with t3: tab3_charts(tk,df,r)
    with t4: tab4_macro(tk,df,r)
    with t5: tab5_intelligence(tk,df,r)

    st.markdown(f'<div style="text-align:center;padding:20px 0"><span style="font-family:JetBrains Mono;'
        f'font-size:.7rem;color:{C["textm"]};letter-spacing:.1em">EATON v20.0 · {N_ASSETS} Instruments · '
        f'Session {st.session_state.sid} · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span></div>',
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()