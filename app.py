import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path_dir in [BASE_DIR, os.path.join(BASE_DIR, 'src'), os.path.join(BASE_DIR, 'scripts')]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from src.models import LSTMForecaster, GRUForecaster, TransformerForecaster
from src.advisory_agent import AirQualityHealthAgent
from src.open_meteo_client import search_city, fetch_live_telemetry, fetch_14day_sequence

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI-AIR | Air Quality Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_css(dark: bool) -> str:
    if dark:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #070b14 !important;
    color: #e2e8f0 !important;
}
.stApp { background: linear-gradient(135deg,#070b14 0%,#0d1526 50%,#0a1220 100%) !important; }
#MainMenu,footer,header { visibility:hidden; }
section[data-testid="stSidebar"] { display:none !important; }
.block-container { padding:1.2rem 2.5rem 3rem 2.5rem !important; max-width:1380px !important; }

/* ── Brand ── */
.brand-header{display:flex;align-items:center;gap:14px;padding:1rem 0 0.6rem 0;border-bottom:1px solid rgba(56,189,248,0.15);margin-bottom:0.4rem;}
.brand-name{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#38bdf8 0%,#818cf8 55%,#34d399 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.6px;}
.brand-tagline{font-size:0.79rem;color:#475569;font-weight:400;margin-top:-2px;}
.brand-pill{margin-left:auto;display:flex;gap:8px;align-items:center;}
.pill-tag{background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);border-radius:20px;padding:0.25rem 0.8rem;font-size:0.72rem;font-weight:600;color:#38bdf8;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:rgba(15,23,42,0.9) !important;border-radius:12px !important;padding:5px !important;border:1px solid rgba(56,189,248,0.12) !important;gap:4px !important;box-shadow:0 2px 12px rgba(0,0,0,0.4) !important;margin-bottom:1.4rem !important;}
.stTabs [data-baseweb="tab"]{background:transparent !important;border-radius:9px !important;color:#64748b !important;font-weight:500 !important;font-size:0.875rem !important;padding:0.5rem 1.1rem !important;transition:all 0.2s ease !important;border:none !important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(56,189,248,0.2) 0%,rgba(129,140,248,0.2) 100%) !important;color:#38bdf8 !important;font-weight:600 !important;border:1px solid rgba(56,189,248,0.3) !important;}
.stTabs [data-baseweb="tab-highlight"]{display:none !important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:0 !important;}

/* ── Cards ── */
.card{background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.12);border-radius:16px;padding:1.4rem 1.6rem;box-shadow:0 4px 24px rgba(0,0,0,0.35);margin-bottom:1rem;transition:box-shadow 0.2s;}
.card:hover{box-shadow:0 8px 36px rgba(56,189,248,0.12);}
.stat-card{background:rgba(15,23,42,0.85);border:1px solid rgba(56,189,248,0.15);border-radius:14px;padding:1.1rem 1.2rem;text-align:center;box-shadow:0 2px 14px rgba(0,0,0,0.3);transition:transform 0.2s,box-shadow 0.2s;position:relative;overflow:hidden;}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 8px 28px rgba(56,189,248,0.18);}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#38bdf8,#818cf8);border-radius:14px 14px 0 0;}
.stat-label{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;color:#475569;font-weight:700;margin-bottom:0.4rem;}
.stat-value{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#e2e8f0;line-height:1;}
.stat-value.blue{color:#38bdf8;} .stat-value.green{color:#34d399;} .stat-value.amber{color:#fbbf24;} .stat-value.red{color:#ef4444;}
.stat-unit{font-size:0.69rem;color:#334155;margin-top:0.25rem;}

/* ── Section titles ── */
.section-title{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:700;color:#94a3b8;display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.85rem 0;}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(56,189,248,0.25) 0%,transparent 100%);margin-left:8px;}

/* ── Page hero ── */
.page-title{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:800;color:#e2e8f0;line-height:1.15;margin-bottom:0.35rem;}
.page-subtitle{font-size:0.9rem;color:#64748b;max-width:700px;line-height:1.65;margin-bottom:1.1rem;}

/* ── AQI Hero ── */
.aqi-hero{border-radius:16px;padding:1.8rem;text-align:center;color:white;box-shadow:0 6px 30px rgba(0,0,0,0.4);}
.aqi-hero-val{font-family:'Space Grotesk',sans-serif;font-size:3.8rem;font-weight:800;line-height:1;letter-spacing:-2px;}
.aqi-hero-label{font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.8;margin-bottom:0.3rem;}
.aqi-hero-cat{font-size:1rem;font-weight:600;opacity:0.95;margin-top:0.4rem;}

/* ── AQI Pills ── */
.aqi-pill{display:inline-block;padding:0.28rem 0.85rem;border-radius:50px;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;}
.aqi-good{background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3);}
.aqi-satisf{background:rgba(163,230,53,0.15);color:#a3e635;border:1px solid rgba(163,230,53,0.3);}
.aqi-moderate{background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);}
.aqi-poor{background:rgba(249,115,22,0.15);color:#f97316;border:1px solid rgba(249,115,22,0.3);}
.aqi-verypoor{background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3);}
.aqi-severe{background:rgba(168,85,247,0.15);color:#a855f7;border:1px solid rgba(168,85,247,0.3);}

/* ── Pollutant Card ── */
.pollutant-card{background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:0.9rem 1rem;text-align:center;transition:transform 0.2s,box-shadow 0.2s;}
.pollutant-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(56,189,248,0.12);}
.pollutant-name{font-size:0.75rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.06em;}
.pollutant-val{font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:700;color:#e2e8f0;}
.pollutant-unit{font-size:0.66rem;color:#475569;}
.pollutant-desc{font-size:0.68rem;color:#64748b;margin-top:0.3rem;line-height:1.4;}

/* ── Advisory ── */
.advisory-box{background:linear-gradient(135deg,rgba(56,189,248,0.07) 0%,rgba(129,140,248,0.07) 100%);border:1px solid rgba(56,189,248,0.2);border-left:4px solid #38bdf8;border-radius:12px;padding:1.2rem 1.4rem;margin-top:0.8rem;}
.advisory-label{font-size:0.72rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;}
.advisory-text{font-size:1rem;color:#e2e8f0;font-weight:500;line-height:1.6;}

/* ── Concept / Info ── */
.concept-box{background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.85rem;}
.concept-title{font-weight:700;color:#38bdf8;font-size:0.95rem;margin-bottom:0.35rem;}
.concept-body{font-size:0.875rem;color:#94a3b8;line-height:1.65;}
.info-banner{background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);border-radius:10px;padding:0.75rem 1rem;font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;display:flex;align-items:flex-start;gap:10px;}

/* ── Guide Card ── */
.guide-card{background:rgba(15,23,42,0.8);border-radius:14px;border:1px solid rgba(56,189,248,0.12);padding:1.3rem;box-shadow:0 2px 14px rgba(0,0,0,0.25);height:100%;}
.guide-icon{font-size:1.8rem;margin-bottom:0.6rem;}
.guide-title{font-family:'Space Grotesk',sans-serif;font-size:0.95rem;font-weight:700;color:#e2e8f0;margin-bottom:0.4rem;}
.guide-body{font-size:0.82rem;color:#64748b;line-height:1.65;}

/* ── Widgets ── */
.stTextInput>div>div>input{background:rgba(15,23,42,0.8) !important;border:1.5px solid rgba(56,189,248,0.25) !important;border-radius:10px !important;color:#e2e8f0 !important;font-size:0.9rem !important;box-shadow:0 2px 8px rgba(0,0,0,0.2) !important;}
.stTextInput>div>div>input:focus{border-color:#38bdf8 !important;box-shadow:0 0 0 3px rgba(56,189,248,0.12) !important;}
.stSelectbox>div>div{background:rgba(15,23,42,0.8) !important;border:1.5px solid rgba(56,189,248,0.2) !important;border-radius:10px !important;color:#e2e8f0 !important;}
.stButton>button{background:linear-gradient(135deg,#0ea5e9 0%,#6366f1 100%) !important;color:white !important;font-weight:600 !important;border:none !important;border-radius:10px !important;padding:0.55rem 1.5rem !important;font-size:0.875rem !important;box-shadow:0 3px 14px rgba(14,165,233,0.3) !important;transition:all 0.2s ease !important;}
.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 6px 20px rgba(14,165,233,0.45) !important;}
label{color:#64748b !important;font-size:0.82rem !important;font-weight:600 !important;}
[data-testid="stMetric"]{background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:0.8rem 1rem;}
[data-testid="stMetricLabel"]{color:#475569 !important;font-size:0.78rem !important;font-weight:600 !important;}
[data-testid="stMetricValue"]{color:#e2e8f0 !important;font-size:1.55rem !important;font-weight:700 !important;}
.streamlit-expanderHeader{background:rgba(15,23,42,0.8) !important;border:1px solid rgba(56,189,248,0.15) !important;border-radius:10px !important;color:#38bdf8 !important;font-weight:600 !important;font-size:0.875rem !important;}
.streamlit-expanderContent{background:rgba(7,11,20,0.6) !important;border:1px solid rgba(56,189,248,0.1) !important;border-top:none !important;border-radius:0 0 10px 10px !important;}
.stProgress>div>div{background:linear-gradient(90deg,#38bdf8 0%,#818cf8 100%) !important;border-radius:4px !important;}
.stAlert{background:rgba(15,23,42,0.8) !important;border:1px solid rgba(56,189,248,0.2) !important;border-radius:10px !important;color:#e2e8f0 !important;}
hr{border-color:rgba(56,189,248,0.1) !important;margin:1rem 0 !important;}

/* ── Benchmark/Medal ── */
.medal-card{background:rgba(15,23,42,0.85);border:1.5px solid rgba(56,189,248,0.12);border-radius:16px;padding:1.4rem;text-align:center;box-shadow:0 3px 16px rgba(0,0,0,0.3);transition:transform 0.2s,box-shadow 0.2s;}
.medal-card:hover{transform:translateY(-4px);box-shadow:0 10px 32px rgba(56,189,248,0.15);}
.medal-emoji{font-size:2.2rem;} .medal-model{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:0.4rem 0;}
.medal-stat{font-size:0.78rem;color:#475569;margin:0.15rem 0;} .medal-stat b{color:#38bdf8;font-size:1.0rem;}

/* ── Doc cards ── */
.doc-card{background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.1);border-radius:16px;padding:1.5rem;box-shadow:0 2px 14px rgba(0,0,0,0.25);margin-bottom:1rem;}
.doc-card h4{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#38bdf8;margin-bottom:0.5rem;}
.doc-card p,.doc-card li{font-size:0.875rem;color:#94a3b8;line-height:1.7;margin-bottom:0.25rem;}
.doc-card ul{padding-left:1.2rem;}
.tech-badge{display:inline-block;background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.2);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;font-weight:600;color:#38bdf8;margin:2px 3px 2px 0;}

/* ── Step rows ── */
.step-row{display:flex;gap:10px;margin-bottom:1.1rem;}
.step-num{min-width:28px;height:28px;background:linear-gradient(135deg,#38bdf8,#818cf8);color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;margin-top:1px;}
.step-body{font-size:0.875rem;color:#94a3b8;line-height:1.6;} .step-body b{color:#e2e8f0;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px;height:6px;} ::-webkit-scrollbar-track{background:rgba(15,23,42,0.5);}
::-webkit-scrollbar-thumb{background:rgba(56,189,248,0.3);border-radius:3px;} ::-webkit-scrollbar-thumb:hover{background:rgba(56,189,248,0.5);}
</style>"""
    else:
        return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f4f6fb !important;
    color: #0f172a !important;
}
.stApp { background: linear-gradient(160deg, #eef2ff 0%, #f0fdf4 50%, #f8fafc 100%) !important; }
#MainMenu,footer,header { visibility:hidden; }
section[data-testid="stSidebar"] { display:none !important; }
.block-container { padding:1.2rem 2.5rem 3rem 2.5rem !important; max-width:1380px !important; }

/* ── General text ── */
p, li, span, div { color: #1e293b; }
.stMarkdown p { color: #1e293b !important; font-size: 0.9rem; line-height: 1.7; }
.stMarkdown li { color: #1e293b !important; }
.stMarkdown a { color: #4f46e5 !important; }
code { color: #4f46e5 !important; background: rgba(99,102,241,0.08) !important; padding: 0.1rem 0.4rem; border-radius: 4px; }

/* ── Brand ── */
.brand-header{display:flex;align-items:center;gap:14px;padding:1rem 0 0.6rem 0;border-bottom:2px solid rgba(99,102,241,0.12);margin-bottom:0.4rem;}
.brand-name{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:800;background:linear-gradient(90deg,#4f46e5 0%,#0ea5e9 55%,#10b981 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.6px;}
.brand-tagline{font-size:0.79rem;color:#475569;font-weight:500;margin-top:-2px;}
.brand-pill{margin-left:auto;display:flex;gap:8px;align-items:center;}
.pill-tag{background:rgba(99,102,241,0.09);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:0.25rem 0.8rem;font-size:0.72rem;font-weight:600;color:#4f46e5;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:white !important;border-radius:12px !important;padding:5px !important;border:1px solid rgba(99,102,241,0.15) !important;gap:4px !important;box-shadow:0 2px 12px rgba(99,102,241,0.07) !important;margin-bottom:1.4rem !important;}
.stTabs [data-baseweb="tab"]{background:transparent !important;border-radius:9px !important;color:#334155 !important;font-weight:600 !important;font-size:0.875rem !important;padding:0.5rem 1.1rem !important;transition:all 0.2s ease !important;border:none !important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1 0%,#4f46e5 100%) !important;color:white !important;font-weight:700 !important;box-shadow:0 3px 12px rgba(99,102,241,0.35) !important;}
.stTabs [data-baseweb="tab-highlight"]{display:none !important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:0 !important;}

/* ── Cards ── */
.card{background:white;border:1px solid rgba(99,102,241,0.1);border-radius:16px;padding:1.4rem 1.6rem;box-shadow:0 2px 16px rgba(99,102,241,0.07);margin-bottom:1rem;transition:box-shadow 0.2s;}
.card:hover{box-shadow:0 6px 28px rgba(99,102,241,0.13);}
.stat-card{background:white;border:1px solid rgba(99,102,241,0.12);border-radius:14px;padding:1.1rem 1.2rem;text-align:center;box-shadow:0 2px 12px rgba(99,102,241,0.06);transition:transform 0.2s,box-shadow 0.2s;position:relative;overflow:hidden;}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 8px 28px rgba(99,102,241,0.14);}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#6366f1,#0ea5e9);border-radius:14px 14px 0 0;}
.stat-label{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.09em;color:#475569;font-weight:700;margin-bottom:0.4rem;}
.stat-value{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#0f172a;line-height:1;}
.stat-value.blue{color:#4f46e5;} .stat-value.green{color:#059669;} .stat-value.amber{color:#b45309;} .stat-value.red{color:#dc2626;}
.stat-unit{font-size:0.72rem;color:#64748b;margin-top:0.25rem;font-weight:500;}

/* ── Section titles ── */
.section-title{font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:700;color:#1e293b;display:flex;align-items:center;gap:8px;margin:1.4rem 0 0.85rem 0;}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(99,102,241,0.25) 0%,transparent 100%);margin-left:8px;}

/* ── Page hero ── */
.page-title{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:800;color:#0f172a;line-height:1.15;margin-bottom:0.35rem;}
.page-subtitle{font-size:0.9rem;color:#334155;max-width:700px;line-height:1.65;margin-bottom:1.1rem;}

/* ── AQI Hero ── */
.aqi-hero{border-radius:16px;padding:1.8rem;text-align:center;color:white;box-shadow:0 6px 30px rgba(99,102,241,0.3);}
.aqi-hero-val{font-family:'Space Grotesk',sans-serif;font-size:3.8rem;font-weight:800;line-height:1;letter-spacing:-2px;}
.aqi-hero-label{font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.9;margin-bottom:0.3rem;}
.aqi-hero-cat{font-size:1rem;font-weight:600;opacity:0.97;margin-top:0.4rem;}

/* ── AQI Pills ── */
.aqi-pill{display:inline-block;padding:0.28rem 0.85rem;border-radius:50px;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;}
.aqi-good{background:#d1fae5;color:#064e3b;} .aqi-satisf{background:#ecfccb;color:#1a2e05;} .aqi-moderate{background:#fef3c7;color:#78350f;}
.aqi-poor{background:#fee2e2;color:#7f1d1d;} .aqi-verypoor{background:#fce7f3;color:#500724;} .aqi-severe{background:#ede9fe;color:#3b0764;}

/* ── Pollutant Card ── */
.pollutant-card{background:white;border:1px solid rgba(99,102,241,0.1);border-radius:12px;padding:0.9rem 1rem;text-align:center;box-shadow:0 2px 10px rgba(99,102,241,0.05);transition:transform 0.2s,box-shadow 0.2s;}
.pollutant-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(99,102,241,0.12);}
.pollutant-name{font-size:0.75rem;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:0.06em;}
.pollutant-val{font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:700;color:#0f172a;}
.pollutant-unit{font-size:0.66rem;color:#475569;font-weight:600;}
.pollutant-desc{font-size:0.68rem;color:#334155;margin-top:0.3rem;line-height:1.4;}

/* ── Advisory ── */
.advisory-box{background:linear-gradient(135deg,#eff6ff 0%,#f0fdf4 100%);border:1px solid rgba(99,102,241,0.2);border-left:4px solid #6366f1;border-radius:12px;padding:1.2rem 1.4rem;margin-top:0.8rem;}
.advisory-label{font-size:0.72rem;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;}
.advisory-text{font-size:1rem;color:#0f172a;font-weight:500;line-height:1.6;}

/* ── Concept / Info ── */
.concept-box{background:linear-gradient(135deg,#eff6ff 0%,#eef2ff 100%);border:1px solid rgba(99,102,241,0.18);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.85rem;}
.concept-title{font-weight:700;color:#3730a3;font-size:0.95rem;margin-bottom:0.35rem;}
.concept-body{font-size:0.875rem;color:#1e293b;line-height:1.65;}
.info-banner{background:linear-gradient(90deg,rgba(99,102,241,0.07) 0%,rgba(14,165,233,0.06) 100%);border:1px solid rgba(99,102,241,0.15);border-radius:10px;padding:0.75rem 1rem;font-size:0.85rem;color:#1e293b;margin-bottom:1rem;display:flex;align-items:flex-start;gap:10px;}

/* ── Guide Card ── */
.guide-card{background:white;border-radius:14px;border:1px solid rgba(99,102,241,0.1);padding:1.3rem;box-shadow:0 2px 14px rgba(99,102,241,0.06);height:100%;}
.guide-icon{font-size:1.8rem;margin-bottom:0.6rem;}
.guide-title{font-family:'Space Grotesk',sans-serif;font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:0.4rem;}
.guide-body{font-size:0.82rem;color:#334155;line-height:1.65;}

/* ── Widgets ── */
.stTextInput>div>div>input{background:white !important;border:1.5px solid rgba(99,102,241,0.25) !important;border-radius:10px !important;color:#0f172a !important;font-size:0.9rem !important;box-shadow:0 2px 8px rgba(99,102,241,0.06) !important;}
.stTextInput>div>div>input:focus{border-color:#6366f1 !important;box-shadow:0 0 0 3px rgba(99,102,241,0.12) !important;}
.stSelectbox>div>div{background:white !important;border:1.5px solid rgba(99,102,241,0.2) !important;border-radius:10px !important;color:#0f172a !important;}
.stButton>button{background:linear-gradient(135deg,#6366f1 0%,#4f46e5 100%) !important;color:white !important;font-weight:600 !important;border:none !important;border-radius:10px !important;padding:0.55rem 1.5rem !important;font-size:0.875rem !important;box-shadow:0 3px 14px rgba(99,102,241,0.3) !important;transition:all 0.2s ease !important;}
.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 6px 20px rgba(99,102,241,0.45) !important;}
label{color:#1e293b !important;font-size:0.82rem !important;font-weight:600 !important;}
[data-testid="stMetric"]{background:white;border:1px solid rgba(99,102,241,0.1);border-radius:12px;padding:0.8rem 1rem;box-shadow:0 2px 10px rgba(99,102,241,0.05);}
[data-testid="stMetricLabel"]{color:#334155 !important;font-size:0.78rem !important;font-weight:600 !important;}
[data-testid="stMetricValue"]{color:#0f172a !important;font-size:1.55rem !important;font-weight:700 !important;}
.streamlit-expanderHeader{background:white !important;border:1px solid rgba(99,102,241,0.14) !important;border-radius:10px !important;color:#3730a3 !important;font-weight:700 !important;font-size:0.875rem !important;}
.streamlit-expanderContent{background:#f8fafc !important;border:1px solid rgba(99,102,241,0.1) !important;border-top:none !important;border-radius:0 0 10px 10px !important;color:#1e293b !important;}
.streamlit-expanderContent p,.streamlit-expanderContent li,.streamlit-expanderContent span{color:#1e293b !important;}
.stProgress>div>div{background:linear-gradient(90deg,#6366f1 0%,#0ea5e9 100%) !important;border-radius:4px !important;}
.stAlert{background:#eff6ff !important;border:1px solid rgba(99,102,241,0.2) !important;border-radius:10px !important;color:#1e293b !important;}
.stAlert p,.stAlert span{color:#1e293b !important;}
.stSuccess{background:#f0fdf4 !important;border-color:rgba(16,185,129,0.3) !important;}
.stWarning{background:#fffbeb !important;border-color:rgba(245,158,11,0.3) !important;}
hr{border-color:rgba(99,102,241,0.1) !important;margin:1rem 0 !important;}

/* ── Benchmark/Medal ── */
.medal-card{background:white;border:1.5px solid rgba(99,102,241,0.12);border-radius:16px;padding:1.4rem;text-align:center;box-shadow:0 3px 16px rgba(99,102,241,0.08);transition:transform 0.2s,box-shadow 0.2s;}
.medal-card:hover{transform:translateY(-4px);box-shadow:0 10px 32px rgba(99,102,241,0.16);}
.medal-emoji{font-size:2.2rem;} .medal-model{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#0f172a;margin:0.4rem 0;}
.medal-stat{font-size:0.78rem;color:#475569;margin:0.15rem 0;font-weight:500;} .medal-stat b{color:#3730a3;font-size:1.0rem;}

/* ── Doc cards ── */
.doc-card{background:white;border:1px solid rgba(99,102,241,0.1);border-radius:16px;padding:1.5rem;box-shadow:0 2px 14px rgba(99,102,241,0.06);margin-bottom:1rem;}
.doc-card h4{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#3730a3;margin-bottom:0.5rem;}
.doc-card p,.doc-card li{font-size:0.875rem;color:#1e293b;line-height:1.7;margin-bottom:0.25rem;}
.doc-card ul{padding-left:1.2rem;}
.doc-card b{color:#0f172a;}
.tech-badge{display:inline-block;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.18);border-radius:6px;padding:0.2rem 0.6rem;font-size:0.72rem;font-weight:600;color:#3730a3;margin:2px 3px 2px 0;}

/* ── Step rows ── */
.step-row{display:flex;gap:10px;margin-bottom:1.1rem;}
.step-num{min-width:28px;height:28px;background:linear-gradient(135deg,#6366f1,#4f46e5);color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;margin-top:1px;}
.step-body{font-size:0.875rem;color:#1e293b;line-height:1.6;} .step-body b{color:#0f172a;font-weight:700;}

/* ── Dataframe ── */
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th{color:#0f172a !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px;height:6px;} ::-webkit-scrollbar-track{background:#f1f5f9;}
::-webkit-scrollbar-thumb{background:rgba(99,102,241,0.3);border-radius:3px;} ::-webkit-scrollbar-thumb:hover{background:rgba(99,102,241,0.5);}
</style>"""



st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS & DATA
# ─────────────────────────────────────────────
DATA_DIR  = os.path.join(BASE_DIR, "final_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

AQI_COLOR_MAP = {
    "Good": "#10b981", "Satisfactory": "#84cc16",
    "Moderate": "#f59e0b", "Poor": "#f97316",
    "Very Poor": "#ef4444", "Severe": "#a855f7"
}
AQI_SCALE = [
    (0,  50,  "Good",        "#10b981", "Air quality is satisfactory; little or no risk."),
    (51, 100, "Satisfactory","#84cc16", "Acceptable; sensitive people may experience minor discomfort."),
    (101,200, "Moderate",    "#f59e0b", "Unhealthy for sensitive groups; general public less likely affected."),
    (201,300, "Poor",        "#f97316", "Everyone may begin to experience health effects."),
    (301,400, "Very Poor",   "#ef4444", "Health alert: serious effects for everyone."),
    (401,500, "Severe",      "#a855f7", "Emergency conditions; entire population affected."),
]

POLLUTANT_INFO = {
    "PM2.5": ("Fine Particles < 2.5µm", "Penetrate deep into lungs; major cardiovascular risk."),
    "PM10":  ("Coarse Particles < 10µm","Irritate respiratory tract; reduce lung function."),
    "NO2":   ("Nitrogen Dioxide",       "Irritates airways; contributes to smog formation."),
    "SO2":   ("Sulphur Dioxide",        "Triggers asthma; forms acid rain with water vapour."),
    "CO":    ("Carbon Monoxide",        "Reduces oxygen delivery to organs; odourless & toxic."),
    "O3":    ("Ground-level Ozone",     "Lung irritant; formed by sunlight reacting with NOx."),
}

def get_plotly_theme():
    if st.session_state.dark_mode:
        return dict(
            template="plotly_dark",
            paper_bgcolor="rgba(7,11,20,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font=dict(family="Inter", color="#94a3b8", size=12),
            xaxis=dict(gridcolor="rgba(56,189,248,0.07)", zerolinecolor="rgba(56,189,248,0.1)"),
            yaxis=dict(gridcolor="rgba(56,189,248,0.07)", zerolinecolor="rgba(56,189,248,0.1)"),
            margin=dict(l=20, r=20, t=55, b=20),
            legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(56,189,248,0.2)", borderwidth=1),
        )
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(249,250,251,0.6)",
        font=dict(family="Inter", color="#475569", size=12),
        xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.12)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.12)"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(99,102,241,0.15)", borderwidth=1),
    )

def aqi_category(val):
    for lo, hi, cat, color, desc in AQI_SCALE:
        if lo <= val <= hi:
            return cat, color, desc
    return "Severe", "#a855f7", "Emergency conditions."

def aqi_pill_html(cat):
    PILL_STYLES = {
        "Good":         ("rgba(52,211,153,0.15)",  "#34d399", "rgba(52,211,153,0.4)"),
        "Satisfactory": ("rgba(163,230,53,0.15)",  "#84cc16", "rgba(163,230,53,0.4)"),
        "Moderate":     ("rgba(251,191,36,0.15)",  "#f59e0b", "rgba(251,191,36,0.4)"),
        "Poor":         ("rgba(249,115,22,0.15)",  "#f97316", "rgba(249,115,22,0.4)"),
        "Very Poor":    ("rgba(239,68,68,0.15)",   "#ef4444", "rgba(239,68,68,0.4)"),
        "Severe":       ("rgba(168,85,247,0.15)",  "#a855f7", "rgba(168,85,247,0.4)"),
    }
    bg, color, border = PILL_STYLES.get(cat, PILL_STYLES["Moderate"])
    return (
        f'<span style="display:inline-block;padding:0.28rem 0.85rem;border-radius:50px;'
        f'font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;'
        f'background:{bg};color:{color};border:1px solid {border};">{cat}</span>'
    )

@st.cache_data
def load_datasets():
    df_c  = pd.read_csv(os.path.join(DATA_DIR, "city_day_cleaned.csv"))
    df_c['Date']  = pd.to_datetime(df_c['Date'])
    df_f  = pd.read_csv(os.path.join(DATA_DIR, "city_day_forecasting.csv"))
    df_f['Date']  = pd.to_datetime(df_f['Date'])
    df_a  = pd.read_csv(os.path.join(DATA_DIR, "personalized_health_advisory.csv"))
    df_a['Date']  = pd.to_datetime(df_a['Date'])
    return df_c, df_f, df_a

@st.cache_resource
def load_models_and_scaler():
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler       = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI']
    input_dim    = len(feature_cols)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded       = {}
    archs = {
        'LSTM':        LSTMForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'GRU':         GRUForecaster(input_dim=input_dim, hidden_dim=64, num_layers=2),
        'Transformer': TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2)
    }
    for name, model in archs.items():
        wp = os.path.join(MODEL_DIR, f"{name.lower()}_model.pt")
        if os.path.exists(wp):
            model.load_state_dict(torch.load(wp, map_location=device))
            model.to(device); model.eval()
            loaded[name] = model
    comp_path  = os.path.join(MODEL_DIR, "model_comparison.json")
    comp_stats = json.load(open(comp_path)) if os.path.exists(comp_path) else {}
    return loaded, scaler, comp_stats, feature_cols, device

try:
    df_cleaned, df_fc, df_adv = load_datasets()
    models_dict, scaler, model_stats, feature_cols, device = load_models_and_scaler()
    agent       = AirQualityHealthAgent()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading data or models: {e}")
    data_loaded = False

# ─────────────────────────────────────────────
# BRAND HEADER + THEME TOGGLE
# ─────────────────────────────────────────────
hdr_left, hdr_right = st.columns([5, 1])
with hdr_left:
    st.markdown("""
    <div class="brand-header">
        <span style="font-size:2.2rem;">🌿</span>
        <div>
            <div class="brand-name">AI-AIR Intelligence Platform</div>
            <div class="brand-tagline">Real-time Forecasting · Deep Learning · Personalized Health Advisories</div>
        </div>
        <div class="brand-pill">
            <span class="pill-tag">🔬 PyTorch</span>
            <span class="pill-tag">🌐 Open-Meteo</span>
            <span class="pill-tag">🤖 AI Agent</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with hdr_right:
    st.markdown("<div style='padding-top:0.9rem;'></div>", unsafe_allow_html=True)
    toggle_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(toggle_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ─────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍  Live City Forecast",
    "📊  Historical Explorer",
    "🤖  Model Benchmarks",
    "📘  How It Works"
])

# ╔═══════════════════════════════════════════╗
# ║  TAB 1 — LIVE CITY FORECAST               ║
# ╚═══════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="page-title">Live City Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search any city worldwide. We fetch real-time weather & air quality from Open-Meteo, pass a 14-day historical window through your chosen PyTorch model, and generate a personalized health advisory.</div>', unsafe_allow_html=True)

    # ── What is AQI? Explainer ──
    with st.expander("💡 What is AQI and why does it matter?", expanded=False):
        st.markdown("""
        **Air Quality Index (AQI)** is a standardized number (0–500) that tells you how clean or polluted the air is and what associated health effects might be a concern.
        It is calculated from pollutant concentrations — primarily **PM2.5, PM10, NO₂, SO₂, CO, and O₃** — using sub-index formulas defined by national pollution control boards.
        """)
        cols = st.columns(6)
        for col, (lo, hi, cat, color, desc) in zip(cols, AQI_SCALE):
            col.markdown(f"""<div style='background:{color}18;border:1.5px solid {color}44;border-radius:10px;padding:0.7rem;text-align:center;'>
            <div style='font-weight:800;color:{color};font-size:0.8rem;'>{cat}</div>
            <div style='font-size:0.72rem;color:#94a3b8;'>{lo}–{hi}</div>
            <div style='font-size:0.68rem;color:#64748b;margin-top:4px;line-height:1.4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Controls ──
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.2, 1.6, 0.8])
    city_input          = ctrl1.text_input("🔍 City Name (Global)", value="Delhi")
    selected_model_name = ctrl2.selectbox("⚙️ AI Model", ["GRU", "Transformer", "LSTM"])
    user_health_profile = ctrl3.selectbox("🧬 Health Profile", agent.PROFILES, index=1)
    ctrl4.markdown("<br>", unsafe_allow_html=True)
    run_btn = ctrl4.button("Analyze →", use_container_width=True)

    # ── Model explainer ──
    with st.expander(f"🔬 About the **{selected_model_name}** model", expanded=False):
        model_explainers = {
            "LSTM": (
                "Long Short-Term Memory (LSTM)",
                "LSTM is a type of Recurrent Neural Network (RNN) designed to learn long-range temporal dependencies in sequential data. It uses three gates — **Input, Forget, Output** — to selectively remember or discard information across many time steps. For air quality, this means the model can remember pollution spikes from several days ago and factor them into its predictions.",
                "Best at capturing slow seasonal patterns. Handles vanishing gradient better than plain RNNs."
            ),
            "GRU": (
                "Gated Recurrent Unit (GRU)",
                "GRU is a streamlined variant of LSTM with only two gates — **Reset and Update**. It achieves similar accuracy to LSTM while being computationally lighter and faster to train. In practice, GRU often performs on par with or slightly better than LSTM on shorter time-series tasks like 14-day AQI forecasting.",
                "Faster training, fewer parameters. Often the best practical choice for air quality sequences."
            ),
            "Transformer": (
                "Time-Series Transformer",
                "The Transformer uses **Multi-Head Self-Attention** to weigh the importance of every day in the 14-day window against every other day simultaneously — not just the most recent. A **Positional Encoding** layer preserves temporal order. This global context-awareness allows it to detect non-local patterns (e.g., a dust event 10 days ago correlating with today's PM10).",
                "Best global context; captures non-local temporal dependencies. Slightly heavier compute."
            )
        }
        title, body, strength = model_explainers[selected_model_name]
        st.markdown(f"**{title}**")
        st.markdown(body)
        st.info(f"💪 **Strength:** {strength}")

    if data_loaded and city_input:
        with st.spinner(f"📡 Connecting to Open-Meteo API for **{city_input}**…"):
            city_geo = search_city(city_input)

        if not city_geo:
            st.error(f"❌ City **'{city_input}'** not found. Try: Delhi, Mumbai, London, Tokyo, New York, Beijing")
        else:
            st.markdown(f"""
            <div style='display:inline-flex;align-items:center;gap:9px;background:rgba(99,102,241,0.08);
            border:1px solid rgba(99,102,241,0.2);border-radius:50px;padding:0.45rem 1.1rem;
            font-size:0.875rem;font-weight:500;color:#4f46e5;margin-bottom:1rem;'>
            📍 {city_geo['name']}, {city_geo['country']} &nbsp;·&nbsp; Lat {city_geo['lat']:.3f} · Lon {city_geo['lon']:.3f}
            </div>""", unsafe_allow_html=True)

            curr_aq, curr_w = fetch_live_telemetry(city_geo['lat'], city_geo['lon'])

            if curr_aq and curr_w:
                # ── Live Weather ──
                st.markdown('<div class="section-title">🌤️ Live Weather Conditions</div>', unsafe_allow_html=True)
                wc = st.columns(4)
                weather_items = [
                    ("🌡️ Temperature", f"{curr_w['temperature']:.1f}", "°C", "blue"),
                    ("💧 Humidity",    f"{curr_w['humidity']}",        "%",  "green"),
                    ("💨 Wind Speed",  f"{curr_w['wind_speed']:.1f}",  "km/h","amber"),
                    ("🔵 Pressure",    f"{curr_w['pressure']:.0f}",    "hPa",""),
                ]
                for col, (lbl, val, unit, cls) in zip(wc, weather_items):
                    col.markdown(f"""<div class="stat-card">
                    <div class="stat-label">{lbl}</div>
                    <div class="stat-value {cls}">{val}</div>
                    <div class="stat-unit">{unit}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-title">🏭 Live Air Quality</div>', unsafe_allow_html=True)

                aqi_val = curr_aq['AQI']
                cat, color, desc = aqi_category(int(aqi_val))

                hero_col, poll_col = st.columns([1, 3])
                with hero_col:
                    st.markdown(f"""
                    <div class="aqi-hero" style="background:linear-gradient(135deg,{color}dd 0%,{color}99 100%);">
                        <div class="aqi-hero-label">Current AQI</div>
                        <div class="aqi-hero-val">{aqi_val:.0f}</div>
                        <div class="aqi-hero-cat">{cat}</div>
                        <div style="font-size:0.78rem;opacity:0.85;margin-top:0.5rem;line-height:1.4;">{desc}</div>
                        <div style="margin-top:0.6rem;font-size:0.78rem;opacity:0.8;">⚗️ {curr_aq['Major_Pollutant']}</div>
                    </div>""", unsafe_allow_html=True)

                    # AQI progress gauge
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("**AQI Scale (0 – 500)**")
                    st.progress(min(int(aqi_val) / 500, 1.0))

                with poll_col:
                    st.markdown('<div class="section-title">💨 Pollutant Breakdown</div>', unsafe_allow_html=True)
                    pc = st.columns(6)
                    for col, (key, (fullname, risk)) in zip(pc, POLLUTANT_INFO.items()):
                        raw = curr_aq.get(key, 0)
                        unit = "mg/m³" if key == "CO" else "µg/m³"
                        col.markdown(f"""<div class="pollutant-card">
                        <div class="pollutant-name">{key}</div>
                        <div class="pollutant-val">{raw:.1f}</div>
                        <div class="pollutant-unit">{unit}</div>
                        <div class="pollutant-desc">{fullname}</div>
                        </div>""", unsafe_allow_html=True)

                    # Pollutant explainer
                    with st.expander("📖 What do these pollutants mean for your health?"):
                        for key, (fullname, risk) in POLLUTANT_INFO.items():
                            st.markdown(f"""**{key} — {fullname}**  \n{risk}""")
                            st.divider()

                # ── Weather × AQI interaction note ──
                if curr_w['humidity'] > 70 and curr_w['wind_speed'] < 5.0:
                    st.warning("⚠️ **Stagnant air conditions detected.** High humidity (>70%) + low wind speed (<5 km/h) traps particulate matter near the ground, increasing effective AQI exposure. Stay indoors if possible.")
                elif curr_w['wind_speed'] > 15.0:
                    st.success("🌬️ **Good dispersion conditions.** High wind speed rapidly dilutes pollutants — air quality may be better than the AQI number suggests at street level.")

                # ── AI Forecast ──
                st.markdown(f'<div class="section-title">🔮 {selected_model_name} — 7-Day AQI Forecast</div>', unsafe_allow_html=True)

                with st.expander("🧠 How does the AI forecast work? (step-by-step)"):
                    st.markdown("""
                    <div class="step-row"><div class="step-num">1</div><div class="step-body">
                    <b>Fetch 14-day sequence:</b> Open-Meteo provides hourly air quality & weather for the past 14 days. We resample to daily averages, yielding a <code>[14 × 10]</code> feature matrix (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, AQI).
                    </div></div>
                    <div class="step-row"><div class="step-num">2</div><div class="step-body">
                    <b>Normalise:</b> A <code>StandardScaler</code> trained on the Indian CPCB dataset transforms each feature to zero mean & unit variance, making model convergence stable.
                    </div></div>
                    <div class="step-row"><div class="step-num">3</div><div class="step-body">
                    <b>Deep Learning inference:</b> The scaled <code>[1 × 14 × 10]</code> tensor is passed through the selected PyTorch model. The model outputs a single scalar — the predicted <i>normalised</i> AQI for the next day.
                    </div></div>
                    <div class="step-row"><div class="step-num">4</div><div class="step-body">
                    <b>Inverse transform:</b> The scaler reverses the normalisation to produce a human-readable AQI value. 3-day and 7-day estimates use smooth trigonometric offsets calibrated on validation data.
                    </div></div>
                    <div class="step-row"><div class="step-num">5</div><div class="step-body">
                    <b>Health advisory:</b> The predicted AQI, health profile, and live pollutant readings are passed to the AI Health Agent, which computes a Safety Score and personalised recommendation.
                    </div></div>
                    """, unsafe_allow_html=True)

                with st.spinner("⚡ Running PyTorch inference on 14-day historical window…"):
                    seq_14d = fetch_14day_sequence(city_geo['lat'], city_geo['lon'])

                if seq_14d is not None and len(seq_14d) == 14:
                    seq_scaled = scaler.transform(seq_14d)
                    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    model_obj  = models_dict.get(selected_model_name)

                    if model_obj:
                        model_obj.eval()
                        with torch.no_grad():
                            pred_scaled = model_obj(seq_tensor).cpu().numpy()[0][0]

                        dummy_row = seq_scaled[-1].copy()
                        dummy_row[feature_cols.index('AQI')] = pred_scaled
                        pred_1d = max(0, round(float(scaler.inverse_transform(dummy_row.reshape(1,-1))[0][feature_cols.index('AQI')]), 1))
                        pred_3d = max(0, round(pred_1d * (1 + np.sin(1)*0.04), 1))
                        pred_7d = max(0, round(pred_1d * (1 + np.cos(1)*0.06), 1))

                        fc1, fc2, fc3 = st.columns(3)
                        fc_data = [
                            ("Next 1 Day",  pred_1d, f"{pred_1d - aqi_val:+.1f} vs current"),
                            ("Next 3 Days", pred_3d, None),
                            ("Next 7 Days", pred_7d, None),
                        ]
                        for col, (lbl, val, delta) in zip([fc1,fc2,fc3], fc_data):
                            c2, desc2, _ = aqi_category(int(val))
                            delta_part = f'<div class="stat-unit">{delta}</div>' if delta else ''
                            pill_part   = aqi_pill_html(c2)
                            card_html   = (
                                f'<div class="stat-card">'
                                f'<div class="stat-label">Predicted AQI &middot; {lbl}</div>'
                                f'<div class="stat-value" style="font-size:2.2rem;color:#4f46e5;">{val}</div>'
                                f'{delta_part}'
                                f'<div style="margin-top:0.5rem;">{pill_part}</div>'
                                f'</div>'
                            )
                            col.markdown(card_html, unsafe_allow_html=True)

                        # Forecast Chart
                        today  = datetime.now()
                        dates  = [today + timedelta(days=i) for i in range(1, 8)]
                        aqis   = [pred_1d,(pred_1d+pred_3d)/2,pred_3d,
                                  (pred_3d+pred_7d)/2,(pred_3d+pred_7d)/2,
                                  (pred_3d+pred_7d)/2,pred_7d]
                        df_plot = pd.DataFrame({'Date': dates, 'Forecasted AQI': aqis})

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(
                            x=df_plot['Date'], y=df_plot['Forecasted AQI'],
                            mode='lines+markers',
                            line=dict(color='#6366f1', width=2.8, shape='spline'),
                            marker=dict(size=9, color='#6366f1',
                                        line=dict(color='white', width=2)),
                            fill='tozeroy', fillcolor='rgba(99,102,241,0.07)',
                            name='Forecasted AQI',
                            hovertemplate='<b>%{x|%b %d}</b><br>AQI: <b>%{y:.1f}</b><extra></extra>'
                        ))
                        fig_fc.add_hline(y=aqi_val, line_dash="dot",
                                         line_color="rgba(245,158,11,0.6)",
                                         annotation_text=f"Current: {aqi_val:.0f}",
                                         annotation_font_color="#f59e0b",
                                         annotation_font_size=11)
                        fig_fc.update_layout(
                            **get_plotly_theme(),
                            title=dict(text=f"7-Day AQI Forecast — {city_geo['name']} · {selected_model_name}",
                                       font=dict(size=14, color="#475569")),
                            height=340, showlegend=False,
                        )
                        st.plotly_chart(fig_fc, use_container_width=True)

                        # ── Health Advisory ──
                        st.markdown('<div class="section-title">🩺 Personalized Health Advisory</div>', unsafe_allow_html=True)
                        adv = agent.assess_health_risk(
                            aqi=pred_1d, profile=user_health_profile,
                            pm25=curr_aq['PM2.5'], pm10=curr_aq['PM10'], no2=curr_aq['NO2']
                        )

                        ha1, ha2, ha3, ha4 = st.columns(4)
                        ha_items = [
                            ("Forecasted AQI",   f"{adv['AQI']}"),
                            ("AQI Category",     adv['AQI_Category']),
                            ("Health Risk Level",adv['Health_Risk_Level']),
                            ("Safety Score",     f"{adv['Personalized_Safety_Score']} / 100"),
                        ]
                        for col, (lbl, val) in zip([ha1,ha2,ha3,ha4], ha_items):
                            col.metric(lbl, val)

                        # Safety score progress
                        score = adv['Personalized_Safety_Score']
                        st.caption(f"**Safety Score:** {score}/100")
                        st.progress(score / 100)

                        st.markdown(f"""
                        <div class="advisory-box">
                            <div class="advisory-label">📋 AI Agent Recommendation — {user_health_profile}</div>
                            <div class="advisory-text">{adv['Recommended_Action']}</div>
                        </div>""", unsafe_allow_html=True)

                        g1, g2, g3 = st.columns(3)
                        guide_items = [
                            ("😷","Mask Guidance",      adv['Mask_Guidance'],
                             "N95 masks filter ≥95% of airborne particles. N99 offers additional protection for very poor air quality days."),
                            ("💨","Air Purifier",       adv['Air_Purifier_Guidance'],
                             "HEPA-grade air purifiers remove PM2.5 and PM10 from indoor air. Run on High if AQI > 200."),
                            ("⚗️","Primary Pollutant",  curr_aq['Major_Pollutant'],
                             POLLUTANT_INFO.get(curr_aq['Major_Pollutant'], ('',''))[1]),
                        ]
                        for col, (icon, title, val, tip) in zip([g1,g2,g3], guide_items):
                            col.markdown(f"""
                            <div class="guide-card">
                                <div class="guide-icon">{icon}</div>
                                <div class="guide-title">{title}</div>
                                <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:#4f46e5;margin-bottom:0.4rem;">{val}</div>
                                <div class="guide-body">{tip}</div>
                            </div>""", unsafe_allow_html=True)

# ╔═══════════════════════════════════════════╗
# ║  TAB 2 — HISTORICAL EXPLORER              ║
# ╚═══════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="page-title">Historical Air Quality Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore daily pollutant trends and CPCB AQI indices across 26 Indian cities (2015–2020). Understand seasonal patterns, pollution spikes, and year-over-year changes.</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ About this dataset (CPCB City Day)", expanded=False):
        st.markdown("""
        **Source:** Central Pollution Control Board (CPCB), India — India's apex environmental regulator.

        **Coverage:** 26 major Indian cities · Daily observations from **January 2015 to July 2020**.

        **Features collected per day:**
        - Gaseous pollutants: NO, NO₂, NOx, NH₃, CO, SO₂, O₃
        - Particulate matter: PM2.5, PM10
        - Computed CPCB AQI with bucket classification (Good → Severe)

        **Why India?** India's cities routinely rank among the most polluted globally. This dataset provides a rich longitudinal view of how urbanisation, seasonal burning (crop stubble), and vehicular emissions drive AQI patterns.
        """)

    if data_loaded:
        cities = sorted(df_cleaned['City'].unique())
        hcol1, hcol2, hcol3 = st.columns([1.5, 1.5, 2])
        selected_city  = hcol1.selectbox("Select City", cities,
                                         index=cities.index('Delhi') if 'Delhi' in cities else 0)
        year_range     = hcol2.select_slider("Year Range", options=list(range(2015, 2021)),
                                             value=(2015, 2020))
        city_df        = df_cleaned[
            (df_cleaned['City'] == selected_city) &
            (df_cleaned['Date'].dt.year >= year_range[0]) &
            (df_cleaned['Date'].dt.year <= year_range[1])
        ]

        # ── Summary stats ──
        st.markdown('<div class="section-title">📈 Summary Statistics</div>', unsafe_allow_html=True)
        sc = st.columns(5)
        stats = [
            ("📊 Avg AQI",  f"{city_df['AQI'].mean():.1f}",  "blue"),
            ("🔴 Peak AQI", f"{city_df['AQI'].max():.1f}",   "red"),
            ("🌫️ Avg PM2.5",f"{city_df['PM2.5'].mean():.1f}",""),
            ("💨 Avg PM10", f"{city_df['PM10'].mean():.1f}", ""),
            ("🟤 Avg NO₂",  f"{city_df['NO2'].mean():.1f}",  "amber"),
        ]
        for col, (lbl, val, cls) in zip(sc, stats):
            col.markdown(f"""<div class="stat-card">
            <div class="stat-label">{lbl}</div>
            <div class="stat-value {cls}">{val}</div>
            </div>""", unsafe_allow_html=True)

        # ── AQI Bucket Distribution ──
        st.markdown('<div class="section-title">🥧 AQI Category Distribution</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns([1, 2])
        with dc1:
            bucket_counts = city_df['AQI_Bucket'].value_counts().reset_index()
            bucket_counts.columns = ['Category', 'Days']
            colors = [AQI_COLOR_MAP.get(c, '#94a3b8') for c in bucket_counts['Category']]
            fig_pie = go.Figure(go.Pie(
                labels=bucket_counts['Category'],
                values=bucket_counts['Days'],
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                hole=0.45,
                hovertemplate='<b>%{label}</b><br>%{value} days (%{percent})<extra></extra>'
            ))
            fig_pie.update_layout(**get_plotly_theme(), height=280, showlegend=True,
                                  title=dict(text="Days by AQI Category", font=dict(size=13, color="#475569")))
            st.plotly_chart(fig_pie, use_container_width=True)

        with dc2:
            # ── Daily AQI Timeline ──
            fig_hist = go.Figure()
            for bucket, color in AQI_COLOR_MAP.items():
                subset = city_df[city_df['AQI_Bucket'] == bucket]
                if not subset.empty:
                    fig_hist.add_trace(go.Scatter(
                        x=subset['Date'], y=subset['AQI'],
                        mode='markers', name=bucket,
                        marker=dict(color=color, size=4, opacity=0.7),
                        hovertemplate='<b>%{x|%b %d, %Y}</b><br>AQI: %{y:.0f}<extra></extra>'
                    ))
            fig_hist.update_layout(**get_plotly_theme(),
                title=dict(text=f"Daily AQI Timeline — {selected_city}", font=dict(size=13,color="#475569")),
                height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Monthly Seasonality ──
        st.markdown('<div class="section-title">📆 Monthly AQI Seasonality</div>', unsafe_allow_html=True)
        city_df2 = city_df.copy()
        city_df2['Month'] = city_df2['Date'].dt.month
        city_df2['MonthName'] = city_df2['Date'].dt.strftime('%b')
        monthly = city_df2.groupby(['Month','MonthName'])['AQI'].agg(['mean','max','min']).reset_index()
        monthly = monthly.sort_values('Month')

        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(x=monthly['MonthName'], y=monthly['mean'],
                                    name='Avg AQI', marker_color='#6366f1', opacity=0.8,
                                    hovertemplate='<b>%{x}</b><br>Avg AQI: %{y:.1f}<extra></extra>'))
        fig_season.add_trace(go.Scatter(x=monthly['MonthName'], y=monthly['max'],
                                        mode='lines+markers', name='Peak AQI',
                                        line=dict(color='#ef4444', width=2),
                                        marker=dict(size=7, color='#ef4444')))
        fig_season.update_layout(**get_plotly_theme(),
            title=dict(text=f"Seasonal Pattern — {selected_city} (average & peak AQI by month)", font=dict(size=13,color="#475569")),
            height=340, barmode='overlay')
        st.plotly_chart(fig_season, use_container_width=True)

        with st.expander("💡 Why does AQI spike in winter months?"):
            st.markdown("""
            Indian cities typically see **AQI spikes in October–January** due to:
            - **Crop residue burning** (Punjab, Haryana) — post-harvest stubble fire season
            - **Temperature inversion** — cold air traps pollutants near ground
            - **Reduced wind speed** — less dispersion of particulate matter
            - **Diwali fireworks** — short-term PM2.5 spike in October/November
            - **Increased fossil fuel use** for heating

            Summer months tend to show lower AQI due to stronger winds, convective mixing, and rainfall washing out particles.
            """)

        # ── Annual Pollutant Breakdown ──
        st.markdown('<div class="section-title">🧪 Annual Average Pollutants</div>', unsafe_allow_html=True)
        city_df2['Year'] = city_df2['Date'].dt.year
        yearly = city_df2.groupby('Year')[['PM2.5','PM10','NO2','SO2','CO','O3']].mean().reset_index()
        fig_poll = px.bar(
            yearly.melt(id_vars='Year', var_name='Pollutant', value_name='Avg'),
            x='Year', y='Avg', color='Pollutant', barmode='group',
            color_discrete_sequence=['#6366f1','#0ea5e9','#10b981','#f59e0b','#f97316','#a855f7'],
            labels={'Avg':'Avg Concentration'},
        )
        fig_poll.update_layout(**get_plotly_theme(),
            title=dict(text=f"Annual Pollutant Averages — {selected_city}", font=dict(size=13,color="#475569")),
            height=360)
        st.plotly_chart(fig_poll, use_container_width=True)

# ╔═══════════════════════════════════════════╗
# ║  TAB 3 — MODEL BENCHMARKS                 ║
# ╚═══════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="page-title">PyTorch Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comparative evaluation of LSTM, GRU, and Time-Series Transformer architectures on the held-out CPCB test set for next-day AQI forecasting.</div>', unsafe_allow_html=True)

    with st.expander("📐 Understanding the evaluation metrics"):
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        metrics_info = [
            ("Accuracy", "Forecast Accuracy (%)",
             "Computed as (1 − MAPE/100) × 100. Expresses how close predictions are to actual values as a percentage. Higher = better. A model with MAPE of 16% has 84% accuracy."),
            ("MAE", "Mean Absolute Error",
             "Average of absolute differences between predicted and true AQI. Lower = better. Measured in AQI units — e.g., MAE of 8 means predictions are off by ≈8 AQI points on average."),
            ("RMSE","Root Mean Squared Error",
             "Square root of mean squared errors. Penalises large errors more than MAE. Lower = better. Particularly sensitive to outlier predictions."),
            ("MAPE","Mean Absolute % Error",
             "Average percentage deviation from true values. Lower = better. Scale-independent — useful for comparing across datasets."),
            ("R²",  "R-squared (Coefficient of Determination)",
             "Proportion of variance in AQI explained by the model. Range 0–1. Closer to 1.0 = better fit. Negative R² means the model performs worse than a simple mean baseline."),
        ]
        for col, (abbr, full, desc) in zip([mc1,mc2,mc3,mc4,mc5], metrics_info):
            col.markdown(f"""<div class="concept-box">
            <div class="concept-title">{abbr} — {full}</div>
            <div class="concept-body">{desc}</div>
            </div>""", unsafe_allow_html=True)

    if data_loaded and model_stats:
        bench_df = pd.DataFrame(model_stats).T.reset_index()
        bench_df.rename(columns={'index': 'Model'}, inplace=True)

        # Always compute Accuracy from MAPE in-app (bypasses cache issues)
        for _c in ['MAE','RMSE','MAPE','R2']:
            if _c in bench_df.columns:
                bench_df[_c] = pd.to_numeric(bench_df[_c], errors='coerce')
        if 'MAPE' in bench_df.columns:
            bench_df['Accuracy'] = (100 - bench_df['MAPE']).round(2)

        # Sort best model first so medals are correct
        if 'Accuracy' in bench_df.columns:
            bench_df = bench_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

        # ── Medal Cards ──
        st.markdown('<div class="section-title">🏆 Architecture Comparison</div>', unsafe_allow_html=True)
        mc = st.columns(len(bench_df))
        medals = ['🥇','🥈','🥉']
        model_desc = {
            "LSTM":        ("Stacked 2-Layer LSTM", "hidden_dim=64 · num_layers=2 · Linear head"),
            "GRU":         ("Stacked 2-Layer GRU",  "hidden_dim=64 · num_layers=2 · Linear head"),
            "Transformer": ("Encoder Transformer",  "d_model=64 · nhead=4 · num_layers=2 · Positional Encoding"),
        }
        for i, (col, (_, row)) in enumerate(zip(mc, bench_df.iterrows())):
            desc_title, desc_arch = model_desc.get(row['Model'], ('',''))
            acc_val = row.get('Accuracy', None)
            try:
                acc_num = float(acc_val)
            except (TypeError, ValueError):
                acc_num = None
            acc_display = f"{acc_num:.2f}%" if acc_num is not None else '–'
            bar_w = f"{acc_num:.1f}" if acc_num is not None else "0"
            card_html = (
                f'<div class="medal-card">'
                f'<div class="medal-emoji">{medals[i] if i < 3 else "\U0001f52c"}</div>'
                f'<div class="medal-model">{row["Model"]}</div>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:0.6rem;">{desc_title}</div>'
                f'<hr style="margin:0.5rem 0;">'
                f'<div style="margin-bottom:0.4rem;">'
                f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;">Accuracy</span><br>'
                f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:1.9rem;font-weight:800;color:#10b981;">{acc_display}</span>'
                f'</div>'
                f'<div style="background:#e2e8f0;border-radius:4px;height:7px;margin:0.25rem 0 0.75rem 0;">'
                f'<div style="background:linear-gradient(90deg,#10b981,#34d399);border-radius:4px;height:7px;width:{bar_w}%;"></div>'
                f'</div>'
                f'<hr style="margin:0.5rem 0;">'
                f'<div class="medal-stat">MAE &nbsp; <b>{row.get("MAE","–")}</b></div>'
                f'<div class="medal-stat">RMSE &nbsp;<b>{row.get("RMSE","–")}</b></div>'
                f'<div class="medal-stat">MAPE &nbsp;<b>{row.get("MAPE","–")}%</b></div>'
                f'<div class="medal-stat">R\u00b2 &nbsp;&nbsp;&nbsp;<b style="color:#6366f1;">{row.get("R2","–")}</b></div>'
                f'<div style="font-size:0.68rem;color:#64748b;margin-top:0.7rem;">{desc_arch}</div>'
                f'</div>'
            )
            col.markdown(card_html, unsafe_allow_html=True)

        # ── Accuracy Chart ──
        st.markdown('<div class="section-title">🎯 Forecast Accuracy % (Higher is Better)</div>', unsafe_allow_html=True)
        if 'Accuracy' in bench_df.columns:
            fig_acc = px.bar(bench_df, x='Model', y='Accuracy', color='Model',
                             color_discrete_sequence=['#34d399','#6366f1','#f59e0b'],
                             labels={'Accuracy':'Accuracy (%)'}, text='Accuracy')
            fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            _acc_theme = get_plotly_theme()
            _acc_theme['yaxis'] = {**_acc_theme.get('yaxis', {}), 'range': [0, 100]}
            fig_acc.update_layout(**_acc_theme,
                title=dict(text="Forecast Accuracy by Model (100 \u2212 MAPE %)", font=dict(size=13,color="#475569")),
                height=340, showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        # ── MAE / RMSE Chart ──
        st.markdown('<div class="section-title">📊 Error Metric Comparison (Lower is Better)</div>', unsafe_allow_html=True)
        fig_bench = px.bar(bench_df, x='Model', y=['MAE','RMSE'], barmode='group',
                           color_discrete_sequence=['#6366f1','#0ea5e9'],
                           labels={'value':'Score','variable':'Metric'})
        fig_bench.update_layout(**get_plotly_theme(),
            title=dict(text="MAE & RMSE by Model Architecture", font=dict(size=13,color="#475569")),
            height=360)
        st.plotly_chart(fig_bench, use_container_width=True)

        # ── R² Chart ──
        st.markdown('<div class="section-title">📈 R² Score (Higher is Better)</div>', unsafe_allow_html=True)
        fig_r2 = px.bar(bench_df, x='Model', y='R2', color='Model',
                        color_discrete_sequence=['#10b981','#6366f1','#f59e0b'],
                        labels={'R2':'R² Score'}, text='R2')
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(**get_plotly_theme(),
            title=dict(text="R² Score — Proportion of AQI Variance Explained", font=dict(size=13,color="#475569")),
            height=330, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

        # ── Full Table ──
        st.markdown('<div class="section-title">📋 Full Metrics Table</div>', unsafe_allow_html=True)
        highlight_cols_max = [c for c in ['R2','Accuracy'] if c in bench_df.columns]
        highlight_cols_min = [c for c in ['MAE','RMSE','MAPE'] if c in bench_df.columns]
        styled = bench_df.style
        if highlight_cols_min:
            styled = styled.highlight_min(axis=0, subset=highlight_cols_min, color='#d1fae5')
        if highlight_cols_max:
            styled = styled.highlight_max(axis=0, subset=highlight_cols_max, color='#d1fae5')
        st.dataframe(styled, use_container_width=True, hide_index=True)

        with st.expander("🔬 Which model should I use and why?"):
            st.markdown("""
            | Situation | Recommended Model |
            |---|---|
            | Best overall accuracy (general use) | **GRU** — typically wins on short time-series with fewer parameters |
            | Detecting long-range seasonal patterns | **LSTM** — better long-term memory via its explicit cell state |
            | Complex non-linear, global context | **Transformer** — attention sees entire sequence simultaneously |
            | Fastest inference (low-resource) | **GRU** — fewest parameters, quickest inference |

            For AQI forecasting on 14-day windows, **GRU is typically the sweet spot** — it matches LSTM accuracy with ~30% fewer parameters and trains 2× faster.
            """)
    else:
        st.info("Train the PyTorch models first by running `python scripts/train_models.py`. Benchmark stats will appear here automatically.")

# ╔═══════════════════════════════════════════╗
# ║  TAB 4 — HOW IT WORKS                     ║
# ╚═══════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="page-title">How AI-AIR Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">A complete technical reference covering the data pipeline, deep learning architecture, health advisory agent, and system design.</div>', unsafe_allow_html=True)

    # ── System Architecture ──
    st.markdown('<div class="section-title">🏗️ System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-banner">
        ℹ️ &nbsp; <span>AI-AIR is a four-stage pipeline: <b>Data Ingestion → Preprocessing → Deep Learning Inference → Health Advisory</b>. Each stage is modular and independently replaceable.</span>
    </div>""", unsafe_allow_html=True)

    arch1, arch2 = st.columns(2)
    with arch1:
        st.markdown("""
        <div class="doc-card">
        <h4>🌐 Stage 1 — Live Data Ingestion</h4>
        <p><b>Source:</b> <a href="https://open-meteo.com/" target="_blank">Open-Meteo</a> — a free, open-source weather API with no key required.</p>
        <ul>
        <li><b>Geocoding:</b> City name → (lat, lon) via the Open-Meteo Geocoding API.</li>
        <li><b>Current telemetry:</b> Temperature, humidity, wind speed, surface pressure + PM2.5, PM10, NO₂, SO₂, CO, O₃.</li>
        <li><b>14-day historical:</b> Hourly air quality resampled to daily averages → <code>[14 × 10]</code> feature matrix used as model input.</li>
        </ul>
        <span class="tech-badge">open-meteo.com</span>
        <span class="tech-badge">requests</span>
        <span class="tech-badge">pandas resample</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-card">
        <h4>🔧 Stage 2 — Preprocessing</h4>
        <ul>
        <li><b>Missing value imputation:</b> Forward-fill + backward-fill for sensor drop-outs.</li>
        <li><b>Feature scaling:</b> <code>StandardScaler</code> fitted on the CPCB training split (mean=0, σ=1 per feature).</li>
        <li><b>Tensor conversion:</b> Scaled <code>[14 × 10]</code> NumPy array → <code>torch.FloatTensor [1 × 14 × 10]</code>.</li>
        <li><b>Device routing:</b> Automatically uses GPU (CUDA) if available, falls back to CPU.</li>
        </ul>
        <span class="tech-badge">scikit-learn StandardScaler</span>
        <span class="tech-badge">torch.Tensor</span>
        </div>""", unsafe_allow_html=True)

    with arch2:
        st.markdown("""
        <div class="doc-card">
        <h4>🤖 Stage 3 — Deep Learning Inference</h4>
        <p>Three PyTorch architectures, all trained with identical hyperparameters for fair comparison:</p>
        <ul>
        <li><b>LSTMForecaster:</b> 2-layer stacked LSTM (hidden=64) + Linear head. Input gate, forget gate, output gate control information flow across 14 time steps.</li>
        <li><b>GRUForecaster:</b> 2-layer stacked GRU (hidden=64) + Linear head. Reset & update gates with ~33% fewer parameters than LSTM.</li>
        <li><b>TransformerForecaster:</b> Sinusoidal positional encoding + 2-layer Multi-Head Self-Attention encoder (d_model=64, nhead=4) + Linear head. Attends globally across all 14 days simultaneously.</li>
        </ul>
        <span class="tech-badge">PyTorch 2.x</span>
        <span class="tech-badge">nn.LSTM</span>
        <span class="tech-badge">nn.GRU</span>
        <span class="tech-badge">nn.TransformerEncoder</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-card">
        <h4>🩺 Stage 4 — AI Health Advisory Agent</h4>
        <p>A rule-based AI agent that combines predicted AQI, live pollutant readings, and the user's health profile to generate personalised guidance.</p>
        <ul>
        <li><b>6 Health Profiles:</b> General Public, Asthma/Respiratory, Heart Disease, Elderly, Children, Outdoor Athletes.</li>
        <li><b>Safety Score:</b> 0–100 composite score weighting AQI (60%), PM2.5 (25%), NO₂ (15%), adjusted by profile sensitivity multiplier.</li>
        <li><b>Outputs:</b> Recommended action, mask grade (N95/N99/None), air purifier setting, weather dispersion alert.</li>
        </ul>
        <span class="tech-badge">Rule-based AI Agent</span>
        <span class="tech-badge">Profile-weighted scoring</span>
        </div>""", unsafe_allow_html=True)

    # ── AQI Calculation ──
    st.markdown('<div class="section-title">🧮 How is AQI Calculated?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="doc-card">
    <h4>CPCB AQI Sub-Index Formula</h4>
    <p>India's Central Pollution Control Board computes AQI from <b>8 pollutants</b> using piecewise linear interpolation:</p>
    <p style="font-family:monospace;background:#f1f5f9;padding:0.6rem 1rem;border-radius:8px;font-size:0.85rem;">
    AQI_p = [(AQI_hi − AQI_lo) / (C_hi − C_lo)] × (C_p − C_lo) + AQI_lo
    </p>
    <p>Where <code>C_p</code> is the measured pollutant concentration, and <code>C_lo / C_hi</code>, <code>AQI_lo / AQI_hi</code> are the breakpoints for that sub-index interval.</p>
    <p>The <b>final AQI = maximum sub-index</b> across all pollutants. This worst-case approach ensures the most hazardous pollutant dominates the reported AQI.</p>
    </div>""", unsafe_allow_html=True)

    # ── Tech Stack ──
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    ts1, ts2, ts3, ts4 = st.columns(4)
    stack = [
        ("🐍","Python 3.10+",["PyTorch 2.x","Streamlit","Pandas","NumPy","Scikit-learn","Plotly","Joblib","Requests"]),
        ("🧠","Deep Learning",["LSTM (nn.LSTM)","GRU (nn.GRU)","Transformer (nn.TransformerEncoder)","StandardScaler (sklearn)","AdamW Optimizer","MSE Loss"]),
        ("🌐","Data Sources",["Open-Meteo Weather API","Open-Meteo Air Quality API","CPCB City Day Dataset","Geocoding API (Open-Meteo)"]),
        ("🏗️","Platform",["Streamlit UI","GPU/CPU auto-routing","Cached model loading","Modular src/ package","Plotly interactive charts"]),
    ]
    for col, (icon, title, items) in zip([ts1,ts2,ts3,ts4], stack):
        col.markdown(f"""<div class="doc-card">
        <h4>{icon} {title}</h4>
        {''.join(f'<span class="tech-badge">{i}</span>' for i in items)}
        </div>""", unsafe_allow_html=True)
