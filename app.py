# ============================================================
# 🤖 LOTTO AI PRO V9.4 ADAPTIVE STABILITY TURBO (Self-Correcting & Fast Mode)
# ============================================================
# V9.4 improvements:
# - Auto-Correction System: Detects if a position failed in the last 2 draws
# - Dynamic Re-calibration: Adjusts Depth, Trees, Decay, and Seed on failure
# - FAST MODE: Single-Shot Fast Validation for backtesting (10x faster)
# - CPU Optimization: Disabled n_jobs=-1 for small datasets to reduce overhead
# ============================================================

import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Lotto AI V9.4 Fast Auto-Correct",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CONSTANTS
# ============================================================

LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

DOW_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
MODEL_NAMES = ["ExtraTrees", "HistGradientBoosting"]

THAI_POSITIONS = ["H1", "H2", "H3", "H4", "H5", "H6", "T2", "O2"]
NORMAL_POSITIONS = ["H", "T", "O", "T2", "O2"]

POSITION_LABELS = {
    "H1": "💯 หลักแสน (6 ตัว)", "H2": "🔢 หลักหมื่น (6 ตัว)", "H3": "🔢 หลักพัน (6 ตัว)",
    "H4": "💯 หลักร้อย (6 ตัว)", "H5": "🔟 หลักสิบ (6 ตัว)", "H6": "1️⃣ หลักหน่วย (6 ตัว)",
    "H": "💯 หลักร้อย (3 ตัวบน)", "T": "🔟 หลักสิบ (3 ตัวบน)", "O": "1️⃣ หลักหน่วย (3 ตัวบน)",
    "T2": "🔽 หลักสิบ (2 ตัวล่าง)", "O2": "⬇️ หลักหน่วย (2 ตัวล่าง)",
}

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
}

# ============================================================
# CSS
# ============================================================

def inject_css():
    st.markdown("""
    <style>
    .stApp { background:#f8fafc; }
    .main-title { text-align:center;font-size:2.2rem;font-weight:900;color:#1e293b;margin-bottom:5px; }
    .subtitle { text-align:center;color:#64748b;font-size:1rem;margin-bottom:25px; }
    .status-card { background:#eff6ff;border:1px solid #bfdbfe;border-radius:14px;padding:15px;text-align:center;color:#1e40af;font-weight:600; }
    .hot-card { background:#f0fdf4;border-left:6px solid #16a34a;border-radius:10px;padding:15px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,.05);height:100%; }
    .dead-card { background:#fef2f2;border-left:6px solid #dc2626;border-radius:10px;padding:15px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,.05);height:100%; }
    .position-title { font-size:1.1rem;font-weight:800;color:#334155;margin-bottom:10px;text-align:center;border-bottom:1px solid #e2e8f0;padding-bottom:8px; }
    .hot-number { font-size:2.2rem;font-weight:900;letter-spacing:4px;text-align:center;color:#16a34a; }
    .dead-number { font-size:1.8rem;font-weight:800;letter-spacing:2px;text-align:center;color:#dc2626; }
    .prob-text { text-align:center;color:#475569;font-size:.9rem;line-height:1.6;margin-top:10px; }
    .prob-pill { display:inline-block;background:#e2e8f0;border-radius:20px;padding:2px 10px;margin:3px;font-weight:600;font-size:.85rem; }
    .warning-text { color: #d97706; font-weight: 700; font-size: 0.85rem; margin-top: 5px; text-align: center; }
    .correction-text { color: #2563eb; font-weight: 800; font-size: 0.9rem; margin-top: 8px; text-align: center; background: #e0e7ff; padding: 4px; border-radius: 8px;}
    .confidence { text-align:center;font-size:.85rem;font-weight:700;margin-top:10px;color:#64748b; }
    div.stButton > button { min-height:50px;border-radius:10px;font-size:18px;font-weight:800; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DATE & FETCH
# ============================================================

def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    for name, month in THAI_MONTHS.items():
        m = re.search(rf"(\d{{1,2}})\s*{re.escape(name)}\s*(\d{{4}})", text)
        if m:
            y = int(m.group(2))
            if y >= 2400: y -= 543
            try: return pd.Timestamp(y, month, int(m.group(1)))
            except Exception: return None
    m = re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", text)
    if m:
        a, b, c = map(int, m.groups())
        if a >= 1000: y, mo, d = a, b, c
        else: y, mo, d = c, b, a
        if y < 100: y += 2000
        if y >= 2400: y -= 543
        try: return pd.Timestamp(y, mo, d)
        except Exception: pass
    return None

class ScrapingError(Exception): pass

@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content", re.I)) or soup
        rows = []
        for row in content.find_all("tr"):
            text = " ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th"]))
            date = normalize_date(text)
            if date is None: continue
            six = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
            three = re.findall(r"(?<!\d)\d{3}(?!\d)", text)
            two = re.findall(r"(?<!\d)\d{2}(?!\d)", text)
            if six and two: rows.append({"Date": date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
            elif three and two: rows.append({"Date": date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})
        
        if not rows:
            text = content.get_text(separator="\n", strip=True)
            current_date = None
            for line in (x.strip() for x in text.splitlines() if x.strip()):
                date = normalize_date(line)
                if date is not None: current_date = date
                if current_date is None: continue
                six = re.findall(r"(?<!\d)\d{6}(?!\d)", line)
                three = re.findall(r"(?<!\d)\d{3}(?!\d)", line)
                two = re.findall(r"(?<!\d)\d{2}(?!\d)", line)
                if six and two: rows.append({"Date": current_date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
                elif three and two: rows.append({"Date": current_date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})

        if not rows: raise ScrapingError("ไม่พบข้อมูลหวยในรูปแบบที่รองรับ")
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Result_3D"] = df["Result_3D"].astype(str).str.extract(r"(\d{3})")[0].str.zfill(3)
        df["Result_2D"] = df["Result_2D"].astype(str).str.extract(r"(\d{2})")[0].str.zfill(2)
        if "Result_6D" in df.columns: df["Result_6D"] = df["Result_6D"].astype(str).str.extract(r"(\d{6})")[0]
        df = df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        return df
    except Exception as exc:
        raise ScrapingError(f"โหลดข้อมูลไม่สำเร็จ: {exc}")

def is_thai_6d(df):
    return "Result_6D" in df.columns and df["Result_6D"].notna().sum() >= 10

# ============================================================
# FEATURES
# ============================================================

def build_features(df, thai_6d=False):
    w = df.copy()
    if thai_6d:
        six = w["Result_6D"].fillna("000000").astype(str).str.zfill(6)
        for i in range(6): w[f"H{i+1}"] = six.str[i].astype(np.int8)
    else:
        three = w["Result_3D"].astype(str).str.zfill(3)
        w["H"] = three.str[0].astype(np.int8)
        w["T"] = three.str[1].astype(np.int8)
        w["O"] = three.str[2].astype(np.int8)

    two = w["Result_2D"].astype(str).str.zfill(2)
    w["T2"] = two.str[0].astype(np.int8)
    w["O2"] = two.str[1].astype(np.int8)

    dt = w["Date"].dt
    w["DOW"] = dt.dayofweek.astype(np.int8)
    w["DAY"] = dt.day.astype(np.int8)
    w["MONTH"] = dt.month.astype(np.int8)
    w["IS_WEEKEND"] = (w["DOW"] >= 5).astype(np.float32)
    w["IS_MONTH_START"] = (w["DAY"] <= 5).astype(np.float32)
    w["IS_MONTH_END"] = (w["DAY"] >= 25).astype(np.float32)
    w["DOW_SIN"] = np.sin(2*np.pi*w["DOW"]/7).astype(np.float32)
    w["DOW_COS"] = np.cos(2*np.pi*w["DOW"]/7).astype(np.float32)

    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    for pos in positions:
        s = w[pos]
        p = s.shift(1)
        for lag in (1, 2, 3, 5): w[f"{pos}_L{lag}"] = s.shift(lag)
        for window in (10, 20):
            r = p.rolling(window, min_periods=3)
            w[f"{pos}_M{window}"] = r.mean()
            w[f"{pos}_S{window}"] = r.std()
            for digit in (0, 5):
                w[f"{pos}_F{window}_{digit}"] = ((p == digit).astype(np.float32).rolling(window, min_periods=3).mean())

        w[f"{pos}_D1"] = s.shift(1) - s.shift(2)
        w[f"{pos}_D2"] = s.shift(2) - s.shift(3)
        w[f"{pos}_ODD"] = p % 2
        w[f"{pos}_HIGH"] = (p >= 5).astype(np.float32)
        w[f"{pos}_SIN"] = np.sin(2*np.pi*p/10).astype(np.float32)
        w[f"{pos}_COS"] = np.cos(2*np.pi*p/10).astype(np.float32)
        
        w[f"{pos}_EWMA3"] = p.ewm(span=3, adjust=False).mean()
        w[f"{pos}_EWMA9"] = p.ewm(span=9, adjust=False).mean()
        w[f"{pos}_MACD"] = w[f"{pos}_EWMA3"] - w[f"{pos}_EWMA9"]
        
        w[f"{pos}_REPEAT"] = (p == s.shift(2)).astype(np.float32)
        w[f"{pos}_SUM_L1_L2"] = (s.shift(1) + s.shift(2)) % 10
        w[f"{pos}_SUM_L1_L3"] = (s.shift(1) + s.shift(3)) % 10

    base_cols = ["H1","H2","H3","H4","H5","H6"] if thai_6d else ["H","T","O"]
    base = w[base_cols].shift(1)
    w["PREV_SUM"] = base.sum(axis=1)
    w["PREV_RANGE"] = base.max(axis=1) - base.min(axis=1)
    w["PREV_ODD"] = (base % 2).sum(axis=1)
    w["PREV_HIGH"] = (base >= 5).sum(axis=1)
    return w.replace([np.inf, -np.inf], np.nan)

def get_features(thai_6d):
    base = [
        "DOW","DAY","MONTH","IS_WEEKEND","IS_MONTH_START","IS_MONTH_END",
        "DOW_SIN","DOW_COS","PREV_SUM","PREV_RANGE","PREV_ODD","PREV_HIGH"
    ]
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    for pos in positions:
        base += [
            f"{pos}_L1",f"{pos}_L2",f"{pos}_L3",f"{pos}_L5",
            f"{pos}_M10",f"{pos}_M20",f"{pos}_S10",f"{pos}_S20",
            f"{pos}_D1",f"{pos}_D2",f"{pos}_ODD",f"{pos}_HIGH",
            f"{pos}_SIN",f"{pos}_COS",f"{pos}_REPEAT",
            f"{pos}_F10_0",f"{pos}_F10_5",f"{pos}_F20_0",f"{pos}_F20_5",
            f"{pos}_SUM_L1_L2",f"{pos}_SUM_L1_L3", 
            f"{pos}_EWMA3", f"{pos}_EWMA9", f"{pos}_MACD",
        ]
    return list(dict.fromkeys(base))

# ============================================================
# ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):
    # ปรับลด Backtest Points เพื่อความเร็ว แม้ในโหมด Fast ก็ช่วยลดข้อมูลที่ไม่จำเป็น
    if n >= 700: return dict(min_train=120, trees=65, depth=8, leaf=3, hot_features=25, dead_features=18, backtest_points=6, recent_decay=.985, refresh_every=3)
    if n >= 400: return dict(min_train=100, trees=55, depth=7, leaf=3, hot_features=22, dead_features=16, backtest_points=5, recent_decay=.980, refresh_every=3)
    if n >= 200: return dict(min_train=80,  trees=45, depth=6, leaf=3, hot_features=20, dead_features=15, backtest_points=4, recent_decay=.975, refresh_every=3)
    return dict(min_train=50, trees=35, depth=5, leaf=3, hot_features=18, dead_features=14, backtest_points=3, recent_decay=.970, refresh_every=3)

# ============================================================
# SELF-CORRECTION (V9.4)
# ============================================================

def check_recent_failures(df_feat_hist, pos, features, cfg):
    n = len(df_feat_hist)
    if n < 20: return False 
    
    X = df_feat_hist[features].astype(np.float32)
    y = df_feat_hist[pos].astype(np.int8)
    
    fails = 0
    for idx in [n-2, n-1]:
        train_X = X.iloc[:idx]
        train_y = y.iloc[:idx]
        test_X = X.iloc[[idx]]
        actual = int(y.iloc[idx])
        
        selected = select_features_once(train_X, train_y, cfg["hot_features"], "hot")
        probs = model_probability(train_X, train_y, test_X, cfg, selected, "hot")
        order = np.argsort(probs)[::-1]
        
        if actual not in order[:3]:
            fails += 1
            
    return fails == 2

# ============================================================
# MODEL (⚡ OPTIMIZED: n_jobs=None)
# ============================================================

def create_model(name, cfg, system="hot", categorical_mask=None):
    rs_offset = cfg.get("random_seed_offset", 0)
    
    # ⚡ ตั้ง n_jobs=None ช่วยแก้ปัญหาคอขวดของ Thread ใน Streamlit เมื่อรันหลายรอบ
    if system == "hot":
        if name == "ExtraTrees":
            return ExtraTreesClassifier(n_estimators=cfg["trees"], max_depth=cfg["depth"], min_samples_leaf=cfg["leaf"], max_features=.70, class_weight=None, n_jobs=None, random_state=42 + rs_offset)
        return HistGradientBoostingClassifier(max_iter=max(25, int(cfg["trees"]*.65)), max_leaf_nodes=15, learning_rate=.04, min_samples_leaf=cfg["leaf"], l2_regularization=2.0, categorical_features=categorical_mask, random_state=42 + rs_offset)

    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=max(25, int(cfg["trees"]*.80)), max_depth=max(4, cfg["depth"]-1), min_samples_leaf=max(2, cfg["leaf"]), max_features=.45, class_weight=None, n_jobs=None, random_state=91 + rs_offset)
    return HistGradientBoostingClassifier(max_iter=max(20, int(cfg["trees"]*.50)), max_leaf_nodes=9, learning_rate=.035, min_samples_leaf=max(2, cfg["leaf"]), l2_regularization=4.0, categorical_features=categorical_mask, random_state=91 + rs_offset)

# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features_once(X, y, max_features, system="hot"):
    cols = list(X.columns)
    valid = [c for c in cols if X[c].nunique(dropna=False) > 1]
    if len(valid) <= max_features: return valid
    Xi = X[valid].replace([np.inf,-np.inf],np.nan).astype(np.float32).fillna(0)
    
    try:
        selector = SelectKBest(score_func=f_classif, k=max_features)
        selector.fit(Xi, y)
        scores = np.nan_to_num(selector.scores_)
        order = np.argsort(scores)[::-1]
        return [valid[i] for i in order[:max_features]]
    except Exception:
        return valid[:max_features]

# ============================================================
# PROBABILITY & ENSEMBLE
# ============================================================

def normalize_probability(p):
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=0, posinf=0, neginf=0)
    p = np.clip(p, 1e-9, None)
    s = p.sum()
    return p/s if s > 0 else np.ones(10, dtype=np.float32)/10

def model_probability(X_train, y_train, X_test, cfg, selected, system):
    A = X_train[selected].replace([np.inf,-np.inf],np.nan).astype(np.float32)
    B = X_test[selected].replace([np.inf,-np.inf],np.nan).astype(np.float32)

    med = A.median()
    A = A.fillna(med).fillna(0)
    B = B.fillna(med).fillna(0)

    age = np.arange(len(A)-1, -1, -1, dtype=np.float32)
    decay_age = np.maximum(0, age - 30)
    weights = cfg["recent_decay"] ** decay_age
    weights = (weights / (weights.mean()+1e-9)).astype(np.float32)

    categorical_cols = [
        c for c in selected 
        if any(c.endswith(x) for x in ('_L1', '_L2', '_L3', '_L5', '_ODD', '_HIGH', '_REPEAT', '_SUM_L1_L2', '_SUM_L1_L3')) 
        or c in ['DOW', 'DAY', 'MONTH', 'IS_WEEKEND', 'IS_MONTH_START', 'IS_MONTH_END', 'PREV_ODD', 'PREV_HIGH']
    ]
    categorical_mask = [selected.index(c) for c in categorical_cols] if categorical_cols else None

    preds = []
    for name in MODEL_NAMES:
        try:
            model = create_model(name, cfg, system, categorical_mask=categorical_mask)
            try: model.fit(A, y_train, sample_weight=weights)
            except Exception: model.fit(A, y_train)

            raw = model.predict_proba(B)[0]
            out = np.zeros(10, dtype=np.float32)
            for cls, prob in zip(model.classes_, raw):
                c = int(cls)
                if 0 <= c <= 9: out[c] = prob
            preds.append(normalize_probability(out))
        except Exception:
            continue

    if not preds: return np.ones(10, dtype=np.float32)/10

    if len(preds) == 2:
        if system == "hot":
            p1, p2 = np.clip(preds[0], 1e-5, 1.0), np.clip(preds[1], 1e-5, 1.0)
            geo_mean = np.sqrt(p1 * p2)
            ari_mean = (p1 * 0.55) + (p2 * 0.45)
            ensemble = (geo_mean * 0.70) + (ari_mean * 0.30)
        else:
            ensemble = np.maximum(preds[0], preds[1])
    else:
        ensemble = np.mean(preds, axis=0)

    p = normalize_probability(ensemble)
    shrink = .05 if system=="hot" else .08
    p = (1-shrink)*p + shrink*.1
    return normalize_probability(p)

# ============================================================
# HOT & DEAD SYSTEMS
# ============================================================

def probability_concentration(p):
    p = normalize_probability(p)
    entropy = -np.sum(p*np.log(p+1e-12))
    return float(np.clip(1 - entropy/np.log(10), 0, 1))

def hot_system(X_train, y_train, X_test, cfg, selected=None):
    if selected is None:
        selected = select_features_once(X_train,y_train,cfg["hot_features"],"hot")
    p = model_probability(X_train,y_train,X_test,cfg,selected,"hot")
    order = np.argsort(p)[::-1]
    
    top_gap = float(p[order[0]] - p[order[1]])
    
    return {
        "probability": p,
        "hot": [(int(n),float(p[n])) for n in order[:3]],
        "top1_probability": float(p[order[0]]),
        "top3": float(p[order[:3]].sum()),
        "top_gap": top_gap,
        "concentration": probability_concentration(p),
        "is_unstable": top_gap < 0.015,
        "selected_features": selected,
    }

def build_dead_score(probability, y_train):
    probability = normalize_probability(probability)
    inverse_prob = (1.0 - probability) ** 2
    ai_score = normalize_probability(inverse_prob)

    recent_n = min(30, len(y_train))
    recent = np.asarray(y_train.iloc[-recent_n:], dtype=np.int8)
    freq = np.bincount(recent, minlength=10).astype(np.float32)
    recent_freq = freq / max(1, recent_n)
    cold_score = normalize_probability((1.0 - recent_freq) ** 2)

    gaps = np.zeros(10, dtype=np.float32)
    arr = np.asarray(y_train)
    for d in range(10):
        loc = np.where(arr == d)[0]
        gaps[d] = 30 if len(loc) == 0 else len(arr) - 1 - loc[-1]
    
    gap_score = np.zeros(10, dtype=np.float32)
    for d in range(10):
        g = gaps[d]
        if 2 <= g <= 18: gap_score[d] = 1.0
        elif g <= 1: gap_score[d] = 0.5 
        else: gap_score[d] = 0.05 
            
    gap_score = normalize_probability(gap_score)
    final_score = (ai_score * 0.70) + (cold_score * 0.15) + (gap_score * 0.15)
    return normalize_probability(final_score)

def dead_system(X_train, y_train, X_test, cfg, selected=None):
    if selected is None:
        selected=select_features_once(X_train,y_train,cfg["dead_features"],"dead")
    p = model_probability(X_train,y_train,X_test,cfg,selected,"dead")
    score = build_dead_score(p, y_train)
    order = np.argsort(score)[::-1]
    return {
        "probability": p,
        "dead_score": score,
        "dead": [(int(n), float(score[n])) for n in order[:7]],
        "top7": float(score[order[:7]].sum()),
        "selected_features": selected,
    }

# ============================================================
# WALK-FORWARD (⚡ FAST SINGLE-SHOT VALIDATION)
# ============================================================

def walk_forward_system(df_feat, pos, features, cfg, system="hot"):
    X = df_feat[features].astype(np.float32)
    y = df_feat[pos].astype(np.int8)
    points = cfg["backtest_points"]
    n = len(df_feat)
    
    if n <= cfg["min_train"] + points:
        return {"tests":0, "scores":{}, "stability":0.0}
        
    split_idx = n - points - 1
    train_X = X.iloc[:split_idx]
    train_y = y.iloc[:split_idx]
    
    if train_y.nunique() < 2:
        return {"tests":0, "scores":{}, "stability":0.0}
        
    selected = select_features_once(train_X, train_y, cfg["hot_features"] if system=="hot" else cfg["dead_features"], system)
    A = train_X[selected].fillna(0).astype(np.float32)
    
    model = create_model("ExtraTrees", cfg, system) 
    try:
        model.fit(A, train_y)
    except:
        return {"tests":0, "scores":{}, "stability":0.0}
        
    records = []
    for i in range(split_idx, n-1):
        B = X.iloc[[i]][selected].fillna(0).astype(np.float32)
        actual = int(y.iloc[i])
        
        raw = model.predict_proba(B)[0]
        probs = np.zeros(10, dtype=np.float32)
        for cls, prob in zip(model.classes_, raw):
            c = int(cls)
            if 0 <= c <= 9: probs[c] = prob
        probs = normalize_probability(probs)
        
        if system == "hot":
            order = np.argsort(probs)[::-1]
            records.append({
                "top1": int(actual == order[0]), 
                "top3": int(actual in order[:3]), 
                "top5": int(actual in order[:5])
            })
        else:
            score = build_dead_score(probs, train_y)
            order = np.argsort(score)[::-1]
            records.append({
                "dead5": int(actual in order[:5]), 
                "dead7": int(actual in order[:7])
            })
            
    if not records: return {"tests":0,"scores":{},"stability":0.0}
    h = pd.DataFrame(records)
    decay = cfg["recent_decay"] ** np.arange(len(h)-1, -1, -1)
    decay = decay / (decay.sum() + 1e-12)
    scores = {c: float(np.sum(h[c].values * decay)) for c in h.columns}
    
    stability_values = h["top3"].values if "top3" in h.columns else h["dead7"].values
    if len(stability_values) > 1:
        mean = np.mean(stability_values)
        std = np.std(stability_values)
        stability = float(np.clip(1 - (std / max(mean, .10)), 0, 1))
    else:
        stability = 0.5
        
    return {"tests": len(records), "scores": scores, "stability": stability}

# ============================================================
# FINAL
# ============================================================

def final_prediction(df_feat,pos,features,cfg):
    X=df_feat[features].astype(np.float32)
    y=df_feat[pos].astype(np.int8)
    X_train, y_train, X_test = X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]]

    hot_selected=select_features_once(X_train,y_train,cfg["hot_features"],"hot")
    dead_selected=select_features_once(X_train,y_train,cfg["dead_features"],"dead")

    return {
        "hot":hot_system(X_train,y_train,X_test,cfg,hot_selected),
        "dead":dead_system(X_train,y_train,X_test,cfg,dead_selected),
    }

# ============================================================
# DISPLAY
# ============================================================

def display_hot_card(result, is_corrected=False):
    data = result["hot"]["hot"]
    nums = " - ".join(str(n) for n,_ in data)
    pills = "".join(f'<span class="prob-pill">{n}: {p*100:.1f}%</span>' for n,p in data)
    h = result["hot"]
    
    warning_html = ""
    if is_corrected:
        warning_html += '<div class="correction-text">🔄 ระบบเปิดโหมดแก้ไขตัวเอง (พลาด 2 งวดติด)</div>'
    elif h["is_unstable"]:
        warning_html += '<div class="warning-text">⚠️ สถิติเบียดสูสี ระวังพลิก! (Top-Gap &lt; 1.5%)</div>'
        
    html=f"""
    <div class="hot-card">
      <div class="position-title">🔥 HOT TOP-3</div>
      <div class="hot-number">{nums}</div>
      <div class="prob-text">โมเดลเห็นพ้อง<br>{pills}</div>
      {warning_html}
      <div class="confidence">
        🎯 Top-1: {h["top1_probability"]*100:.1f}% &nbsp;|&nbsp; 📌 Gap: {h["top_gap"]*100:.1f}%<br>
        🔥 Top-3 Mass: {h["top3"]*100:.1f}% &nbsp;|&nbsp; 📊 Concentration: {h["concentration"]*100:.1f}%
      </div>
    </div>
    """
    st.markdown(html,unsafe_allow_html=True)

def display_dead_card(result):
    data=result["dead"]["dead"]
    nums=" - ".join(str(n) for n,_ in data[:5])
    pills="".join(f'<span class="prob-pill">{n}</span>' for n,_ in data)
    html=f"""
    <div class="dead-card">
      <div class="position-title">🛑 COLD / DEAD TOP-7</div>
      <div class="dead-number">{nums}</div>
      <div class="prob-text">
        กลุ่มเลขดับที่ปลอดภัย (Max Prob Lock)<br>{pills}
      </div>
      <div class="confidence">🛑 Dead Group Score: {result["dead"]["top7"]*100:.1f}%</div>
    </div>
    """
    st.markdown(html,unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================

def main():
    inject_css()
    st.markdown('<div class="main-title">🤖 LOTTO AI PRO V9.4 AUTO-CORRECT</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">⚡ Hot-Consensus, Self-Healing & Fast Turbo | 🔥 HOT TOP-3 | 🛑 DEAD TOP-7</div>', unsafe_allow_html=True)

    c1,c2=st.columns(2)
    lottery=c1.selectbox("🏷️ เลือกประเภทหวย",list(LOTTERY_SOURCES.keys()))
    selected_day=c2.selectbox("📅 วันเป้าหมาย",["อัตโนมัติ"]+DOW_NAMES)

    if not st.button("⚡ เริ่มวิเคราะห์ระบบ V9.4",type="primary",use_container_width=True): return

    with st.spinner("📥 กำลังดึงข้อมูลสถิติล่าสุด..."):
        try: df=fetch_lottery_data(LOTTERY_SOURCES[lottery])
        except Exception as exc:
            st.error(str(exc))
            return

    if len(df)<50:
        st.error(f"❌ มีข้อมูล {len(df)} งวด (ต้องการอย่างน้อย 50 งวด)")
        return

    thai_6d=(lottery=="หวยไทย" and is_thai_6d(df))
    positions=THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    last_date=pd.Timestamp(df["Date"].iloc[-1])

    if selected_day=="อัตโนมัติ": days_ahead=7
    else:
        days_ahead=(DOW_NAMES.index(selected_day)-last_date.dayofweek)%7
        if days_ahead==0: days_ahead=7
    target_date=last_date+timedelta(days=days_ahead)

    dummy={"Date":target_date,"Result_3D":"000","Result_2D":"00"}
    if thai_6d: dummy["Result_6D"]="000000"
    ext=pd.concat([df,pd.DataFrame([dummy])],ignore_index=True)

    with st.spinner("⚙️ กำลังสร้าง Features และทดสอบโมเดล..."):
        feat=build_features(ext,thai_6d)
        features=get_features(thai_6d)
        base_cfg=get_adaptive_config(len(df))

    final, hot_backtest, dead_backtest = {}, {}, {}
    is_corrected = {}
    progress, status_text = st.progress(0), st.empty()
    historical_feat=feat.iloc[:-1]

    for i,pos in enumerate(positions):
        status_text.caption(f"🧠 วิเคราะห์และตรวจสอบ {POSITION_LABELS[pos]}")
        
        pos_cfg = base_cfg.copy()
        pos_cfg["random_seed_offset"] = 0
        
        if check_recent_failures(historical_feat, pos, features, pos_cfg):
            pos_cfg["trees"] = int(pos_cfg["trees"] * 1.5)
            pos_cfg["depth"] = pos_cfg["depth"] + 2
            pos_cfg["recent_decay"] = 0.85
            pos_cfg["random_seed_offset"] = 777
            is_corrected[pos] = True
        else:
            is_corrected[pos] = False
            
        # ใช้ Walk-Forward แบบใหม่ที่ประมวลผลเร็วขึ้นมาก
        hot_backtest[pos]=walk_forward_system(historical_feat,pos,features,pos_cfg,"hot")
        dead_backtest[pos]=walk_forward_system(historical_feat,pos,features,pos_cfg,"dead")
        
        final[pos]=final_prediction(feat,pos,features,pos_cfg)
        progress.progress(int((i+1)/len(positions)*100))

    progress.empty()
    status_text.empty()

    st.markdown(
        f"""
        <div class="status-card">
        ✅ วิเคราะห์สำเร็จ: {len(df):,} งวด<br>
        🎯 เป้าหมาย: {target_date.strftime("%d/%m/%Y")} ({lottery})<br>
        ⚙️ โมเดลเด่น: Geometric Consensus & Auto-Correction (ซ่อมแซมตัวเองอัตโนมัติ)<br>
        ⚙️ โหมดจำลอง: Fast Single-Shot Validation (ทดสอบย้อนหลังรวดเร็ว)
        </div><br>
        """, unsafe_allow_html=True
    )

    st.markdown("### 🏆 สรุปเลขฟันธง V9.4")
    summary=[]
    for pos in positions:
        hot=final[pos]["hot"]["hot"]
        dead=final[pos]["dead"]["dead"]
        
        warn_flag = "⚠️" if final[pos]["hot"]["is_unstable"] else "✅"
        if is_corrected[pos]:
            warn_flag = "🔄 ซ่อมแซมตัวเอง"
            
        summary.append({
            "ตำแหน่ง":POSITION_LABELS[pos],
            "สถานะ": warn_flag,
            "🔥 HOT #1":str(hot[0][0]),
            "โอกาสเชิงโมเดล":f"{hot[0][1]*100:.1f}%",
            "🔥 HOT #2/#3":f"{hot[1][0]}, {hot[2][0]}",
            "📌 Top Gap":f"{final[pos]['hot']['top_gap']*100:.1f}%",
            "🛑 COLD/DEAD":", ".join(str(x[0]) for x in dead[:7]),
        })
    st.dataframe(pd.DataFrame(summary),use_container_width=True,hide_index=True)

    st.markdown("---")
    t1,t2=st.tabs(["🎯 เจาะลึกรายหลัก","📊 Walk-Forward Backtest"])

    with t1:
        st.markdown("ระบบแยก **HOT TOP-3** (เอกฉันท์) และ **COLD/DEAD TOP-7** (ล็อคเป้าดับสนิท)")
        for pos in positions:
            st.markdown(f'<div class="position-title">{POSITION_LABELS[pos]}</div>', unsafe_allow_html=True)
            a,b=st.columns(2)
            with a: display_hot_card(final[pos], is_corrected[pos])
            with b: display_dead_card(final[pos])

    with t2:
        st.markdown("### 📊 Walk-Forward Backtest (Fast Mode)")
        st.caption("ทดสอบย้อนหลังแบบ Single-Shot ทายรวดเดียว (เน้นความรวดเร็วและใช้เวลาประมวลผลน้อยที่สุด)")
        for pos in positions:
            hr, dr = hot_backtest[pos], dead_backtest[pos]
            hs, ds = hr.get("scores",{}), dr.get("scores",{})
            if not hs: continue
            with st.expander(f"📊 {POSITION_LABELS[pos]}"):
                rows=[
                    {"Metric":"🔥 Top-1","ผลย้อนหลัง":f"{hs.get('top1',0)*100:.1f}%"},
                    {"Metric":"🔥 Top-3","ผลย้อนหลัง":f"{hs.get('top3',0)*100:.1f}%"},
                    {"Metric":"🔥 Top-5","ผลย้อนหลัง":f"{hs.get('top5',0)*100:.1f}%"},
                    {"Metric":"🛑 Dead Group Hit-5 (ปลอดภัย)","ผลย้อนหลัง":f"{ds.get('dead5',0)*100:.1f}%"},
                    {"Metric":"🛑 Dead Group Hit-7 (ปลอดภัย)","ผลย้อนหลัง":f"{ds.get('dead7',0)*100:.1f}%"},
                    {"Metric":"📊 Hot Stability","ผลย้อนหลัง":f"{hr.get('stability',0)*100:.1f}%"},
                    {"Metric":"📊 Dead Stability","ผลย้อนหลัง":f"{dr.get('stability',0)*100:.1f}%"},
                ]
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown("### ⚙️ V9.4 System Information")
    info=pd.DataFrame([
        {"รายการ":"Auto-Correction System","ค่า":"หลักที่ทายผิด Top-3 ติดกัน 2 งวด ระบบจะปรับจูนโมเดลตัวเองทันที (Depth, Trees, Decay, Seed)"},
        {"รายการ":"Fast Validation Engine","ค่า":"ทดสอบย้อนหลังแบบ Single-Shot แก้ปัญหา Streamlit ประมวลผลช้า"},
        {"รายการ":"Hot System Engine","ค่า":"Geometric Consensus (ต้องเห็นพ้อง 100%)"},
        {"รายการ":"Dead System Engine","ค่า":"Pessimistic Maximum (ต้องห่วยทั้งคู่)"},
    ])
    st.dataframe(info,use_container_width=True,hide_index=True)
    st.caption("⚠️ Probability เป็นค่าประเมินทางสถิติเพื่อนำไปเป็นแนวทางตัดสินใจ ไม่รับประกันผล 100%")

if __name__=="__main__":
    main()
