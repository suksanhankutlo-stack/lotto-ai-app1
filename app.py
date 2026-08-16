# ============================================================
# 🤖 LOTTO AI PRO V9.0 TURBO EXTREME (Upgraded from V8.3.1)
# ============================================================
# HOT TOP-3 + DEAD TOP-7 EDITION
# IMPROVED ACCURACY & CLEAN UI
# ============================================================

import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Lotto AI V9.0 Turbo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. DATA SOURCES
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

# ============================================================
# 3. CONSTANTS
# ============================================================
DOW_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
MODEL_NAMES = ["ExtraTrees", "HistGradientBoosting"]

THAI_POSITIONS = ["H1", "H2", "H3", "H4", "H5", "H6", "T2", "O2"]
NORMAL_POSITIONS = ["H", "T", "O", "T2", "O2"]

POSITION_LABELS = {
    "H1": "💯 หลักแสน (6 ตัว)",
    "H2": "🔢 หลักหมื่น (6 ตัว)",
    "H3": "🔢 หลักพัน (6 ตัว)",
    "H4": "💯 หลักร้อย (6 ตัว)",
    "H5": "🔟 หลักสิบ (6 ตัว)",
    "H6": "1️⃣ หลักหน่วย (6 ตัว)",
    "H": "💯 หลักร้อย (3 ตัวบน)",
    "T": "🔟 หลักสิบ (3 ตัวบน)",
    "O": "1️⃣ หลักหน่วย (3 ตัวบน)",
    "T2": "🔽 หลักสิบ (2 ตัวล่าง)",
    "O2": "⬇️ หลักหน่วย (2 ตัวล่าง)",
}

# ============================================================
# 4. CSS (IMPROVED UI)
# ============================================================
def inject_css():
    st.markdown("""
        <style>
        .stApp { background:#f8fafc; }
        .main-title { text-align:center; font-size:2.2rem; font-weight:900; color:#1e293b; margin-bottom:5px; }
        .subtitle { text-align:center; color:#64748b; font-size:1rem; margin-bottom:25px; }
        .status-card { background:#eff6ff; border:1px solid #bfdbfe; border-radius:14px; padding:15px; text-align:center; color:#1e40af; font-weight:600; }
        
        .hot-card { background:#f0fdf4; border-left:6px solid #16a34a; border-radius:10px; padding:15px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;}
        .dead-card { background:#fef2f2; border-left:6px solid #dc2626; border-radius:10px; padding:15px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;}
        
        .position-title { font-size:1.1rem; font-weight:800; color:#334155; margin-bottom: 10px; text-align: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;}
        .hot-number { font-size:2.2rem; font-weight:900; letter-spacing:4px; text-align:center; color:#16a34a; }
        .dead-number { font-size:1.8rem; font-weight:800; letter-spacing:2px; text-align:center; color:#dc2626; }
        
        .prob-text { text-align:center; color:#475569; font-size:0.9rem; line-height:1.6; margin-top:10px; }
        .prob-pill { display: inline-block; background: #e2e8f0; border-radius: 20px; padding: 2px 10px; margin: 3px; font-weight: 600; font-size: 0.85rem;}
        
        .confidence { text-align:center; font-size:0.85rem; font-weight:700; margin-top:10px; color:#64748b; }
        div.stButton > button { min-height:50px; border-radius:10px; font-size:18px; font-weight:800; }
        </style>
        """, unsafe_allow_html=True)

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "ม.ค.": 1, "ก.พ.": 2, "มี.ค.": 3, "เม.ย.": 4, "พ.ค.": 5, "มิ.ย.": 6,
    "ก.ค.": 7, "ส.ค.": 8, "ก.ย.": 9, "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12
}

# ============================================================
# 5. DATA FETCHING & NORMALIZING
# ============================================================
def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    for name, month in THAI_MONTHS.items():
        match = re.search(rf"(\d{{1,2}})\s*{re.escape(name)}\s*(\d{{4}})", text)
        if match:
            y = int(match.group(2))
            if y >= 2400: y -= 543
            try: return pd.Timestamp(y, month, int(match.group(1)))
            except: return None

    match = re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", text)
    if match:
        a, b, c = map(int, match.groups())
        y, m, d = (a, b, c) if a >= 1000 else (c, b, a)
        if y < 100: y += 2000
        if y >= 2400: y -= 543
        try: return pd.Timestamp(y, m, d)
        except: pass
    return None

class ScrapingError(Exception): pass

@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # ขยาย class ในการค้นหาให้กว้างขึ้น
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content", re.I)) or soup
        rows = []
        
        # 1. พยายามดึงแบบตารางก่อน (Table parsing)
        for row in content.find_all("tr"):
            text = " ".join([c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])])
            date = normalize_date(text)
            if not date: continue
            six = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
            three = re.findall(r"(?<!\d)\d{3}(?!\d)", text)
            two = re.findall(r"(?<!\d)\d{2}(?!\d)", text)
            
            if six and two: rows.append({"Date": date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
            elif three and two: rows.append({"Date": date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})
            
        # 2. ถ้าดึงแบบตารางไม่เจอ ให้ดึงจากข้อความธรรมดา (Text Fallback)
        if not rows:
            text = content.get_text(separator="\n", strip=True)
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            current_date = None
            
            for line in lines:
                date = normalize_date(line)
                if date is not None: current_date = date
                if current_date is None: continue
                
                six = re.findall(r"(?<!\d)\d{6}(?!\d)", line)
                three = re.findall(r"(?<!\d)\d{3}(?!\d)", line)
                two = re.findall(r"(?<!\d)\d{2}(?!\d)", line)
                
                if six and two: rows.append({"Date": current_date, "Result_6D": six[0], "Result_3D": six[0][-3:], "Result_2D": two[-1]})
                elif three and two: rows.append({"Date": current_date, "Result_6D": None, "Result_3D": three[0], "Result_2D": two[-1]})

        if not rows:
            raise ScrapingError("ไม่พบข้อมูลหวยในรูปแบบที่รองรับ")

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Result_3D"] = df["Result_3D"].astype(str).str.extract(r"(\d{3})")[0].str.zfill(3)
        df["Result_2D"] = df["Result_2D"].astype(str).str.extract(r"(\d{2})")[0].str.zfill(2)
        if "Result_6D" in df.columns:
            df["Result_6D"] = df["Result_6D"].astype(str).str.extract(r"(\d{6})")[0]
            
        return df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception as exc:
        raise ScrapingError(f"โหลดข้อมูลไม่สำเร็จ: {exc}")

def is_thai_6d(df): return "Result_6D" in df.columns and df["Result_6D"].notna().sum() >= 10

# ============================================================
# 6. ENHANCED FEATURE ENGINEERING
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
    w["DAY_OF_YEAR"] = dt.dayofyear.astype(np.int16)
    
    # NEW Calendar Features
    w["IS_WEEKEND"] = (w["DOW"] >= 5).astype(np.float32)
    w["IS_MONTH_START"] = (w["DAY"] <= 5).astype(np.float32)
    w["IS_MONTH_END"] = (w["DAY"] >= 25).astype(np.float32)

    w["DOW_SIN"] = np.sin(2 * np.pi * w["DOW"] / 7).astype(np.float32)
    w["DOW_COS"] = np.cos(2 * np.pi * w["DOW"] / 7).astype(np.float32)
    w["MONTH_SIN"] = np.sin(2 * np.pi * w["MONTH"] / 12).astype(np.float32)
    w["MONTH_COS"] = np.cos(2 * np.pi * w["MONTH"] / 12).astype(np.float32)

    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS

    for pos in positions:
        s = w[pos]
        p = s.shift(1)

        for lag in (1, 2, 3, 5): w[f"{pos}_L{lag}"] = s.shift(lag)

        for window in (10, 20):
            r = p.rolling(window, min_periods=2)
            w[f"{pos}_M{window}"] = r.mean()
            w[f"{pos}_S{window}"] = r.std()
            for digit in (0, 5):
                w[f"{pos}_F{window}_{digit}"] = (p == digit).astype(np.float32).rolling(window, min_periods=2).mean()

        w[f"{pos}_D1"] = s.shift(1) - s.shift(2)
        w[f"{pos}_D2"] = s.shift(2) - s.shift(3)
        w[f"{pos}_ODD"] = p % 2
        w[f"{pos}_HIGH"] = (p >= 5).astype(np.float32)
        w[f"{pos}_MOD3"] = p % 3

        w[f"{pos}_SIN"] = np.sin(2 * np.pi * p / 10).astype(np.float32)
        w[f"{pos}_COS"] = np.cos(2 * np.pi * p / 10).astype(np.float32)
        w[f"{pos}_EWMA7"] = p.ewm(span=7, adjust=False).mean()
        
        # NEW Momentum MACD Style
        ema5 = p.ewm(span=5, adjust=False).mean()
        ema15 = p.ewm(span=15, adjust=False).mean()
        w[f"{pos}_MACD"] = ema5 - ema15

        w[f"{pos}_REPEAT"] = (p == s.shift(2)).astype(np.float32)

    base = w[["H1","H2","H3","H4","H5","H6"]].shift(1) if thai_6d else w[["H","T","O"]].shift(1)
    w["PREV_SUM"] = base.sum(axis=1)
    w["PREV_RANGE"] = base.max(axis=1) - base.min(axis=1)
    w["PREV_ODD"] = (base % 2).sum(axis=1)
    w["PREV_HIGH"] = (base >= 5).sum(axis=1)

    return w.replace([np.inf, -np.inf], np.nan)

def get_features(thai_6d):
    base = [
        "DOW", "DAY", "MONTH", "DAY_OF_YEAR", 
        "IS_WEEKEND", "IS_MONTH_START", "IS_MONTH_END",
        "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS",
        "PREV_SUM", "PREV_RANGE", "PREV_ODD", "PREV_HIGH"
    ]
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS
    for pos in positions:
        base.extend([
            f"{pos}_L1", f"{pos}_L2", f"{pos}_L3", f"{pos}_L5",
            f"{pos}_M10", f"{pos}_M20", f"{pos}_S10", f"{pos}_S20",
            f"{pos}_D1", f"{pos}_D2", f"{pos}_ODD", f"{pos}_HIGH",
            f"{pos}_MOD3", f"{pos}_SIN", f"{pos}_COS",
            f"{pos}_EWMA7", f"{pos}_MACD", f"{pos}_REPEAT"
        ])
        for w in (10, 20):
            for d in (0, 5): base.append(f"{pos}_F{w}_{d}")
    return list(dict.fromkeys(base))

# ============================================================
# 7. ADAPTIVE CONFIG (TUNED FOR ACCURACY)
# ============================================================
def get_adaptive_config(n):
    if n >= 700:
        return {"min_train": 120, "trees": 70, "depth": 8, "leaf": 2, "hot_features": 22, "dead_features": 18, "backtest_points": 8, "recent_decay": 0.985}
    if n >= 400:
        return {"min_train": 100, "trees": 60, "depth": 7, "leaf": 2, "hot_features": 20, "dead_features": 16, "backtest_points": 7, "recent_decay": 0.980}
    if n >= 200:
        return {"min_train": 80, "trees": 50, "depth": 6, "leaf": 2, "hot_features": 18, "dead_features": 15, "backtest_points": 6, "recent_decay": 0.975}
    return {"min_train": 50, "trees": 40, "depth": 5, "leaf": 2, "hot_features": 16, "dead_features": 14, "backtest_points": 5, "recent_decay": 0.970}

# ============================================================
# 8. MODELS
# ============================================================
def create_model(name, cfg, system="hot"):
    t, d, l = cfg["trees"], cfg["depth"], cfg["leaf"]
    if system == "hot":
        if name == "ExtraTrees":
            return ExtraTreesClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, max_features=0.75, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
        return HistGradientBoostingClassifier(max_iter=int(t*0.7), max_leaf_nodes=15, learning_rate=0.04, min_samples_leaf=l, l2_regularization=2.0, random_state=42)
    
    # Dead Model: focus on lower variance to catch solid negatives
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=int(t*0.9), max_depth=max(4, d-1), min_samples_leaf=max(2, l), max_features=0.4, class_weight="balanced_subsample", n_jobs=-1, random_state=91)
    return HistGradientBoostingClassifier(max_iter=int(t*0.5), max_leaf_nodes=9, learning_rate=0.035, min_samples_leaf=max(2, l), l2_regularization=4.0, random_state=91)

def select_features_once(X, y, max_features, system="hot"):
    cols = list(X.columns)
    if len(cols) <= max_features: return cols
    valid = [c for c in cols if X[c].nunique(dropna=False) > 1]
    if len(valid) <= max_features: return valid
    
    Xi = X[valid].replace([np.inf, -np.inf], np.nan).astype(np.float32).fillna(0.0)
    seed, trees, depth = (123, 15, 6) if system == "hot" else (321, 10, 5)
    
    selector = ExtraTreesClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=seed)
    selector.fit(Xi, y)
    order = np.argsort(selector.feature_importances_)[::-1]
    return [valid[i] for i in order[:max_features]]

def normalize_probability(p):
    p = np.clip(np.nan_to_num(np.asarray(p, dtype=np.float32), nan=0.0), 1e-9, None)
    return p / p.sum() if p.sum() > 0 else np.ones(10, dtype=np.float32)/10

def model_probability(X_train, y_train, X_test, cfg, selected, system):
    A = X_train[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    B = X_test[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    med = A.median()
    A, B = A.fillna(med).fillna(0.0), B.fillna(med).fillna(0.0)
    
    predictions = []
    age = np.arange(len(A) - 1, -1, -1).astype(np.float32)
    weights = cfg["recent_decay"] ** age
    weights = (weights / (np.mean(weights) + 1e-9)).astype(np.float32)

    for name in MODEL_NAMES:
        try:
            model = create_model(name, cfg, system)
            try: model.fit(A, y_train, sample_weight=weights)
            except: model.fit(A, y_train)
            
            raw = model.predict_proba(B)[0]
            out = np.zeros(10, dtype=np.float32)
            for cls, prob in zip(model.classes_, raw):
                if 0 <= int(cls) <= 9: out[int(cls)] = prob
            predictions.append(normalize_probability(out))
        except: continue
        
    if not predictions: return np.ones(10, dtype=np.float32)/10
    
    if len(predictions) == 2:
        ensemble = (predictions[0]*0.6 + predictions[1]*0.4) if system == "hot" else (predictions[0]*0.55 + predictions[1]*0.45)
    else:
        ensemble = np.mean(predictions, axis=0)
    return normalize_probability(ensemble)

def hot_system(X_train, y_train, X_test, cfg):
    selected = select_features_once(X_train, y_train, cfg["hot_features"], system="hot")
    probability = model_probability(X_train, y_train, X_test, cfg, selected, system="hot")
    hot_prob = normalize_probability(np.power(probability, 1.1)) # Sharpen slightly
    order = np.argsort(hot_prob)[::-1]
    return {
        "probability": hot_prob,
        "hot": [(int(n), float(hot_prob[n])) for n in order[:3]],
        "confidence": float(hot_prob[order[0]] - hot_prob[order[1]]) if len(order)>=2 else 0.0,
        "top3": float(hot_prob[order[:3]].sum()),
        "selected_features": selected
    }

def dead_system(X_train, y_train, X_test, cfg):
    selected = select_features_once(X_train, y_train, cfg["dead_features"], system="dead")
    probability = model_probability(X_train, y_train, X_test, cfg, selected, system="dead")
    dead_score = normalize_probability(1.0 - probability)
    order = np.argsort(dead_score)[::-1]
    return {
        "dead_score": dead_score,
        "dead": [(int(n), float(dead_score[n])) for n in order[:7]],
        "top7": float(dead_score[order[:7]].sum()),
        "selected_features": selected
    }

# Backtest Engine simplified for brevity
def walk_forward_system(df_feat, pos, features, cfg, system="hot"):
    X, y = df_feat[features].astype(np.float32), df_feat[pos].astype(np.int8)
    start = cfg["min_train"]
    if len(df_feat) <= start + 2: return {"tests": 0, "scores": {}}
    
    tests = min(cfg["backtest_points"], len(df_feat) - start)
    test_indices = np.unique(np.linspace(start, len(df_feat)-1, tests, dtype=int))
    selected = select_features_once(X.iloc[:test_indices[0]], y.iloc[:test_indices[0]], cfg[f"{system}_features"], system=system)
    
    records = []
    for idx in test_indices:
        if y.iloc[:idx].nunique() < 2: continue
        try: probs = model_probability(X.iloc[:idx], y.iloc[:idx], X.iloc[[idx]], cfg, selected, system)
        except: continue
        
        actual = int(y.iloc[idx])
        if system == "hot":
            order = np.argsort(probs)[::-1]
            records.append({"top1": int(actual == order[0]), "top3": int(actual in order[:3]), "top5": int(actual in order[:5])})
        else:
            dead_order = np.argsort(1.0 - probs)[::-1]
            records.append({"dead5": int(actual in dead_order[:5]), "dead7": int(actual in dead_order[:7])})
            
    if not records: return {"tests": 0, "scores": {}}
    h = pd.DataFrame(records)
    decay = cfg["recent_decay"] ** np.arange(len(h)-1, -1, -1)
    decay /= decay.sum()
    
    scores = {k: float(np.sum(h[k]*decay)) for k in h.columns}
    return {"tests": len(h), "scores": scores}

def final_prediction(df_feat, pos, features, cfg):
    X, y = df_feat[features].astype(np.float32), df_feat[pos].astype(np.int8)
    X_train, y_train, X_test = X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]]
    return {"hot": hot_system(X_train, y_train, X_test, cfg), "dead": dead_system(X_train, y_train, X_test, cfg)}

# ============================================================
# 9. DISPLAY COMPONENTS (NEW UI)
# ============================================================
def display_hot_card(pos, result):
    data = result["hot"]["hot"]
    nums = " - ".join(str(n) for n, _ in data)
    probs_html = "".join(f'<span class="prob-pill">{n}: {p*100:.1f}%</span>' for n, p in data)
    
    html = f"""
    <div class="hot-card">
        <div class="hot-number">{nums}</div>
        <div class="prob-text">🔥 โอกาสมา (TOP 3)<br>{probs_html}</div>
        <div class="confidence">📌 Win Gap: {result["hot"]["confidence"]*100:.1f}%</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_dead_card(pos, result):
    data = result["dead"]["dead"]
    nums = " - ".join(str(n) for n, _ in data[:5]) # Show only top 5 in large font to save space
    probs_html = "".join(f'<span class="prob-pill">{n}</span>' for n, p in data)
    
    html = f"""
    <div class="dead-card">
        <div class="dead-number">{nums}</div>
        <div class="prob-text">🛑 โอกาสดับ (TOP 7)<br>{probs_html}</div>
        <div class="confidence">🎯 Dead Confidence: {result["dead"]["top7"]*100:.1f}%</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# 10. MAIN APP
# ============================================================
def main():
    inject_css()
    st.markdown('<div class="main-title">🤖 LOTTO AI PRO V9.0 TURBO</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🚀 High Accuracy & Momentum Features | 🔥 HOT TOP-3 & 🛑 DEAD TOP-7</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    lottery = c1.selectbox("🏷️ เลือกประเภทหวย", list(LOTTERY_SOURCES.keys()))
    selected_day = c2.selectbox("📅 วันเป้าหมาย", ["อัตโนมัติ"] + DOW_NAMES)

    if not st.button("⚡ เริ่มวิเคราะห์ระบบ V9.0", type="primary", use_container_width=True): return

    with st.spinner("📥 กำลังดึงข้อมูลสถิติล่าสุด..."):
        try: df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        except Exception as exc: return st.error(str(exc))

    if len(df) < 50: return st.error(f"❌ มีข้อมูล {len(df)} งวด (ต้องการอย่างน้อย 50 งวดเพื่อความแม่นยำ)")

    thai_6d = lottery == "หวยไทย" and is_thai_6d(df)
    positions = THAI_POSITIONS if thai_6d else NORMAL_POSITIONS

    last_date = pd.Timestamp(df["Date"].iloc[-1])
    days_ahead = 7 if selected_day == "อัตโนมัติ" else max(1, (DOW_NAMES.index(selected_day) - last_date.dayofweek) % 7 or 7)
    target_date = last_date + timedelta(days=days_ahead)

    dummy = {"Date": target_date, "Result_3D": "000", "Result_2D": "00"}
    if thai_6d: dummy["Result_6D"] = "000000"
    ext = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)

    with st.spinner("⚡ กำลังประมวลผลฟีเจอร์ขั้นสูง (MACD, Calendar Dynamics)..."):
        feat = build_features(ext, thai_6d)
        features = get_features(thai_6d)
        cfg = get_adaptive_config(len(df))

    # processing
    final, hot_backtest, dead_backtest = {}, {}, {}
    progress = st.progress(0)
    status_text = st.empty()

    for i, pos in enumerate(positions):
        status_text.caption(f"🧠 กำลังวิเคราะห์หลัก: {POSITION_LABELS[pos]}")
        hot_backtest[pos] = walk_forward_system(feat.iloc[:-1], pos, features, cfg, "hot")
        dead_backtest[pos] = walk_forward_system(feat.iloc[:-1], pos, features, cfg, "dead")
        final[pos] = final_prediction(feat, pos, features, cfg)
        progress.progress(int(((i + 1) / len(positions)) * 100))
        
    progress.empty()
    status_text.empty()

    st.markdown(f"""
        <div class="status-card">
        ✅ วิเคราะห์สำเร็จ: {len(df):,} งวด | เป้าหมาย: {target_date.strftime("%d/%m/%Y")} ({lottery})<br>
        ⚙️ โมเดลใช้ <b>{cfg['trees']} Trees</b> | ฟีเจอร์สูงสุด <b>{len(features)} ตัว</b>
        </div><br>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # HIGHLIGHT SUMMARY
    # ---------------------------------------------------------
    st.markdown("### 🏆 สรุปเลขฟันธง V9.0 (AI Recommend)")
    summary_data = []
    for pos in positions:
        hot_top = final[pos]["hot"]["hot"][0]
        summary_data.append({
            "ตำแหน่ง": POSITION_LABELS[pos],
            "🔥 เลขเด่นสุด (ฟันธง)": f"{hot_top[0]}",
            "โอกาสเข้า (ความมั่นใจ)": f"{hot_top[1]*100:.1f}%",
            "📌 เลขเด่นรอง (กันพลาด)": f"{final[pos]['hot']['hot'][1][0]}, {final[pos]['hot']['hot'][2][0]}",
            "🛑 เลขดับ (ตัดทิ้ง 3 อันดับ)": f"{final[pos]['dead']['dead'][0][0]}, {final[pos]['dead']['dead'][1][0]}, {final[pos]['dead']['dead'][2][0]}"
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ---------------------------------------------------------
    # DETAILED TABS
    # ---------------------------------------------------------
    t1, t2 = st.tabs(["🎯 เจาะลึกรายหลัก (Hot & Dead)", "📊 สถิติความแม่นยำย้อนหลัง"])

    with t1:
        st.markdown("ระบบจัดเรียงเลขเด่นและเลขดับแบบ **เทียบกันชัดๆ ในแต่ละหลัก**")
        for pos in positions:
            st.markdown(f'<div class="position-title">{POSITION_LABELS[pos]}</div>', unsafe_allow_html=True)
            col_hot, col_dead = st.columns(2)
            with col_hot:
                display_hot_card(pos, final[pos])
            with col_dead:
                display_dead_card(pos, final[pos])

    with t2:
        st.markdown("### สถิติ Backtest เชิงลึก (Walk-Forward)")
        for pos in positions:
            h_sc = hot_backtest[pos].get("scores", {})
            d_sc = dead_backtest[pos].get("scores", {})
            if not h_sc: continue
            
            with st.expander(f"📊 สถิติของ {POSITION_LABELS[pos]}"):
                st.dataframe(pd.DataFrame([
                    {"Metric": "🔥 ทายถูกอันดับ 1 (Top-1)", "ความแม่นยำ AI": f"{h_sc.get('top1',0)*100:.1f}%"},
                    {"Metric": "🔥 ทายถูกในกลุ่ม (Top-3)", "ความแม่นยำ AI": f"{h_sc.get('top3',0)*100:.1f}%"},
                    {"Metric": "🛑 ตัดเลขดับรอด (Avoid Dead 7)", "ความแม่นยำ AI": f"{d_sc.get('not_dead7',0)*100:.1f}%"}
                ]), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
