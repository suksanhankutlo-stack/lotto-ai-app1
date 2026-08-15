# ============================================================
# 🤖 LOTTO AI PRO V8.1 MAX (REFACTORED: TUNED & FAST)
# STRICT WALK-FORWARD • NO PERSISTENT MEMORY • LEAKAGE SAFE
# ============================================================
import re, hashlib, warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT & CONSTANTS
# ============================================================
st.set_page_config(page_title="Lotto AI V8.1 MAX", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

LOTTERY_SOURCES = {
    "หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html", "หวยธกส": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html", "หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html", "หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html", "หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html", "หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}
POSITIONS = ["H", "T", "O", "T2", "O2"]
POSITION_LABELS = {"H": "💯 หลักร้อย 3 ตัวบน", "T": "🔟 หลักสิบ 3 ตัวบน", "O": "1️⃣ หลักหน่วย 3 ตัวบน", "T2": "🔽 หลักสิบ 2 ตัวล่าง", "O2": "⬇️ หลักหน่วย 2 ตัวล่าง"}
DOW_NAMES, MODEL_NAMES = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"], ["ExtraTrees", "RandomForest", "HistGradientBoosting"]

def inject_css():
    st.markdown("""
        <style>
        .stApp { background: #f8fafc; }
        .main-title { text-align: center; font-size: 2.35rem; font-weight: 900; margin-bottom: 2px; }
        .subtitle { text-align: center; color: #64748b; font-size: .95rem; margin-bottom: 18px; }
        .status-card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 14px; padding: 14px; text-align: center; color: #1e40af; font-weight: 700; line-height: 1.75; }
        .hot-card { background: #f0fdf4; border-left: 7px solid #16a34a; border-radius: 14px; padding: 15px; margin: 10px 0; }
        .dead-card { background: #fef2f2; border-left: 7px solid #dc2626; border-radius: 14px; padding: 15px; margin: 10px 0; }
        .position-title { font-size: 1.15rem; font-weight: 900; color: #334155; margin-bottom: 6px; }
        .hot-number, .dead-number { font-size: 2.15rem; font-weight: 900; letter-spacing: 3px; text-align: center; }
        .hot-number { color: #16a34a; } .dead-number { color: #dc2626; }
        .prob-text { text-align: center; color: #64748b; font-size: .82rem; margin-top: 4px; }
        .model-badge { text-align: center; background: white; border-radius: 9px; padding: 7px; margin-top: 7px; color: #475569; font-weight: 700; }
        .confidence { text-align: center; font-size: .9rem; font-weight: 800; margin-top: 5px; color: #334155; }
        div.stButton > button { width: 100%; min-height: 48px; border-radius: 10px; font-size: 16px; font-weight: 800; }
        </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. DATE & SCRAPING
# ============================================================
THAI_MONTHS = {"มกราคม":1, "กุมภาพันธ์":2, "มีนาคม":3, "เมษายน":4, "พฤษภาคม":5, "มิถุนายน":6, "กรกฎาคม":7, "สิงหาคม":8, "กันยายน":9, "ตุลาคม":10, "พฤศจิกายน":11, "ธันวาคม":12,
               "ม.ค.":1, "ก.พ.":2, "มี.ค.":3, "เม.ย.":4, "พ.ค.":5, "มิ.ย.":6, "ก.ค.":7, "ส.ค.":8, "ก.ย.":9, "ต.ค.":10, "พ.ย.":11, "ธ.ค.":12}

def normalize_date(value):
    if not value: return None
    text = str(value).strip()
    for m_name, m_num in THAI_MONTHS.items():
        if match := re.search(rf"(\d{{1,2}})\s*{re.escape(m_name)}\s*(\d{{4}})", text):
            y = int(match.group(2))
            y = y - 543 if y >= 2400 else y
            try: return pd.Timestamp(y, m_num, int(match.group(1)))
            except: return None
    if match := re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", text):
        a, b, c = map(int, match.groups())
        y, m, d = (a, b, c) if a >= 1000 else (c, b, a)
        y = y + 2000 if y < 100 else (y - 543 if y >= 2400 else y)
        try: return pd.Timestamp(y, m, d)
        except: pass
    return None

class ScrapingError(Exception): pass
class NetworkScrapingError(ScrapingError): pass
class HTTPStatusScrapingError(ScrapingError): pass
class ParsingScrapingError(ScrapingError): pass

@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):
    headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36", "Accept-Language": "th-TH,th;q=0.9,en;q=0.8"}
    try:
        try: response = requests.get(url, headers=headers, timeout=15)
        except requests.exceptions.RequestException as exc: raise NetworkScrapingError(f"Network error: {exc}") from exc
        if response.status_code != 200: raise HTTPStatusScrapingError(f"HTTP {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content", re.I)) or soup
        extracted, current_date = [], None

        for table in content.find_all("table"):
            for row in table.find_all("tr"):
                cells = " ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th"]))
                if not cells: continue
                if parsed_date := normalize_date(cells): current_date = parsed_date
                n3, n2 = re.findall(r"\b\d{3}\b", cells), re.findall(r"\b\d{2}\b", cells)
                if current_date and n3 and n2: extracted.append({"Date": current_date, "Result_3D": n3[0], "Result_2D": n2[-1]})
        
        if not extracted:
            for line in filter(None, (l.strip() for l in content.get_text(separator="\n").split("\n"))):
                if parsed_date := normalize_date(line): current_date = parsed_date
                if match := re.search(r"\b(\d{3})\b.*?\b(\d{2})\b", line):
                    if current_date: extracted.append({"Date": current_date, "Result_3D": match.group(1), "Result_2D": match.group(2)})

        if not extracted: raise ParsingScrapingError("ไม่พบรูปแบบข้อมูลหวยที่ระบบรู้จัก")
        df = pd.DataFrame(extracted)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Result_3D"] = df["Result_3D"].astype(str).str.extract(r"(\d{3})")[0].str.zfill(3)
        df["Result_2D"] = df["Result_2D"].astype(str).str.extract(r"(\d{2})")[0].str.zfill(2)
        df = df.dropna().query("Result_3D.str.match('^\\d{3}$') and Result_2D.str.match('^\\d{2}$')").drop_duplicates().sort_values("Date").reset_index(drop=True)
        if df.empty: raise ParsingScrapingError("ข้อมูลถูกกรองออกทั้งหมด")
        return df
    except ScrapingError: raise
    except Exception as exc: raise ParsingScrapingError(f"Parsing error: {exc}") from exc

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def gap_since_digit(series, digit):
    arr, output, last_seen = series.shift(1).to_numpy(), np.zeros(len(series), dtype=float), -1
    for i, v in enumerate(arr):
        if pd.notna(v) and int(v) == digit: last_seen = i
        output[i] = i + 1 if last_seen < 0 else i - last_seen
    return pd.Series(output, index=series.index)

def build_features(df):
    work = df.copy()
    work["H"], work["T"], work["O"] = work["Result_3D"].str[0].astype(int), work["Result_3D"].str[1].astype(int), work["Result_3D"].str[2].astype(int)
    work["T2"], work["O2"] = work["Result_2D"].str[0].astype(int), work["Result_2D"].str[1].astype(int)
    
    dt = work["Date"].dt
    work["DOW"], work["DAY"], work["MONTH"], work["DAY_OF_YEAR"], work["WEEK_OF_YEAR"] = dt.dayofweek, dt.day, dt.month, dt.dayofyear, dt.isocalendar().week.astype(int)
    work["DOW_SIN"], work["DOW_COS"] = np.sin(2 * np.pi * work["DOW"] / 7), np.cos(2 * np.pi * work["DOW"] / 7)
    work["MONTH_SIN"], work["MONTH_COS"] = np.sin(2 * np.pi * work["MONTH"] / 12), np.cos(2 * np.pi * work["MONTH"] / 12)
    work["DAY_SIN"], work["DAY_COS"] = np.sin(2 * np.pi * work["DAY"] / 31), np.cos(2 * np.pi * work["DAY"] / 31)

    p3, p2 = work[["H", "T", "O"]].shift(1), work[["T2", "O2"]].shift(1)
    work["PREV_SUM3"], work["PREV_SUM2"] = p3.sum(axis=1), p2.sum(axis=1)
    work["PREV_RANGE3"], work["PREV_MEAN3"] = p3.max(axis=1) - p3.min(axis=1), p3.mean(axis=1)
    work["PREV_HIGH_COUNT"], work["PREV_ODD_COUNT"] = (p3 >= 5).sum(axis=1), (p3 % 2).sum(axis=1)
    work["PREV_UNIQUE3"] = p3.nunique(axis=1)
    work["PREV_REPEAT3"] = 3 - work["PREV_UNIQUE3"]

    for pos in POSITIONS:
        s, shifted = work[pos], work[pos].shift(1)
        for lag in range(1, 8): work[f"{pos}_L{lag}"] = s.shift(lag)
        for w in [3, 5, 10, 20]:
            roll = shifted.rolling(w, min_periods=2)
            work[f"{pos}_M{w}"], work[f"{pos}_S{w}"] = roll.mean(), roll.std()
            for d in [0, 2, 5, 7]: work[f"{pos}_F{w}_{d}"] = (shifted == d).astype(float).rolling(w, min_periods=2).mean()
        
        work[f"{pos}_D1"], work[f"{pos}_D2"], work[f"{pos}_D3"] = s.shift(1)-s.shift(2), s.shift(2)-s.shift(3), s.shift(3)-s.shift(4)
        work[f"{pos}_ODD"], work[f"{pos}_HIGH"] = shifted % 2, (shifted >= 5).astype(float)
        work[f"{pos}_MOD3"], work[f"{pos}_MOD5"], work[f"{pos}_MIRROR"] = shifted % 3, shifted % 5, 9 - shifted
        work[f"{pos}_SIN"], work[f"{pos}_COS"] = np.sin(2 * np.pi * shifted / 10), np.cos(2 * np.pi * shifted / 10)
        
        work[f"{pos}_EWMA3"] = shifted.ewm(span=3, adjust=False).mean()
        work[f"{pos}_EWMA7"] = shifted.ewm(span=7, adjust=False).mean()
        work[f"{pos}_IS_REPEAT"] = (shifted == s.shift(2)).astype(float)
        
        for d in [0, 2, 5, 7]: work[f"{pos}_GAP_{d}"] = gap_since_digit(s, d)
        
        # แก้ไขจุดที่อาจเกิด Walrus Operator Syntax Error ได้
        for w in [5, 10, 20]:
            counts = [(shifted == d).astype(float).rolling(w, min_periods=2).sum() for d in range(10)]
            total = shifted.rolling(w, min_periods=2).count().replace(0, np.nan)
            entropy = pd.Series(0.0, index=work.index, dtype=float)
            for c in counts:
                p = c / total
                entropy += np.where(p > 0, -p * np.log(p), 0.0)
            work[f"{pos}_ENT{w}"] = entropy

    for lag in [1, 2, 3]:
        c = work[["H", "T", "O"]].shift(lag)
        work[f"PREV_SUM3_L{lag}"], work[f"PREV_RANGE3_L{lag}"], work[f"PREV_ODD3_L{lag}"] = c.sum(axis=1), c.max(axis=1) - c.min(axis=1), (c % 2).sum(axis=1)
    return work.replace([np.inf, -np.inf], np.nan)

# Generate FEATURE list dynamically
FEATURES = ["DOW", "DAY", "MONTH", "DAY_OF_YEAR", "WEEK_OF_YEAR", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS", "DAY_SIN", "DAY_COS",
            "PREV_SUM3", "PREV_SUM2", "PREV_RANGE3", "PREV_MEAN3", "PREV_HIGH_COUNT", "PREV_ODD_COUNT", "PREV_UNIQUE3", "PREV_REPEAT3"]
for l in [1,2,3]: FEATURES.extend([f"PREV_SUM3_L{l}", f"PREV_RANGE3_L{l}", f"PREV_ODD3_L{l}"])
for pos in POSITIONS:
    FEATURES.extend([f"{pos}_L{l}" for l in range(1, 8)])
    for w in [3, 5, 10, 20]: FEATURES.extend([f"{pos}_M{w}", f"{pos}_S{w}"] + [f"{pos}_F{w}_{d}" for d in [0,2,5,7]])
    FEATURES.extend([f"{pos}_{x}" for x in ["D1","D2","D3","ODD","HIGH","MOD3","MOD5","SIN","COS","MIRROR","EWMA3","EWMA7","IS_REPEAT"]])
    FEATURES.extend([f"{pos}_GAP_{d}" for d in [0,2,5,7]] + [f"{pos}_ENT{w}" for w in [5,10,20]])
FEATURES = list(dict.fromkeys(FEATURES))

# ============================================================
# 4. CONFIG & MODELS (TUNED & SPEED)
# ============================================================
def get_adaptive_config(n):
    if n >= 700: return {"min_train": 120, "trees": 150, "depth": 10, "leaf": 2, "max_features": "sqrt", "selected_features": 35, "backtest_start": 120, "recent_decay": 0.985, "max_backtest": 120}
    if n >= 400: return {"min_train": 100, "trees": 120, "depth": 8, "leaf": 2, "max_features": "sqrt", "selected_features": 30, "backtest_start": 100, "recent_decay": 0.98, "max_backtest": 100}
    if n >= 200: return {"min_train": 80,  "trees": 100, "depth": 7, "leaf": 2, "max_features": "sqrt", "selected_features": 25, "backtest_start": 80,  "recent_decay": 0.975, "max_backtest": 80}
    return {"min_train": 50, "trees": 75, "depth": 6, "leaf": 2, "max_features": "sqrt", "selected_features": 20, "backtest_start": 50, "recent_decay": 0.97, "max_backtest": 60}

def create_model(name, cfg):
    t, d, l = cfg["trees"], cfg["depth"], cfg["leaf"]
    if name == "ExtraTrees": return ExtraTreesClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, max_features=cfg["max_features"], class_weight="balanced", n_jobs=-1, random_state=42)
    if name == "RandomForest": return RandomForestClassifier(n_estimators=t, max_depth=d, min_samples_leaf=l, max_features=cfg["max_features"], class_weight="balanced", n_jobs=-1, random_state=42)
    return HistGradientBoostingClassifier(max_iter=max(50, int(t*0.85)), max_leaf_nodes=31, learning_rate=0.03, min_samples_leaf=l, l2_regularization=2.5, random_state=42)

def select_features_training_only(X, y, f_names, max_f):
    if len(f_names) <= max_f: return list(f_names)
    valid = [col for col in f_names if X[col].nunique(dropna=False) > 1]
    if not valid: return list(f_names[:max_f])
    try:
        X_imp = SimpleImputer(strategy="median").fit_transform(X[valid])
        sel = ExtraTreesClassifier(n_estimators=50, max_depth=6, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=123).fit(X_imp, y)
        chosen = [valid[i] for i in np.argsort(sel.feature_importances_)[::-1][:max_f]]
        if chosen: return chosen
    except: pass
    return list(valid[:max_f])

def stabilize_probability(probs, temperature=0.85):
    logits = np.log(np.clip(np.asarray(probs, dtype=float), 1e-9, 1)) / temperature
    exp_logits = np.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()

# ============================================================
# 5. CORE ENGINE (TRAIN & BACKTEST)
# ============================================================
def train_and_predict(X_train, y_train, X_test, config, cached_features=None):
    sel_f = cached_features or select_features_training_only(X_train, y_train, list(X_train.columns), config["selected_features"])
    imp = SimpleImputer(strategy="median")
    X_train_imp, X_test_imp = imp.fit_transform(X_train[sel_f]), imp.transform(X_test[sel_f])
    model_probs = {}
    for m in MODEL_NAMES:
        try:
            model = create_model(m, config).fit(X_train_imp, y_train)
            raw = model.predict_proba(X_test_imp)[0]
            out = np.zeros(10, dtype=float)
            for c, p in zip(model.classes_, raw):
                if 0 <= int(c) <= 9: out[int(c)] = float(p)
            model_probs[m] = stabilize_probability(out / (out.sum() or 1), temperature=0.85)
        except: continue
    if not model_probs: return None
    ensemble = stabilize_probability(np.mean(list(model_probs.values()), axis=0), temperature=0.85)
    return ensemble, model_probs, sel_f

def strict_walk_forward_backtest(df_feat, pos, config):
    X, y, n = df_feat[FEATURES].astype(float), df_feat[pos].astype(int), len(df_feat)
    start = max(config["min_train"], config["backtest_start"])
    start = n - config.get("max_backtest", 100) if n - start > config.get("max_backtest", 100) else start
    history, cached_features = [], None

    for t_idx in range(start, n):
        if t_idx % 5 == 0: cached_features = None # Cache refresh
        if y.iloc[:t_idx].nunique() < 2: continue
        res = train_and_predict(X.iloc[:t_idx], y.iloc[:t_idx], X.iloc[[t_idx]], config, cached_features)
        if not res: continue
        ens, _, cached_features = res
        act = int(y.iloc[t_idx])
        history.append({"index": t_idx, "actual": act, "top1": int(act in np.argsort(ens)[::-1][:1]),
                        "top3": int(act in np.argsort(ens)[::-1][:3]), "top5": int(act in np.argsort(ens)[::-1][:5]),
                        "dead7": int(act in np.argsort(ens)[:7]), "logloss": -np.log(max(ens[act], 1e-9)), "brier": np.sum((ens - np.eye(10)[act])**2)})

    if not history: return {"scores": {}, "tests": 0}
    hist = pd.DataFrame(history)
    w = config["recent_decay"] ** (len(hist) - np.arange(len(hist)) - 1); w /= w.sum()
    metrics = {k: float(np.sum(hist[k] * w)) for k in ["top1", "top3", "top5", "dead7", "logloss", "brier"]}
    r_top3 = hist["top3"].rolling(min(30, max(5, len(hist))), min_periods=5).mean()
    metrics["stability"] = float(np.clip(1.0 - r_top3.std(), 0, 1)) if len(r_top3.dropna()) else 0.0
    metrics["score"] = 0.25*metrics["top1"] + 0.25*metrics["top3"] + 0.15*metrics["top5"] + 0.10*metrics["stability"] + 0.10*(1/(1+metrics["logloss"])) + 0.10*(1/(1+metrics["brier"])) + 0.05*(1-metrics["dead7"])
    for k in ["top1", "top3", "top5", "dead7"]: metrics[f"raw_{k}"] = float(hist[k].mean())
    return {"scores": metrics, "tests": len(hist), "history": history}

def final_prediction(df_feat, pos, config):
    X, y = df_feat[FEATURES].astype(float), df_feat[pos].astype(int)
    res = train_and_predict(X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]], config)
    ens, m_probs, sel_f = res if res else (np.ones(10)/10, {}, [])
    sorted_idx = np.argsort(ens)
    
    # แก้ไขจุดที่เกิด Error Comprehension Iterable
    ranks = [np.argsort(p)[::-1][:5] for p in m_probs.values()]
    agreements = [len(set(a).intersection(b))/5 for i, a in enumerate(ranks) for b in ranks[i+1:]] if len(ranks) >= 2 else []
    
    sel_model = max(m_probs.keys(), key=lambda k: np.sort(m_probs[k])[::-1][:3].sum()) if m_probs else "Ensemble"
    return {"model": sel_model, "weights": {k: 1/max(len(m_probs),1) for k in m_probs}, "model_probabilities": m_probs,
            "probabilities": ens, "hot": [(int(i), float(ens[i])) for i in sorted_idx[::-1][:5]], "dead": [(int(i), float(ens[i])) for i in sorted_idx[:7]],
            "confidence": float(np.sort(ens)[-1] - np.sort(ens)[-2]), "top3_concentration": float(np.sort(ens)[::-1][:3].sum()),
            "agreement": float(np.mean(agreements)) if agreements else 0.0, "selected_features": sel_f}

# ============================================================
# 6. UI HELPERS
# ============================================================
def display_card(pos, res, is_hot=True):
    data, style = (res["hot"], "hot") if is_hot else (res["dead"], "dead")
    nums = " - ".join(str(n) for n, _ in data)
    probs = " | ".join(f"{n}: {p*100:.1f}%" for n, p in data)
    html = f'<div class="{style}-card"><div class="position-title">{POSITION_LABELS[pos]}</div><div class="{style}-number">{nums}</div>'
    html += f'<div class="prob-text">AI Probability{" ต่ำสุด" if not is_hot else ""}: {probs}</div>'
    if is_hot: html += f'<div class="confidence">📌 Top-1 Gap: {res["confidence"]*100:.1f}% &nbsp;|&nbsp; Top-3: {res["top3_concentration"]*100:.1f}% &nbsp;|&nbsp; Agreement: {res["agreement"]*100:.1f}%</div>'
    html += f'<div class="model-badge">🤖 {"Final AI: "+res["model"] if is_hot else "Adaptive Walk-Forward AI"}</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# 7. MAIN APP
# ============================================================
def main():
    inject_css()
    st.markdown('<div class="main-title">🤖 LOTTO AI PRO V8.1 MAX (REFACTORED)</div><div class="subtitle">STRICT WALK-FORWARD • EWMA MOMENTUM • SHARP PROBABILITIES • ⚡ SPEED OPTIMIZED</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    lottery = c1.selectbox("🏷️ เลือกประเภทหวย", list(LOTTERY_SOURCES.keys()))
    selected_day = c2.selectbox("📅 วันเป้าหมาย", ["อัตโนมัติ"] + DOW_NAMES)
    if not st.button("🚀 เริ่มวิเคราะห์ PRO V8.1 MAX", type="primary", use_container_width=True):
        return st.info("เลือกหวยและวันเป้าหมาย แล้วกด 🚀 เริ่มวิเคราะห์")

    with st.spinner("📥 กำลังโหลดข้อมูลย้อนหลัง..."):
        try: df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        except Exception as exc: return st.error(f"Error: {exc}")
    if len(df) < 50: return st.error(f"❌ พบข้อมูลเพียง {len(df)} งวด (ต้องการ ≥50)")

    tgt_dow = None if selected_day == "อัตโนมัติ" else DOW_NAMES.index(selected_day.replace("วัน",""))
    last_d = pd.Timestamp(df["Date"].iloc[-1])
    days_ahead = (tgt_dow - last_d.dayofweek) % 7 if tgt_dow is not None else max((last_d - pd.Timestamp(df["Date"].iloc[-2])).days if len(df)>=2 else 7, 7)
    target_date = last_d + timedelta(days=days_ahead or 7)

    with st.spinner("🧠 สร้าง Strict Causal Features + EWMA Momentum..."):
        ext_df = pd.concat([df, pd.DataFrame([{"Date": target_date, "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        feat_df, config = build_features(ext_df), get_adaptive_config(len(df))

    st.info(f"⚡ V8.1 REFACTORED • ข้อมูล {len(df):,} งวด • Min Train {config['min_train']} • Feature Pool {len(FEATURES)} • Selected ≤{config['selected_features']} • Trees {config['trees']}")
    
    backtest_res, final_res, prog, stat = {}, {}, st.progress(0), st.empty()
    for i, p in enumerate(POSITIONS):
        stat.caption(f"🧠 Strict Walk-Forward: {POSITION_LABELS[p]}")
        backtest_res[p] = strict_walk_forward_backtest(feat_df.iloc[:-1], p, config)
        prog.progress(int((i+1)/5*100))
    prog.empty(); stat.empty()

    with st.spinner("🤖 Train Final AI จากข้อมูลย้อนหลังเท่านั้น..."):
        for p in POSITIONS: final_res[p] = final_prediction(feat_df, p, config)

    bt_tests = [b["tests"] for b in backtest_res.values()]
    st.markdown(f'<div class="status-card">🤖 <b>LOTTO AI V8.1 MAX</b><br>📊 ข้อมูล: {len(df):,} งวด | 📅 เป้าหมาย: {target_date.strftime("%d/%m/%Y")}<br>🎯 Selected Features: ≤{config["selected_features"]} | 🌳 Trees: {config["trees"]}<br>🧪 Backtest: {min(bt_tests)}–{max(bt_tests)} งวด</div><br>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🎯 เลขเด่น AI", "🛑 เลขดับ 7", "📊 Walk-Forward Accuracy"])
    with t1:
        for p in POSITIONS: display_card(p, final_res[p], True)
    with t2:
        for p in POSITIONS: display_card(p, final_res[p], False)
    with t3:
        for p in POSITIONS:
            st.markdown(f"### {POSITION_LABELS[p]}")
            sc = backtest_res[p].get("scores", {})
            if not sc: st.warning("ไม่มีข้อมูล Backtest เพียงพอ"); continue
            st.dataframe(pd.DataFrame([
                {"Metric": "Top-1", "AI": f"{sc['top1']*100:.1f}%", "Random": "10.0%", "Edge": f"{(sc['top1']-0.10)*100:+.1f}%"},
                {"Metric": "Top-3", "AI": f"{sc['top3']*100:.1f}%", "Random": "30.0%", "Edge": f"{(sc['top3']-0.30)*100:+.1f}%"},
                {"Metric": "Top-5", "AI": f"{sc['top5']*100:.1f}%", "Random": "50.0%", "Edge": f"{(sc['top5']-0.50)*100:+.1f}%"},
                {"Metric": "Dead-7", "AI": f"{sc['dead7']*100:.1f}%", "Random": "70.0%", "Edge": f"{(sc['dead7']-0.70)*100:+.1f}%"},
                {"Metric": "LogLoss/Brier", "AI": f"L: {sc['logloss']:.3f} | B: {sc['brier']:.3f}", "Random": "-", "Edge": "-"}
            ]), use_container_width=True, hide_index=True)
            if backtest_res[p]["tests"] < 100: st.warning("⚠️ Backtest ต่ำกว่า 100 งวด มีความผันผวน")

    with st.expander("🎯 สรุปเลขเด่นทั้งหมด"):
        st.dataframe(pd.DataFrame([{"ตำแหน่ง": POSITION_LABELS[p], "Top-1": final_res[p]["hot"][0][0], "Prob": f'{final_res[p]["hot"][0][1]*100:.1f}%', "AI": final_res[p]["model"], "WF Top-1": f'{backtest_res[p].get("scores",{}).get("top1",0)*100:.1f}%'} for p in POSITIONS]), use_container_width=True, hide_index=True)

if __name__ == "__main__": main()
