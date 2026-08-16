# ============================================================
# 🤖 LOTTO AI PRO V8.4 TURBO EXTREME (UPGRADED)
# ============================================================
# STRICT WALK-FORWARD
# LEAKAGE SAFE
# NO PERSISTENT MEMORY
# THAI LOTTERY 6D + 2D
# NORMAL LOTTERY 3D + 2D
#
# 🚀 TURBO EXTREME + ACCURACY BOOST
# ------------------------------------------------------------
# ✅ Fast Feature Engineering + EMA Trend
# ✅ Float32 Matrix
# ✅ Lightweight Feature Selection (Slightly increased)
# ✅ Fast Walk-Forward
# ✅ Backtest = ExtraTrees only
# ✅ Final = Weighted Ensemble (ET 40% + HGB 60%)
# ✅ Adaptive Configuration (Tuned)
# ✅ No Future Leakage
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
# 1. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lotto AI V8.4 Turbo Extreme",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. DATA SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "หวยธกส":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

    "หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "หวยมาเลย์":
        "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",

    "หวยหุ้นไทยเย็น":
        "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",

    "หวยหุ้นนิเคอิบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",

    "หวยหุ้นฮั่งเส็งบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",

    "หวยหุ้นจีนบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}


# ============================================================
# 3. CONSTANTS
# ============================================================

DOW_NAMES = [
    "จันทร์",
    "อังคาร",
    "พุธ",
    "พฤหัสบดี",
    "ศุกร์",
    "เสาร์",
    "อาทิตย์"
]


MODEL_NAMES = [
    "ExtraTrees",
    "HistGradientBoosting"
]


THAI_POSITIONS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "T2",
    "O2"
]


NORMAL_POSITIONS = [
    "H",
    "T",
    "O",
    "T2",
    "O2"
]


POSITION_LABELS = {

    "H1":
        "💯 หลักแสน 6 ตัว",

    "H2":
        "🔢 หลักหมื่น 6 ตัว",

    "H3":
        "🔢 หลักพัน 6 ตัว",

    "H4":
        "🔢 หลักร้อย 6 ตัว",

    "H5":
        "🔟 หลักสิบ 6 ตัว",

    "H6":
        "1️⃣ หลักหน่วย 6 ตัว",

    "H":
        "💯 หลักร้อย 3 ตัวบน",

    "T":
        "🔟 หลักสิบ 3 ตัวบน",

    "O":
        "1️⃣ หลักหน่วย 3 ตัวบน",

    "T2":
        "🔽 หลักสิบ 2 ตัวล่าง",

    "O2":
        "⬇️ หลักหน่วย 2 ตัวล่าง",
}


# ============================================================
# 4. CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        .stApp {
            background:#f8fafc;
        }

        .main-title {
            text-align:center;
            font-size:2.15rem;
            font-weight:900;
        }

        .subtitle {
            text-align:center;
            color:#64748b;
            font-size:.88rem;
            margin-bottom:15px;
        }

        .status-card {
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:14px;
            padding:13px;
            text-align:center;
            color:#1e40af;
            font-weight:700;
            line-height:1.7;
        }

        .hot-card {
            background:#f0fdf4;
            border-left:7px solid #16a34a;
            border-radius:14px;
            padding:14px;
            margin:8px 0;
        }

        .dead-card {
            background:#fef2f2;
            border-left:7px solid #dc2626;
            border-radius:14px;
            padding:14px;
            margin:8px 0;
        }

        .position-title {
            font-size:1.05rem;
            font-weight:900;
            color:#334155;
        }

        .hot-number,
        .dead-number {
            font-size:2.05rem;
            font-weight:900;
            letter-spacing:3px;
            text-align:center;
        }

        .hot-number {
            color:#16a34a;
        }

        .dead-number {
            color:#dc2626;
        }

        .prob-text {
            text-align:center;
            color:#64748b;
            font-size:.8rem;
        }

        .model-badge {
            text-align:center;
            background:white;
            border-radius:9px;
            padding:6px;
            margin-top:6px;
            color:#475569;
            font-weight:700;
        }

        .confidence {
            text-align:center;
            font-size:.85rem;
            font-weight:800;
            margin-top:5px;
        }

        div.stButton > button {
            width:100%;
            min-height:48px;
            border-radius:10px;
            font-size:16px;
            font-weight:800;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. THAI MONTH
# ============================================================

THAI_MONTHS = {

    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,

    "ม.ค.": 1,
    "ก.พ.": 2,
    "มี.ค.": 3,
    "เม.ย.": 4,
    "พ.ค.": 5,
    "มิ.ย.": 6,
    "ก.ค.": 7,
    "ส.ค.": 8,
    "ก.ย.": 9,
    "ต.ค.": 10,
    "พ.ย.": 11,
    "ธ.ค.": 12
}


# ============================================================
# 6. DATE NORMALIZER
# ============================================================

def normalize_date(value):

    if not value:
        return None

    text = str(value).strip()

    for name, month in THAI_MONTHS.items():

        match = re.search(
            rf"(\d{{1,2}})\s*{re.escape(name)}\s*(\d{{4}})",
            text
        )

        if match:

            y = int(match.group(2))

            if y >= 2400:
                y -= 543

            try:

                return pd.Timestamp(
                    y,
                    month,
                    int(match.group(1))
                )

            except Exception:
                return None

    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a, b, c = map(
            int,
            match.groups()
        )

        if a >= 1000:

            y = a
            m = b
            d = c

        else:

            y = c
            m = b
            d = a

        if y < 100:
            y += 2000

        if y >= 2400:
            y -= 543

        try:

            return pd.Timestamp(
                y,
                m,
                d
            )

        except Exception:
            pass

    return None


# ============================================================
# 7. SCRAPER
# ============================================================

class ScrapingError(Exception):
    pass


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def fetch_lottery_data(url):

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Linux; Android 10) "
            "AppleWebKit/537.36 "
            "Chrome/120 Mobile Safari/537.36",

        "Accept-Language":
            "th-TH,th;q=0.9,en;q=0.8"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content = (
            soup.find(
                "div",
                class_=re.compile(
                    r"post-body|entry-content|post-content|content",
                    re.I
                )
            )
            or soup
        )

        rows = []

        # ====================================================
        # TABLE
        # ====================================================

        for row in content.find_all("tr"):

            cells = [
                c.get_text(
                    " ",
                    strip=True
                )
                for c in row.find_all(
                    ["td", "th"]
                )
            ]

            text = " ".join(cells)

            if not text:
                continue

            date = normalize_date(text)

            if date is None:
                continue

            six = re.findall(
                r"(?<!\d)\d{6}(?!\d)",
                text
            )

            three = re.findall(
                r"(?<!\d)\d{3}(?!\d)",
                text
            )

            two = re.findall(
                r"(?<!\d)\d{2}(?!\d)",
                text
            )

            if six and two:

                rows.append({

                    "Date":
                        date,

                    "Result_6D":
                        six[0],

                    "Result_3D":
                        six[0][-3:],

                    "Result_2D":
                        two[-1]
                })

            elif three and two:

                rows.append({

                    "Date":
                        date,

                    "Result_6D":
                        None,

                    "Result_3D":
                        three[0],

                    "Result_2D":
                        two[-1]
                })

        # ====================================================
        # TEXT FALLBACK
        # ====================================================

        if not rows:

            text = content.get_text(
                separator="\n",
                strip=True
            )

            lines = [
                x.strip()
                for x in text.splitlines()
                if x.strip()
            ]

            current_date = None

            for line in lines:

                date = normalize_date(line)

                if date is not None:
                    current_date = date

                if current_date is None:
                    continue

                six = re.findall(
                    r"(?<!\d)\d{6}(?!\d)",
                    line
                )

                three = re.findall(
                    r"(?<!\d)\d{3}(?!\d)",
                    line
                )

                two = re.findall(
                    r"(?<!\d)\d{2}(?!\d)",
                    line
                )

                if six and two:

                    rows.append({

                        "Date":
                            current_date,

                        "Result_6D":
                            six[0],

                        "Result_3D":
                            six[0][-3:],

                        "Result_2D":
                            two[-1]
                    })

                elif three and two:

                    rows.append({

                        "Date":
                            current_date,

                        "Result_6D":
                            None,

                        "Result_3D":
                            three[0],

                        "Result_2D":
                            two[-1]
                    })

        if not rows:

            raise ScrapingError(
                "ไม่พบข้อมูลหวย"
            )

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.extract(
                r"(\d{3})"
            )[0]
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(
                r"(\d{2})"
            )[0]
            .str.zfill(2)
        )

        if "Result_6D" in df.columns:

            df["Result_6D"] = (
                df["Result_6D"]
                .astype(str)
                .str.extract(
                    r"(\d{6})"
                )[0]
            )

        df = (
            df
            .dropna(
                subset=["Date"]
            )
            .drop_duplicates(
                subset=["Date"]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        raise ScrapingError(
            f"โหลดข้อมูลไม่สำเร็จ: {exc}"
        )


# ============================================================
# 8. FORMAT DETECTION
# ============================================================

def is_thai_6d(df):

    if "Result_6D" not in df.columns:
        return False

    return (
        df["Result_6D"]
        .notna()
        .sum()
        >= 10
    )


# ============================================================
# 9. FAST FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    thai_6d=False
):

    w = df.copy()

    # ========================================================
    # DIGITS
    # ========================================================

    if thai_6d:

        six = (
            w["Result_6D"]
            .fillna("000000")
            .astype(str)
            .str.zfill(6)
        )

        for i in range(6):

            w[f"H{i+1}"] = (
                six.str[i]
                .astype(np.int8)
            )

    else:

        three = (
            w["Result_3D"]
            .astype(str)
            .str.zfill(3)
        )

        w["H"] = (
            three.str[0]
            .astype(np.int8)
        )

        w["T"] = (
            three.str[1]
            .astype(np.int8)
        )

        w["O"] = (
            three.str[2]
            .astype(np.int8)
        )

    two = (
        w["Result_2D"]
        .astype(str)
        .str.zfill(2)
    )

    w["T2"] = (
        two.str[0]
        .astype(np.int8)
    )

    w["O2"] = (
        two.str[1]
        .astype(np.int8)
    )

    # ========================================================
    # DATE
    # ========================================================

    dt = w["Date"].dt

    w["DOW"] = (
        dt.dayofweek
        .astype(np.int8)
    )

    w["DAY"] = (
        dt.day
        .astype(np.int8)
    )

    w["MONTH"] = (
        dt.month
        .astype(np.int8)
    )

    w["DAY_OF_YEAR"] = (
        dt.dayofyear
        .astype(np.int16)
    )

    w["DOW_SIN"] = np.sin(
        2 * np.pi * w["DOW"] / 7
    ).astype(np.float32)

    w["DOW_COS"] = np.cos(
        2 * np.pi * w["DOW"] / 7
    ).astype(np.float32)

    w["MONTH_SIN"] = np.sin(
        2 * np.pi * w["MONTH"] / 12
    ).astype(np.float32)

    w["MONTH_COS"] = np.cos(
        2 * np.pi * w["MONTH"] / 12
    ).astype(np.float32)

    # ========================================================
    # POSITIONS
    # ========================================================

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    # ========================================================
    # FAST POSITION FEATURES
    # ========================================================

    for pos in positions:

        s = w[pos]
        p = s.shift(1)

        # ----------------------------------------------------
        # LAGS
        # ----------------------------------------------------

        for lag in (1, 2, 3, 5):
            w[f"{pos}_L{lag}"] = s.shift(lag)

        # ----------------------------------------------------
        # ROLLING
        # ----------------------------------------------------

        for window in (10, 20):
            r = p.rolling(window, min_periods=2)
            w[f"{pos}_M{window}"] = r.mean()
            w[f"{pos}_S{window}"] = r.std()

            # เฉพาะ digit 0 และ 5
            for digit in (0, 5):
                w[f"{pos}_F{window}_{digit}"] = (
                    (p == digit)
                    .astype(np.float32)
                    .rolling(window, min_periods=2)
                    .mean()
                )

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        w[f"{pos}_D1"] = s.shift(1) - s.shift(2)
        w[f"{pos}_D2"] = s.shift(2) - s.shift(3)

        # ----------------------------------------------------
        # SIMPLE STATES
        # ----------------------------------------------------

        w[f"{pos}_ODD"] = (p % 2)
        w[f"{pos}_HIGH"] = (p >= 5).astype(np.float32)
        w[f"{pos}_MOD3"] = (p % 3)

        # ----------------------------------------------------
        # CYCLIC DIGIT
        # ----------------------------------------------------

        w[f"{pos}_SIN"] = np.sin(2 * np.pi * p / 10).astype(np.float32)
        w[f"{pos}_COS"] = np.cos(2 * np.pi * p / 10).astype(np.float32)

        # ----------------------------------------------------
        # EWMA & TREND (UPGRADED) 🚀
        # ----------------------------------------------------

        w[f"{pos}_EWMA3"] = p.ewm(span=3, adjust=False).mean()
        w[f"{pos}_EWMA7"] = p.ewm(span=7, adjust=False).mean()
        # โมเมนตัมแนวโน้มระยะสั้นตัดระยะกลาง (MACD 3-7) ช่วยให้ AI เดาแพทเทิร์นได้คมขึ้น
        w[f"{pos}_TREND"] = w[f"{pos}_EWMA3"] - w[f"{pos}_EWMA7"]

        # ----------------------------------------------------
        # REPEAT
        # ----------------------------------------------------

        w[f"{pos}_REPEAT"] = (p == s.shift(2)).astype(np.float32)

    # ========================================================
    # PREVIOUS DRAW AGGREGATES
    # ========================================================

    if thai_6d:
        base = w[["H1", "H2", "H3", "H4", "H5", "H6"]].shift(1)
    else:
        base = w[["H", "T", "O"]].shift(1)

    w["PREV_SUM"] = base.sum(axis=1)
    w["PREV_RANGE"] = base.max(axis=1) - base.min(axis=1)
    w["PREV_MEAN"] = base.mean(axis=1)
    w["PREV_ODD"] = (base % 2).sum(axis=1)
    w["PREV_HIGH"] = (base >= 5).sum(axis=1)
    w["PREV_UNIQUE"] = base.nunique(axis=1)

    return (
        w.replace([np.inf, -np.inf], np.nan)
    )


# ============================================================
# 10. FEATURE LIST
# ============================================================

def get_features(thai_6d):

    base = [
        "DOW", "DAY", "MONTH", "DAY_OF_YEAR",
        "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS",
        "PREV_SUM", "PREV_RANGE", "PREV_MEAN",
        "PREV_ODD", "PREV_HIGH", "PREV_UNIQUE"
    ]

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    for pos in positions:

        base.extend([
            f"{pos}_L1", f"{pos}_L2", f"{pos}_L3", f"{pos}_L5",
            f"{pos}_M10", f"{pos}_M20",
            f"{pos}_S10", f"{pos}_S20",
            f"{pos}_D1", f"{pos}_D2",
            f"{pos}_ODD", f"{pos}_HIGH", f"{pos}_MOD3",
            f"{pos}_SIN", f"{pos}_COS",
            f"{pos}_EWMA3", f"{pos}_EWMA7", f"{pos}_TREND", # 🚀 Added New Features
            f"{pos}_REPEAT"
        ])

        for window in (10, 20):
            for digit in (0, 5):
                base.append(f"{pos}_F{window}_{digit}")

    return list(dict.fromkeys(base))


# ============================================================
# 11. ADAPTIVE CONFIG (TUNED FOR ACCURACY) 🧠
# ============================================================

def get_adaptive_config(n):
    # ปรับเพิ่ม selected_features เล็กน้อยเพื่อให้ข้อมูลกับโมเดลมากขึ้น
    if n >= 700:
        return {
            "min_train": 120,
            "trees": 55,
            "depth": 7,
            "leaf": 2,
            "selected_features": 20,
            "backtest_points": 6,
            "recent_decay": 0.985
        }
    if n >= 400:
        return {
            "min_train": 100,
            "trees": 45,
            "depth": 6,
            "leaf": 2,
            "selected_features": 18,
            "backtest_points": 6,
            "recent_decay": 0.98
        }
    if n >= 200:
        return {
            "min_train": 80,
            "trees": 38,
            "depth": 6,
            "leaf": 2,
            "selected_features": 16,
            "backtest_points": 5,
            "recent_decay": 0.975
        }
    return {
        "min_train": 50,
        "trees": 30,
        "depth": 5,
        "leaf": 2,
        "selected_features": 15,
        "backtest_points": 4,
        "recent_decay": 0.97
    }


# ============================================================
# 12. MODEL FACTORY
# ============================================================

def create_model(name, cfg):

    t = cfg["trees"]
    d = cfg["depth"]
    l = cfg["leaf"]

    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=t,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt", # ทำงานไวกว่าและกัน overfitting ได้ดี
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    # จูน HGB ให้แม่นขึ้น (เพิ่ม max_leaf_nodes และปรับ LR) โดยเวลาคำนวณยังเร็วเท่าเดิม
    return HistGradientBoostingClassifier(
        max_iter=max(25, int(t * 0.55)),
        max_leaf_nodes=15, 
        learning_rate=0.075,
        min_samples_leaf=l,
        l2_regularization=2,
        random_state=42
    )


# ============================================================
# 13. FAST FEATURE SELECTION
# ============================================================

def select_features_once(X, y, max_features):

    cols = list(X.columns)

    if len(cols) <= max_features:
        return cols

    valid = [
        c for c in cols
        if X[c].nunique(dropna=False) > 1
    ]

    if len(valid) <= max_features:
        return valid

    Xi = (
        X[valid]
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
        .fillna(0.0)
    )

    selector = ExtraTreesClassifier(
        n_estimators=8,
        max_depth=4,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=123
    )

    selector.fit(Xi, y)

    importance = selector.feature_importances_
    order = np.argsort(importance)[::-1]

    return [valid[i] for i in order[:max_features]]


# ============================================================
# 14. PROBABILITY
# ============================================================

def normalize_probability(p):

    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 1e-9, None)
    total = p.sum()

    if total <= 0:
        return np.ones(10, dtype=np.float32) / 10

    return (p / total).astype(np.float32)


# ============================================================
# 15. FAST MATRIX
# ============================================================

def prepare_matrix(X_train, X_test, selected):

    A = X_train[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    B = X_test[selected].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    med = A.median()

    A = A.fillna(med).fillna(0)
    B = B.fillna(med).fillna(0)

    return A, B


# ============================================================
# 16. TRAIN FINAL ENSEMBLE (WEIGHTED) ⚖️
# ============================================================

def train_models(X_train, y_train, X_test, cfg, selected_features):

    A, B = prepare_matrix(X_train, X_test, selected_features)
    
    preds_dict = {}

    for name in MODEL_NAMES:
        try:
            model = create_model(name, cfg)
            model.fit(A, y_train)
            raw = model.predict_proba(B)[0]

            out = np.zeros(10, dtype=np.float32)
            for cls, prob in zip(model.classes_, raw):
                cls = int(cls)
                if 0 <= cls <= 9:
                    out[cls] = prob

            preds_dict[name] = normalize_probability(out)

        except Exception:
            continue

    if not preds_dict:
        return np.ones(10, dtype=np.float32) / 10

    # ⚖️ ถ้ารันผ่านทั้ง 2 ตัว ให้น้ำหนัก HGB มากกว่า (เพราะมักจะจับแพทเทิร์นลึกได้ดีกว่า)
    if len(preds_dict) == 2:
        ensemble_prob = (preds_dict["ExtraTrees"] * 0.40) + (preds_dict["HistGradientBoosting"] * 0.60)
        return normalize_probability(ensemble_prob)
    else:
        # ถ้าพังไป 1 ตัว ให้ใช้ตัวที่เหลือ
        return normalize_probability(list(preds_dict.values())[0])


# ============================================================
# 17. FAST WALK-FORWARD
# ============================================================

def strict_walk_forward(df_feat, pos, features, cfg):

    X = df_feat[features].astype(np.float32)
    y = df_feat[pos].astype(np.int8)

    n = len(df_feat)
    start = cfg["min_train"]

    if n <= start + 2:
        return {"tests": 0, "scores": {}}

    available = (n - start)
    tests = min(cfg["backtest_points"], available)

    test_indices = np.linspace(start, n - 1, tests, dtype=int)
    test_indices = np.unique(test_indices)

    selection_end = test_indices[0]

    if y.iloc[:selection_end].nunique() < 2:
        return {"tests": 0, "scores": {}}

    selected = select_features_once(
        X.iloc[:selection_end],
        y.iloc[:selection_end],
        cfg["selected_features"]
    )

    records = []

    for idx in test_indices:
        if idx < start:
            continue

        y_train = y.iloc[:idx]
        if y_train.nunique() < 2:
            continue

        A, B = prepare_matrix(X.iloc[:idx], X.iloc[[idx]], selected)

        model = ExtraTreesClassifier(
            n_estimators=max(18, int(cfg["trees"] * 0.55)),
            max_depth=cfg["depth"],
            min_samples_leaf=cfg["leaf"],
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        try:
            model.fit(A, y_train)
            raw = model.predict_proba(B)[0]

            probs = np.zeros(10, dtype=np.float32)
            for cls, prob in zip(model.classes_, raw):
                cls = int(cls)
                if 0 <= cls <= 9:
                    probs[cls] = prob

            probs = normalize_probability(probs)
        except Exception:
            continue

        actual = int(y.iloc[idx])
        ranking = np.argsort(probs)[::-1]

        records.append({
            "top1": int(actual == ranking[0]),
            "top3": int(actual in ranking[:3]),
            "top5": int(actual in ranking[:5]),
            "dead7": int(actual in np.argsort(probs)[:7]),
            "logloss": -np.log(max(float(probs[actual]), 1e-9))
        })

    if not records:
        return {"tests": 0, "scores": {}}

    h = pd.DataFrame(records)

    decay = (cfg["recent_decay"] ** (len(h) - np.arange(len(h)) - 1))
    decay /= decay.sum()

    scores = {
        "top1": float(np.sum(h["top1"] * decay)),
        "top3": float(np.sum(h["top3"] * decay)),
        "top5": float(np.sum(h["top5"] * decay)),
        "dead7": float(np.sum(h["dead7"] * decay)),
        "logloss": float(np.sum(h["logloss"] * decay))
    }

    scores["score"] = (
        0.35 * scores["top1"] +
        0.30 * scores["top3"] +
        0.20 * scores["top5"] +
        0.10 * (1 / (1 + scores["logloss"])) +
        0.05 * (1 - scores["dead7"])
    )

    return {"tests": len(h), "scores": scores}


# ============================================================
# 18. FINAL PREDICTION
# ============================================================

def final_prediction(df_feat, pos, features, cfg):

    X = df_feat[features].astype(np.float32)
    y = df_feat[pos].astype(np.int8)

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test = X.iloc[[-1]]

    selected = select_features_once(
        X_train,
        y_train,
        cfg["selected_features"]
    )

    probs = train_models(
        X_train,
        y_train,
        X_test,
        cfg,
        selected
    )

    order = np.argsort(probs)[::-1]
    hot = [(int(i), float(probs[i])) for i in order[:5]]
    dead = [(int(i), float(probs[i])) for i in np.argsort(probs)[:7]]

    confidence = probs[order[0]] - probs[order[1]]

    return {
        "probabilities": probs,
        "hot": hot,
        "dead": dead,
        "confidence": float(confidence),
        "top3": float(probs[order[:3]].sum()),
        "selected_features": selected
    }


# ============================================================
# 19. DISPLAY CARD
# ============================================================

def display_card(pos, result, hot=True):

    data = result["hot"] if hot else result["dead"]
    style = "hot" if hot else "dead"

    nums = " - ".join(str(n) for n, _ in data)
    probs = " | ".join(f"{n}: {p*100:.1f}%" for n, p in data)

    html = f"""
    <div class="{style}-card">
        <div class="position-title">{POSITION_LABELS[pos]}</div>
        <div class="{style}-number">{nums}</div>
        <div class="prob-text">AI Probability: {probs}</div>
    """

    if hot:
        html += f"""
        <div class="confidence">
            📌 Top-1 Gap: {result["confidence"]*100:.1f}% &nbsp;|&nbsp; 
            Top-3: {result["top3"]*100:.1f}%
        </div>
        """

    html += """
        <div class="model-badge">
            🤖 AI Ensemble: ExtraTrees (40%) + HistGradientBoosting (60%)
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# 20. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        """
        <div class="main-title">
            🤖 LOTTO AI PRO V8.4 TURBO EXTREME
        </div>
        <div class="subtitle">
            STRICT WALK-FORWARD • LEAKAGE SAFE • NO PERSISTENT MEMORY • ⚡ TURBO + ACCURACY BOOST
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    lottery = c1.selectbox(
        "🏷️ เลือกประเภทหวย",
        list(LOTTERY_SOURCES.keys())
    )

    selected_day = c2.selectbox(
        "📅 วันเป้าหมาย",
        ["อัตโนมัติ"] + DOW_NAMES
    )

    if not st.button("🚀 เริ่มวิเคราะห์ V8.4 (TURBO + ACCURACY)", type="primary", use_container_width=True):
        return

    with st.spinner("📥 โหลดข้อมูล..."):
        try:
            df = fetch_lottery_data(LOTTERY_SOURCES[lottery])
        except Exception as exc:
            st.error(str(exc))
            return

    if len(df) < 50:
        st.error(f"❌ มีข้อมูล {len(df)} งวด ต้องมีอย่างน้อย 50 งวด")
        return

    thai_6d = (lottery == "หวยไทย" and is_thai_6d(df))
    positions = (THAI_POSITIONS if thai_6d else NORMAL_POSITIONS)

    last_date = pd.Timestamp(df["Date"].iloc[-1])

    if selected_day == "อัตโนมัติ":
        if len(df) >= 2:
            gap = (df["Date"].iloc[-1] - df["Date"].iloc[-2]).days
            days_ahead = max(int(gap), 1)
        else:
            days_ahead = 7
    else:
        target_dow = DOW_NAMES.index(selected_day)
        days_ahead = (target_dow - last_date.dayofweek) % 7
        if days_ahead == 0:
            days_ahead = 7

    target_date = last_date + timedelta(days=days_ahead)

    dummy = {
        "Date": target_date,
        "Result_3D": "000",
        "Result_2D": "00"
    }

    if thai_6d:
        dummy["Result_6D"] = "000000"

    ext = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)

    with st.spinner("⚡ สร้าง Turbo Feature Matrix (Upgraded)..."):
        feat = build_features(ext, thai_6d)
        features = get_features(thai_6d)
        cfg = get_adaptive_config(len(df))

    st.info(
        f"""
        ⚡ V8.4 TURBO EXTREME | ข้อมูล {len(df):,} งวด | 
        {"หวยไทย 6 หลัก + 2 หลัก" if thai_6d else "3 หลัก + 2 หลัก"} | 
        Features {len(features)} | Selected ≤ {cfg["selected_features"]} | 
        Trees {cfg["trees"]} | WF Tests {cfg["backtest_points"]}
        """
    )

    backtest = {}
    final = {}

    progress = st.progress(0)
    status = st.empty()
    total = len(positions)

    for i, pos in enumerate(positions):

        status.caption(f"⚡ V8.4 TURBO: {POSITION_LABELS[pos]}")

        backtest[pos] = strict_walk_forward(feat.iloc[:-1], pos, features, cfg)
        final[pos] = final_prediction(feat, pos, features, cfg)

        progress.progress(int(((i + 1) / total) * 100))

    progress.empty()
    status.empty()

    tests = [x["tests"] for x in backtest.values() if x.get("tests", 0) > 0]
    test_text = f"{min(tests)}–{max(tests)} จุด" if tests else "ไม่มี"

    st.markdown(
        f"""
        <div class="status-card">
        🤖 <b>LOTTO AI PRO V8.4 TURBO EXTREME</b><br>
        📊 ข้อมูล: {len(df):,} งวด<br>
        📅 เป้าหมาย: {target_date.strftime("%d/%m/%Y")}<br>
        🎯 Mode: {"6D + 2D" if thai_6d else "3D + 2D"}<br>
        🧠 Features: {len(features)}<br>
        🎯 Selected: ≤{cfg["selected_features"]}<br>
        🌳 Trees: {cfg["trees"]}<br>
        ⚡ Walk-Forward: {test_text}
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

    t1, t2, t3 = st.tabs(["🎯 เลขเด่น AI", "🛑 เลขดับ 7", "📊 Backtest"])

    with t1:
        for pos in positions:
            display_card(pos, final[pos], True)

    with t2:
        for pos in positions:
            display_card(pos, final[pos], False)

    with t3:
        for pos in positions:
            st.markdown(f"### {POSITION_LABELS[pos]}")
            score = backtest[pos].get("scores", {})

            if not score:
                st.warning("ไม่มี Backtest")
                continue

            st.dataframe(
                pd.DataFrame([
                    {"Metric": "Top-1", "AI": f"{score['top1']*100:.1f}%", "Random": "10%", "Edge": f"{(score['top1']-.10)*100:+.1f}%"},
                    {"Metric": "Top-3", "AI": f"{score['top3']*100:.1f}%", "Random": "30%", "Edge": f"{(score['top3']-.30)*100:+.1f}%"},
                    {"Metric": "Top-5", "AI": f"{score['top5']*100:.1f}%", "Random": "50%", "Edge": f"{(score['top5']-.50)*100:+.1f}%"},
                    {"Metric": "Dead-7", "AI": f"{score['dead7']*100:.1f}%", "Random": "70%", "Edge": f"{(score['dead7']-.70)*100:+.1f}%"},
                    {"Metric": "LogLoss", "AI": f"{score['logloss']:.3f}", "Random": "-", "Edge": "-"}
                ]),
                use_container_width=True,
                hide_index=True
            )

    with st.expander("🎯 สรุปเลขเด่นทั้งหมด"):
        data = []
        for pos in positions:
            sc = backtest[pos].get("scores", {})
            data.append({
                "ตำแหน่ง": POSITION_LABELS[pos],
                "Top-1": final[pos]["hot"][0][0],
                "Probability": f'{final[pos]["hot"][0][1]*100:.1f}%',
                "WF Top-1": f'{sc.get("top1", 0)*100:.1f}%',
                "WF Top-3": f'{sc.get("top3", 0)*100:.1f}%',
                "Confidence": f'{final[pos]["confidence"]*100:.1f}%'
            })

        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
