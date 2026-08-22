# ============================================================
# 🤖 LOTTO AI PRO V9.1 ADAPTIVE STABILITY TURBO
# ============================================================
# UPGRADED FROM V9.0
#
# ✅ LEAKAGE-SAFE WALK-FORWARD
# ✅ EXTRA TREES + HIST GRADIENT BOOSTING
# ✅ ADAPTIVE FEATURE SELECTION
# ✅ RECENCY WEIGHTING
# ✅ STABILITY-AWARE BACKTEST
# ✅ HOT TOP-3
# ✅ COLD / DEAD TOP-7
# ✅ FIXED DEAD BACKTEST METRIC
# ✅ NO BALANCED CLASS WEIGHT
# ✅ REDUCED REDUNDANT FEATURES
# ✅ FAST FEATURE REUSE
# ✅ MOBILE FRIENDLY UI
#
# NOTE:
# This system estimates historical statistical patterns.
# It cannot guarantee future lottery results.
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
    page_title="Lotto AI V9.1 Adaptive Turbo",
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
    "H1", "H2", "H3",
    "H4", "H5", "H6",
    "T2", "O2"
]

NORMAL_POSITIONS = [
    "H", "T", "O",
    "T2", "O2"
]

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
# 4. CSS
# ============================================================

def inject_css():

    st.markdown(
        """
        <style>

        .stApp {
            background: #f8fafc;
        }

        .main-title {
            text-align:center;
            font-size:2.2rem;
            font-weight:900;
            color:#1e293b;
            margin-bottom:5px;
        }

        .subtitle {
            text-align:center;
            color:#64748b;
            font-size:1rem;
            margin-bottom:25px;
        }

        .status-card {
            background:#eff6ff;
            border:1px solid #bfdbfe;
            border-radius:14px;
            padding:15px;
            text-align:center;
            color:#1e40af;
            font-weight:600;
        }

        .hot-card {
            background:#f0fdf4;
            border-left:6px solid #16a34a;
            border-radius:10px;
            padding:15px;
            margin-bottom:15px;
            box-shadow:0 2px 4px rgba(0,0,0,0.05);
            height:100%;
        }

        .dead-card {
            background:#fef2f2;
            border-left:6px solid #dc2626;
            border-radius:10px;
            padding:15px;
            margin-bottom:15px;
            box-shadow:0 2px 4px rgba(0,0,0,0.05);
            height:100%;
        }

        .position-title {
            font-size:1.1rem;
            font-weight:800;
            color:#334155;
            margin-bottom:10px;
            text-align:center;
            border-bottom:1px solid #e2e8f0;
            padding-bottom:8px;
        }

        .hot-number {
            font-size:2.2rem;
            font-weight:900;
            letter-spacing:4px;
            text-align:center;
            color:#16a34a;
        }

        .dead-number {
            font-size:1.8rem;
            font-weight:800;
            letter-spacing:2px;
            text-align:center;
            color:#dc2626;
        }

        .prob-text {
            text-align:center;
            color:#475569;
            font-size:0.9rem;
            line-height:1.6;
            margin-top:10px;
        }

        .prob-pill {
            display:inline-block;
            background:#e2e8f0;
            border-radius:20px;
            padding:2px 10px;
            margin:3px;
            font-weight:600;
            font-size:0.85rem;
        }

        .confidence {
            text-align:center;
            font-size:0.85rem;
            font-weight:700;
            margin-top:10px;
            color:#64748b;
        }

        div.stButton > button {
            min-height:50px;
            border-radius:10px;
            font-size:18px;
            font-weight:800;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 5. DATE NORMALIZER
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

        a, b, c = map(int, match.groups())

        if a >= 1000:
            y, m, d = a, b, c
        else:
            y, m, d = c, b, a

        if y < 100:
            y += 2000

        if y >= 2400:
            y -= 543

        try:
            return pd.Timestamp(y, m, d)
        except Exception:
            pass

    return None


class ScrapingError(Exception):
    pass


# ============================================================
# 6. DATA FETCH
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def fetch_lottery_data(url):

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36",

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

        # ----------------------------------------------------
        # TABLE PARSER
        # ----------------------------------------------------

        for row in content.find_all("tr"):

            text = " ".join(
                [
                    c.get_text(" ", strip=True)
                    for c in row.find_all(["td", "th"])
                ]
            )

            date = normalize_date(text)

            if not date:
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

                rows.append(
                    {
                        "Date": date,
                        "Result_6D": six[0],
                        "Result_3D": six[0][-3:],
                        "Result_2D": two[-1]
                    }
                )

            elif three and two:

                rows.append(
                    {
                        "Date": date,
                        "Result_6D": None,
                        "Result_3D": three[0],
                        "Result_2D": two[-1]
                    }
                )

        # ----------------------------------------------------
        # TEXT FALLBACK
        # ----------------------------------------------------

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

                    rows.append(
                        {
                            "Date": current_date,
                            "Result_6D": six[0],
                            "Result_3D": six[0][-3:],
                            "Result_2D": two[-1]
                        }
                    )

                elif three and two:

                    rows.append(
                        {
                            "Date": current_date,
                            "Result_6D": None,
                            "Result_3D": three[0],
                            "Result_2D": two[-1]
                        }
                    )

        if not rows:
            raise ScrapingError(
                "ไม่พบข้อมูลหวยในรูปแบบที่รองรับ"
            )

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.extract(r"(\d{3})")[0]
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(r"(\d{2})")[0]
            .str.zfill(2)
        )

        if "Result_6D" in df.columns:

            df["Result_6D"] = (
                df["Result_6D"]
                .astype(str)
                .str.extract(r"(\d{6})")[0]
            )

        df = (
            df
            .dropna(subset=["Date"])
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        raise ScrapingError(
            f"โหลดข้อมูลไม่สำเร็จ: {exc}"
        )


def is_thai_6d(df):

    return (
        "Result_6D" in df.columns
        and
        df["Result_6D"].notna().sum() >= 10
    )


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def build_features(df, thai_6d=False):

    w = df.copy()

    # --------------------------------------------------------
    # DIGITS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CALENDAR
    # --------------------------------------------------------

    dt = w["Date"].dt

    w["DOW"] = dt.dayofweek.astype(np.int8)
    w["DAY"] = dt.day.astype(np.int8)
    w["MONTH"] = dt.month.astype(np.int8)

    w["IS_WEEKEND"] = (
        (w["DOW"] >= 5)
        .astype(np.float32)
    )

    w["IS_MONTH_START"] = (
        (w["DAY"] <= 5)
        .astype(np.float32)
    )

    w["IS_MONTH_END"] = (
        (w["DAY"] >= 25)
        .astype(np.float32)
    )

    w["DOW_SIN"] = (
        np.sin(
            2 * np.pi * w["DOW"] / 7
        )
        .astype(np.float32)
    )

    w["DOW_COS"] = (
        np.cos(
            2 * np.pi * w["DOW"] / 7
        )
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # POSITION FEATURES
    # --------------------------------------------------------

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    for pos in positions:

        s = w[pos]

        p = s.shift(1)

        # LAG
        for lag in (1, 2, 3, 5):

            w[f"{pos}_L{lag}"] = (
                s.shift(lag)
            )

        # ROLLING
        for window in (10, 20):

            r = p.rolling(
                window,
                min_periods=2
            )

            w[f"{pos}_M{window}"] = (
                r.mean()
            )

            w[f"{pos}_S{window}"] = (
                r.std()
            )

            # Frequency of 0 and 5
            for digit in (0, 5):

                w[
                    f"{pos}_F{window}_{digit}"
                ] = (
                    (p == digit)
                    .astype(np.float32)
                    .rolling(
                        window,
                        min_periods=2
                    )
                    .mean()
                )

        # MOMENTUM
        w[f"{pos}_D1"] = (
            s.shift(1) - s.shift(2)
        )

        w[f"{pos}_D2"] = (
            s.shift(2) - s.shift(3)
        )

        # PATTERN
        w[f"{pos}_ODD"] = (
            p % 2
        )

        w[f"{pos}_HIGH"] = (
            (p >= 5)
            .astype(np.float32)
        )

        w[f"{pos}_SIN"] = (
            np.sin(
                2 * np.pi * p / 10
            )
            .astype(np.float32)
        )

        w[f"{pos}_COS"] = (
            np.cos(
                2 * np.pi * p / 10
            )
            .astype(np.float32)
        )

        w[f"{pos}_EWMA7"] = (
            p.ewm(
                span=7,
                adjust=False
            ).mean()
        )

        # Repeat
        w[f"{pos}_REPEAT"] = (
            p == s.shift(2)
        ).astype(np.float32)

    # --------------------------------------------------------
    # PREVIOUS DRAW SUMMARY
    # --------------------------------------------------------

    if thai_6d:

        base = (
            w[
                [
                    "H1", "H2", "H3",
                    "H4", "H5", "H6"
                ]
            ]
            .shift(1)
        )

    else:

        base = (
            w[
                ["H", "T", "O"]
            ]
            .shift(1)
        )

    w["PREV_SUM"] = (
        base.sum(axis=1)
    )

    w["PREV_RANGE"] = (
        base.max(axis=1)
        -
        base.min(axis=1)
    )

    w["PREV_ODD"] = (
        (base % 2).sum(axis=1)
    )

    w["PREV_HIGH"] = (
        (base >= 5).sum(axis=1)
    )

    return (
        w.replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


# ============================================================
# 8. FEATURE LIST
# ============================================================

def get_features(thai_6d):

    base = [
        "DOW",
        "DAY",
        "MONTH",
        "IS_WEEKEND",
        "IS_MONTH_START",
        "IS_MONTH_END",
        "DOW_SIN",
        "DOW_COS",
        "PREV_SUM",
        "PREV_RANGE",
        "PREV_ODD",
        "PREV_HIGH",
    ]

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    for pos in positions:

        base.extend(
            [
                f"{pos}_L1",
                f"{pos}_L2",
                f"{pos}_L3",
                f"{pos}_L5",

                f"{pos}_M10",
                f"{pos}_M20",

                f"{pos}_S10",
                f"{pos}_S20",

                f"{pos}_D1",
                f"{pos}_D2",

                f"{pos}_ODD",
                f"{pos}_HIGH",

                f"{pos}_SIN",
                f"{pos}_COS",

                f"{pos}_EWMA7",

                f"{pos}_REPEAT",
            ]
        )

        for window in (10, 20):

            for digit in (0, 5):

                base.append(
                    f"{pos}_F{window}_{digit}"
                )

    return list(
        dict.fromkeys(base)
    )


# ============================================================
# 9. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "min_train": 120,
            "trees": 65,
            "depth": 8,
            "leaf": 3,
            "hot_features": 22,
            "dead_features": 18,
            "backtest_points": 8,
            "recent_decay": 0.985,
            "refresh_every": 2
        }

    if n >= 400:

        return {
            "min_train": 100,
            "trees": 55,
            "depth": 7,
            "leaf": 3,
            "hot_features": 20,
            "dead_features": 16,
            "backtest_points": 7,
            "recent_decay": 0.980,
            "refresh_every": 2
        }

    if n >= 200:

        return {
            "min_train": 80,
            "trees": 45,
            "depth": 6,
            "leaf": 3,
            "hot_features": 18,
            "dead_features": 15,
            "backtest_points": 6,
            "recent_decay": 0.975,
            "refresh_every": 3
        }

    return {
        "min_train": 50,
        "trees": 35,
        "depth": 5,
        "leaf": 3,
        "hot_features": 16,
        "dead_features": 14,
        "backtest_points": 5,
        "recent_decay": 0.970,
        "refresh_every": 3
    }


# ============================================================
# 10. MODEL
# ============================================================

def create_model(
    name,
    cfg,
    system="hot"
):

    trees = cfg["trees"]
    depth = cfg["depth"]
    leaf = cfg["leaf"]

    if system == "hot":

        if name == "ExtraTrees":

            return ExtraTreesClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=0.70,

                # IMPORTANT:
                # no balanced class weight
                class_weight=None,

                n_jobs=-1,
                random_state=42
            )

        return HistGradientBoostingClassifier(
            max_iter=max(
                25,
                int(trees * 0.65)
            ),
            max_leaf_nodes=15,
            learning_rate=0.04,
            min_samples_leaf=leaf,
            l2_regularization=2.0,
            random_state=42
        )

    # --------------------------------------------------------
    # DEAD / COLD MODEL
    # --------------------------------------------------------

    if name == "ExtraTrees":

        return ExtraTreesClassifier(
            n_estimators=max(
                25,
                int(trees * 0.80)
            ),
            max_depth=max(
                4,
                depth - 1
            ),
            min_samples_leaf=max(
                2,
                leaf
            ),
            max_features=0.45,
            class_weight=None,
            n_jobs=-1,
            random_state=91
        )

    return HistGradientBoostingClassifier(
        max_iter=max(
            20,
            int(trees * 0.50)
        ),
        max_leaf_nodes=9,
        learning_rate=0.035,
        min_samples_leaf=max(
            2,
            leaf
        ),
        l2_regularization=4.0,
        random_state=91
    )


# ============================================================
# 11. FEATURE SELECTION
# ============================================================

def select_features_once(
    X,
    y,
    max_features,
    system="hot"
):

    cols = list(X.columns)

    if len(cols) <= max_features:
        return cols

    valid = [
        c
        for c in cols
        if X[c].nunique(
            dropna=False
        ) > 1
    ]

    if len(valid) <= max_features:
        return valid

    Xi = (
        X[valid]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
        .fillna(0.0)
    )

    if system == "hot":

        selector = ExtraTreesClassifier(
            n_estimators=12,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.70,
            n_jobs=-1,
            random_state=123
        )

    else:

        selector = ExtraTreesClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=3,
            max_features=0.60,
            n_jobs=-1,
            random_state=321
        )

    try:

        selector.fit(
            Xi,
            y
        )

        importance = (
            selector.feature_importances_
        )

        order = np.argsort(
            importance
        )[::-1]

        return [
            valid[i]
            for i in order[:max_features]
        ]

    except Exception:

        return valid[:max_features]


# ============================================================
# 12. NORMALIZE
# ============================================================

def normalize_probability(p):

    p = np.asarray(
        p,
        dtype=np.float32
    )

    p = np.nan_to_num(
        p,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p = np.clip(
        p,
        1e-9,
        None
    )

    total = p.sum()

    if total <= 0:
        return np.ones(
            10,
            dtype=np.float32
        ) / 10.0

    return p / total


# ============================================================
# 13. MODEL PROBABILITY
# ============================================================

def model_probability(
    X_train,
    y_train,
    X_test,
    cfg,
    selected,
    system
):

    A = (
        X_train[selected]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    B = (
        X_test[selected]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    med = A.median()

    A = (
        A.fillna(med)
        .fillna(0.0)
    )

    B = (
        B.fillna(med)
        .fillna(0.0)
    )

    predictions = []

    # --------------------------------------------------------
    # RECENCY WEIGHT
    # --------------------------------------------------------

    age = np.arange(
        len(A) - 1,
        -1,
        -1,
        dtype=np.float32
    )

    weights = (
        cfg["recent_decay"]
        ** age
    )

    weights = (
        weights /
        (
            weights.mean()
            + 1e-9
        )
    ).astype(np.float32)

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    for name in MODEL_NAMES:

        try:

            model = create_model(
                name,
                cfg,
                system
            )

            fitted = False

            try:

                model.fit(
                    A,
                    y_train,
                    sample_weight=weights
                )

                fitted = True

            except Exception:

                pass

            if not fitted:

                model.fit(
                    A,
                    y_train
                )

            raw = model.predict_proba(B)[0]

            out = np.zeros(
                10,
                dtype=np.float32
            )

            for cls, prob in zip(
                model.classes_,
                raw
            ):

                c = int(cls)

                if 0 <= c <= 9:
                    out[c] = prob

            predictions.append(
                normalize_probability(out)
            )

        except Exception:

            continue

    if not predictions:

        return np.ones(
            10,
            dtype=np.float32
        ) / 10.0

    if len(predictions) == 2:

        if system == "hot":

            ensemble = (
                predictions[0] * 0.60
                +
                predictions[1] * 0.40
            )

        else:

            ensemble = (
                predictions[0] * 0.55
                +
                predictions[1] * 0.45
            )

    else:

        ensemble = np.mean(
            predictions,
            axis=0
        )

    return normalize_probability(
        ensemble
    )


# ============================================================
# 14. ENTROPY / STABILITY
# ============================================================

def probability_entropy(p):

    p = normalize_probability(p)

    return float(
        -np.sum(
            p * np.log(
                p + 1e-12
            )
        )
    )


def probability_concentration(p):

    p = normalize_probability(p)

    entropy = probability_entropy(p)

    max_entropy = np.log(10)

    score = 1.0 - (
        entropy /
        max_entropy
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ============================================================
# 15. HOT SYSTEM
# ============================================================

def hot_system(
    X_train,
    y_train,
    X_test,
    cfg,
    selected=None
):

    if selected is None:

        selected = select_features_once(
            X_train,
            y_train,
            cfg["hot_features"],
            "hot"
        )

    probability = model_probability(
        X_train,
        y_train,
        X_test,
        cfg,
        selected,
        "hot"
    )

    # No aggressive sharpening.
    # Probability remains calibrated as much as possible.
    hot_prob = normalize_probability(
        probability
    )

    order = np.argsort(
        hot_prob
    )[::-1]

    top1 = float(
        hot_prob[order[0]]
    )

    top2 = float(
        hot_prob[order[1]]
    )

    top3_mass = float(
        hot_prob[
            order[:3]
        ].sum()
    )

    gap = (
        top1
        -
        top2
    )

    concentration = (
        probability_concentration(
            hot_prob
        )
    )

    return {
        "probability": hot_prob,

        "hot": [
            (
                int(n),
                float(hot_prob[n])
            )
            for n in order[:3]
        ],

        "top1_probability": top1,

        "top3": top3_mass,

        "top_gap": float(gap),

        "concentration": concentration,

        "selected_features": selected
    }


# ============================================================
# 16. COLD / DEAD SYSTEM
# ============================================================

def build_dead_score(
    probability,
    y_train
):

    probability = normalize_probability(
        probability
    )

    # --------------------------------------------------------
    # 1. AI LOW-PROBABILITY EVIDENCE
    # --------------------------------------------------------

    inverse_ai = (
        1.0
        -
        probability
    )

    inverse_ai = normalize_probability(
        inverse_ai
    )

    # --------------------------------------------------------
    # 2. RECENT COLDNESS
    # --------------------------------------------------------

    recent_n = min(
        20,
        len(y_train)
    )

    recent = np.asarray(
        y_train.iloc[-recent_n:],
        dtype=np.int8
    )

    freq = np.bincount(
        recent,
        minlength=10
    ).astype(np.float32)

    recent_freq = (
        freq /
        max(
            1,
            recent_n
        )
    )

    # Low frequency = cold
    cold_frequency = (
        1.0
        -
        recent_freq
    )

    cold_frequency = normalize_probability(
        cold_frequency
    )

    # --------------------------------------------------------
    # 3. GAP
    # --------------------------------------------------------

    gaps = np.zeros(
        10,
        dtype=np.float32
    )

    for digit in range(10):

        locations = np.where(
            np.asarray(y_train) == digit
        )[0]

        if len(locations) == 0:

            gaps[digit] = min(
                len(y_train),
                20
            )

        else:

            gaps[digit] = (
                len(y_train)
                -
                1
                -
                locations[-1]
            )

    if gaps.max() > 0:

        gap_score = (
            gaps /
            gaps.max()
        )

    else:

        gap_score = np.zeros(
            10,
            dtype=np.float32
        )

    gap_score = normalize_probability(
        gap_score
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    dead_score = (
        inverse_ai * 0.65
        +
        cold_frequency * 0.20
        +
        gap_score * 0.15
    )

    return normalize_probability(
        dead_score
    )


def dead_system(
    X_train,
    y_train,
    X_test,
    cfg,
    selected=None
):

    if selected is None:

        selected = select_features_once(
            X_train,
            y_train,
            cfg["dead_features"],
            "dead"
        )

    probability = model_probability(
        X_train,
        y_train,
        X_test,
        cfg,
        selected,
        "dead"
    )

    dead_score = build_dead_score(
        probability,
        y_train
    )

    order = np.argsort(
        dead_score
    )[::-1]

    return {
        "probability": probability,

        "dead_score": dead_score,

        "dead": [
            (
                int(n),
                float(dead_score[n])
            )
            for n in order[:7]
        ],

        "top7": float(
            dead_score[
                order[:7]
            ].sum()
        ),

        "selected_features": selected
    }


# ============================================================
# 17. WALK-FORWARD BACKTEST
# ============================================================

def walk_forward_system(
    df_feat,
    pos,
    features,
    cfg,
    system="hot"
):

    X = (
        df_feat[features]
        .astype(np.float32)
    )

    y = (
        df_feat[pos]
        .astype(np.int8)
    )

    start = cfg["min_train"]

    if len(df_feat) <= start + 2:

        return {
            "tests": 0,
            "scores": {},
            "stability": 0.0
        }

    tests = min(
        cfg["backtest_points"],
        len(df_feat) - start
    )

    test_indices = np.unique(
        np.linspace(
            start,
            len(df_feat) - 1,
            tests,
            dtype=int
        )
    )

    records = []

    selected = None

    refresh_every = cfg[
        "refresh_every"
    ]

    for step, idx in enumerate(
        test_indices
    ):

        train_X = X.iloc[:idx]
        train_y = y.iloc[:idx]

        if train_y.nunique() < 2:
            continue

        # ----------------------------------------------------
        # ADAPTIVE FEATURE REFRESH
        # ----------------------------------------------------

        if (
            selected is None
            or
            step % refresh_every == 0
        ):

            selected = select_features_once(
                train_X,
                train_y,
                cfg[
                    f"{system}_features"
                ],
                system
            )

        try:

            probs = model_probability(
                train_X,
                train_y,
                X.iloc[[idx]],
                cfg,
                selected,
                system
            )

        except Exception:

            continue

        actual = int(
            y.iloc[idx]
        )

        if system == "hot":

            order = np.argsort(
                probs
            )[::-1]

            records.append(
                {
                    "top1": int(
                        actual == order[0]
                    ),

                    "top3": int(
                        actual in order[:3]
                    ),

                    "top5": int(
                        actual in order[:5]
                    )
                }
            )

        else:

            dead_score = build_dead_score(
                probs,
                train_y
            )

            dead_order = np.argsort(
                dead_score
            )[::-1]

            # Dead Hit:
            # actual digit is inside
            # the predicted low-probability group.
            records.append(
                {
                    "dead5": int(
                        actual
                        in dead_order[:5]
                    ),

                    "dead7": int(
                        actual
                        in dead_order[:7]
                    )
                }
            )

    if not records:

        return {
            "tests": 0,
            "scores": {},
            "stability": 0.0
        }

    h = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # RECENCY-WEIGHTED BACKTEST
    # --------------------------------------------------------

    decay = (
        cfg["recent_decay"]
        **
        np.arange(
            len(h) - 1,
            -1,
            -1
        )
    )

    decay = (
        decay /
        (
            decay.sum()
            + 1e-12
        )
    )

    scores = {}

    for col in h.columns:

        scores[col] = float(
            np.sum(
                h[col].values
                *
                decay
            )
        )

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    top3_values = (
        h["top3"].values
        if "top3" in h.columns
        else h["dead7"].values
    )

    if len(top3_values) > 1:

        mean_value = np.mean(
            top3_values
        )

        std_value = np.std(
            top3_values
        )

        stability = float(
            np.clip(
                1.0
                -
                (
                    std_value /
                    max(
                        mean_value,
                        0.10
                    )
                ),
                0.0,
                1.0
            )
        )

    else:

        stability = 0.5

    return {
        "tests": len(records),
        "scores": scores,
        "stability": stability
    }


# ============================================================
# 18. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_feat,
    pos,
    features,
    cfg
):

    X = (
        df_feat[features]
        .astype(np.float32)
    )

    y = (
        df_feat[pos]
        .astype(np.int8)
    )

    # Last row is target feature row.
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test = X.iloc[[-1]]

    # --------------------------------------------------------
    # Select once for HOT
    # --------------------------------------------------------

    hot_selected = select_features_once(
        X_train,
        y_train,
        cfg["hot_features"],
        "hot"
    )

    # --------------------------------------------------------
    # Select once for DEAD
    # --------------------------------------------------------

    dead_selected = select_features_once(
        X_train,
        y_train,
        cfg["dead_features"],
        "dead"
    )

    hot = hot_system(
        X_train,
        y_train,
        X_test,
        cfg,
        hot_selected
    )

    dead = dead_system(
        X_train,
        y_train,
        X_test,
        cfg,
        dead_selected
    )

    return {
        "hot": hot,
        "dead": dead
    }


# ============================================================
# 19. DISPLAY HOT
# ============================================================

def display_hot_card(
    result
):

    data = result["hot"]["hot"]

    nums = " - ".join(
        str(n)
        for n, _ in data
    )

    probs_html = "".join(
        f'<span class="prob-pill">'
        f'{n}: {p*100:.1f}%'
        f'</span>'
        for n, p in data
    )

    top1 = (
        result["hot"]
        ["top1_probability"]
        * 100
    )

    gap = (
        result["hot"]
        ["top_gap"]
        * 100
    )

    top3 = (
        result["hot"]
        ["top3"]
        * 100
    )

    concentration = (
        result["hot"]
        ["concentration"]
        * 100
    )

    html = f"""

    <div class="hot-card">

        <div class="position-title">
            🔥 HOT TOP-3
        </div>

        <div class="hot-number">
            {nums}
        </div>

        <div class="prob-text">

            โอกาสเชิงโมเดล<br>

            {probs_html}

        </div>

        <div class="confidence">

            🎯 Top-1:
            {top1:.1f}%

            &nbsp; | &nbsp;

            📌 Top Gap:
            {gap:.1f}%

            <br>

            🔥 Top-3 Mass:
            {top3:.1f}%

            &nbsp; | &nbsp;

            📊 Concentration:
            {concentration:.1f}%

        </div>

    </div>

    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 20. DISPLAY DEAD
# ============================================================

def display_dead_card(
    result
):

    data = result["dead"]["dead"]

    nums = " - ".join(
        str(n)
        for n, _ in data[:5]
    )

    probs_html = "".join(
        f'<span class="prob-pill">'
        f'{n}'
        f'</span>'
        for n, _ in data
    )

    html = f"""

    <div class="dead-card">

        <div class="position-title">
            🛑 COLD / DEAD TOP-7
        </div>

        <div class="dead-number">
            {nums}
        </div>

        <div class="prob-text">

            กลุ่มเลขที่ระบบประเมินว่า
            <b>โอกาสต่ำกว่า</b><br>

            {probs_html}

        </div>

        <div class="confidence">

            🛑 Dead Group Score:
            {result["dead"]["top7"]*100:.1f}%

        </div>

    </div>

    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 21. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        '<div class="main-title">'
        '🤖 LOTTO AI PRO V9.1 TURBO'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        '⚡ Adaptive Stability + Fast AI '
        '| 🔥 HOT TOP-3 '
        '| 🛑 COLD/DEAD TOP-7'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    lottery = c1.selectbox(
        "🏷️ เลือกประเภทหวย",
        list(
            LOTTERY_SOURCES.keys()
        )
    )

    selected_day = c2.selectbox(
        "📅 วันเป้าหมาย",
        ["อัตโนมัติ"] + DOW_NAMES
    )

    if not st.button(
        "⚡ เริ่มวิเคราะห์ระบบ V9.1",
        type="primary",
        use_container_width=True
    ):

        return

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    with st.spinner(
        "📥 กำลังดึงข้อมูลสถิติล่าสุด..."
    ):

        try:

            df = fetch_lottery_data(
                LOTTERY_SOURCES[lottery]
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

            return

    if len(df) < 50:

        st.error(
            f"❌ มีข้อมูล {len(df)} งวด "
            f"(ต้องการอย่างน้อย 50 งวด)"
        )

        return

    # --------------------------------------------------------
    # DETECT TYPE
    # --------------------------------------------------------

    thai_6d = (
        lottery == "หวยไทย"
        and
        is_thai_6d(df)
    )

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    # --------------------------------------------------------
    # TARGET DATE
    # --------------------------------------------------------

    last_date = pd.Timestamp(
        df["Date"].iloc[-1]
    )

    if selected_day == "อัตโนมัติ":

        days_ahead = 7

    else:

        days_ahead = (
            DOW_NAMES.index(
                selected_day
            )
            -
            last_date.dayofweek
        ) % 7

        if days_ahead == 0:
            days_ahead = 7

    target_date = (
        last_date
        +
        timedelta(
            days=days_ahead
        )
    )

    # --------------------------------------------------------
    # TARGET ROW
    #
    # Important:
    # The target row contains only placeholder result.
    # All lag/rolling features use SHIFT(1), therefore
    # the target feature row only receives information
    # from historical draws.
    # --------------------------------------------------------

    dummy = {
        "Date": target_date,
        "Result_3D": "000",
        "Result_2D": "00"
    }

    if thai_6d:

        dummy[
            "Result_6D"
        ] = "000000"

    ext = pd.concat(
        [
            df,
            pd.DataFrame([dummy])
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    with st.spinner(
        "⚙️ กำลังสร้าง Leakage-Safe Features..."
    ):

        feat = build_features(
            ext,
            thai_6d
        )

        features = get_features(
            thai_6d
        )

        cfg = get_adaptive_config(
            len(df)
        )

    # --------------------------------------------------------
    # BACKTEST + FINAL
    # --------------------------------------------------------

    final = {}
    hot_backtest = {}
    dead_backtest = {}

    progress = st.progress(0)

    status_text = st.empty()

    historical_feat = feat.iloc[:-1]

    for i, pos in enumerate(
        positions
    ):

        status_text.caption(
            f"🧠 วิเคราะห์ "
            f"{POSITION_LABELS[pos]}"
        )

        # -----------------------------
        # BACKTEST
        # -----------------------------

        hot_backtest[pos] = (
            walk_forward_system(
                historical_feat,
                pos,
                features,
                cfg,
                "hot"
            )
        )

        dead_backtest[pos] = (
            walk_forward_system(
                historical_feat,
                pos,
                features,
                cfg,
                "dead"
            )
        )

        # -----------------------------
        # FINAL
        # -----------------------------

        final[pos] = final_prediction(
            feat,
            pos,
            features,
            cfg
        )

        progress.progress(
            int(
                ((i + 1) / len(positions))
                * 100
            )
        )

    progress.empty()
    status_text.empty()

    # ========================================================
    # STATUS
    # ========================================================

    st.markdown(
        f"""
        <div class="status-card">

        ✅ วิเคราะห์สำเร็จ:
        {len(df):,} งวด

        <br>

        🎯 เป้าหมาย:
        {target_date.strftime("%d/%m/%Y")}

        ({lottery})

        <br>

        ⚙️ ExtraTrees:
        {cfg["trees"]} Trees

        &nbsp; | &nbsp;

        🌲 HGB:
        Adaptive

        &nbsp; | &nbsp;

        🧠 Hot Features:
        {cfg["hot_features"]}

        &nbsp; | &nbsp;

        🛑 Dead Features:
        {cfg["dead_features"]}

        </div>

        <br>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "### 🏆 สรุปเลขฟันธง V9.1"
    )

    summary_data = []

    for pos in positions:

        hot = final[pos]["hot"]["hot"]

        dead = final[pos]["dead"]["dead"]

        summary_data.append(
            {
                "ตำแหน่ง":
                    POSITION_LABELS[pos],

                "🔥 HOT #1":
                    str(hot[0][0]),

                "โอกาสเชิงโมเดล":
                    f"{hot[0][1]*100:.1f}%",

                "🔥 HOT #2/#3":
                    f"{hot[1][0]}, "
                    f"{hot[2][0]}",

                "📌 Top Gap":
                    f"{final[pos]['hot']['top_gap']*100:.1f}%",

                "🛑 COLD/DEAD":
                    ", ".join(
                        str(x[0])
                        for x in dead[:7]
                    )
            }
        )

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ========================================================
    # TABS
    # ========================================================

    t1, t2 = st.tabs(
        [
            "🎯 เจาะลึกรายหลัก",
            "📊 Walk-Forward Backtest"
        ]
    )

    # ========================================================
    # TAB 1
    # ========================================================

    with t1:

        st.markdown(
            "ระบบแยก **HOT TOP-3** "
            "และ **COLD/DEAD TOP-7** "
            "รายหลัก"
        )

        for pos in positions:

            st.markdown(
                f"""
                <div class="position-title">
                    {POSITION_LABELS[pos]}
                </div>
                """,
                unsafe_allow_html=True
            )

            col_hot, col_dead = st.columns(2)

            with col_hot:

                display_hot_card(
                    final[pos]
                )

            with col_dead:

                display_dead_card(
                    final[pos]
                )

    # ========================================================
    # TAB 2
    # ========================================================

    with t2:

        st.markdown(
            "### 📊 Walk-Forward Backtest"
        )

        st.caption(
            "ทดสอบข้อมูลย้อนหลังโดยไม่ใช้ผลของงวดอนาคต"
        )

        for pos in positions:

            h_result = (
                hot_backtest[pos]
            )

            d_result = (
                dead_backtest[pos]
            )

            h_sc = (
                h_result
                .get(
                    "scores",
                    {}
                )
            )

            d_sc = (
                d_result
                .get(
                    "scores",
                    {}
                )
            )

            if not h_sc:
                continue

            with st.expander(
                f"📊 {POSITION_LABELS[pos]}"
            ):

                rows = [
                    {
                        "Metric":
                            "🔥 Top-1",
                        "ผลย้อนหลัง":
                            f"{h_sc.get('top1', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "🔥 Top-3",
                        "ผลย้อนหลัง":
                            f"{h_sc.get('top3', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "🔥 Top-5",
                        "ผลย้อนหลัง":
                            f"{h_sc.get('top5', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "🛑 Dead Group Hit-5",
                        "ผลย้อนหลัง":
                            f"{d_sc.get('dead5', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "🛑 Dead Group Hit-7",
                        "ผลย้อนหลัง":
                            f"{d_sc.get('dead7', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "📊 Hot Stability",
                        "ผลย้อนหลัง":
                            f"{h_result.get('stability', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "📊 Dead Stability",
                        "ผลย้อนหลัง":
                            f"{d_result.get('stability', 0)*100:.1f}%"
                    },
                    {
                        "Metric":
                            "🧪 Backtest Points",
                        "ผลย้อนหลัง":
                            str(
                                h_result.get(
                                    "tests",
                                    0
                                )
                            )
                    }
                ]

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True
                )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### ⚙️ V9.1 System Information"
    )

    info = pd.DataFrame(
        [
            {
                "รายการ":
                    "Dataset",
                "ค่า":
                    f"{len(df):,} งวด"
            },
            {
                "รายการ":
                    "Minimum Train",
                "ค่า":
                    str(cfg["min_train"])
            },
            {
                "รายการ":
                    "Trees",
                "ค่า":
                    str(cfg["trees"])
            },
            {
                "รายการ":
                    "Depth",
                "ค่า":
                    str(cfg["depth"])
            },
            {
                "รายการ":
                    "Leaf",
                "ค่า":
                    str(cfg["leaf"])
            },
            {
                "รายการ":
                    "Recent Decay",
                "ค่า":
                    str(cfg["recent_decay"])
            },
            {
                "รายการ":
                    "Feature Refresh",
                "ค่า":
                    f"ทุก {cfg['refresh_every']} checkpoints"
            },
            {
                "รายการ":
                    "Models",
                "ค่า":
                    "ExtraTrees + HistGradientBoosting"
            },
            {
                "รายการ":
                    "Class Weight",
                "ค่า":
                    "None"
            },
            {
                "รายการ":
                    "Prediction",
                "ค่า":
                    "HOT TOP-3 + COLD/DEAD TOP-7"
            }
        ]
    )

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "⚠️ Probability เป็นค่าประเมินจากโมเดล "
        "ไม่ใช่ความน่าจะเป็นที่รับประกันผลจริง"
    )


# ============================================================
# 22. RUN
# ============================================================

if __name__ == "__main__":
    main()
