# ============================================================
# 🤖 LOTTO AI PRO V8.2 TURBO
# ============================================================
# STRICT WALK-FORWARD
# NO PERSISTENT MEMORY
# LEAKAGE SAFE
# THAI LOTTERY 6D + 2D SUPPORTED
# FAST FEATURE ENGINEERING
# FAST WALK-FORWARD
# SINGLE FEATURE SELECTION / POSITION
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
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lotto AI V8.2 TURBO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

DOW_NAMES = [
    "จันทร์", "อังคาร", "พุธ",
    "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"
]

MODEL_NAMES = [
    "ExtraTrees",
    "RandomForest",
    "HistGradientBoosting"
]

# ============================================================
# 2. POSITIONS
# ============================================================

# หวยไทย:
# 6 หลัก = H1 H2 H3 H4 H5 H6
# 2 หลัก = T2 O2
#
# หวยอื่น:
# 3 หลัก = H T O
# 2 หลัก = T2 O2

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
    "H1": "💯 หลักแสน 6 ตัว",
    "H2": "🔢 หลักหมื่น 6 ตัว",
    "H3": "🔢 หลักพัน 6 ตัว",
    "H4": "🔢 หลักร้อย 6 ตัว",
    "H5": "🔟 หลักสิบ 6 ตัว",
    "H6": "1️⃣ หลักหน่วย 6 ตัว",
    "H": "💯 หลักร้อย 3 ตัวบน",
    "T": "🔟 หลักสิบ 3 ตัวบน",
    "O": "1️⃣ หลักหน่วย 3 ตัวบน",
    "T2": "🔽 หลักสิบ 2 ตัวล่าง",
    "O2": "⬇️ หลักหน่วย 2 ตัวล่าง",
}

# ============================================================
# 3. CSS
# ============================================================

def inject_css():

    st.markdown("""
    <style>

    .stApp {
        background:#f8fafc;
    }

    .main-title {
        text-align:center;
        font-size:2.25rem;
        font-weight:900;
    }

    .subtitle {
        text-align:center;
        color:#64748b;
        font-size:.9rem;
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
    """, unsafe_allow_html=True)


# ============================================================
# 4. DATE
# ============================================================

THAI_MONTHS = {
    "มกราคม":1, "กุมภาพันธ์":2, "มีนาคม":3,
    "เมษายน":4, "พฤษภาคม":5, "มิถุนายน":6,
    "กรกฎาคม":7, "สิงหาคม":8, "กันยายน":9,
    "ตุลาคม":10, "พฤศจิกายน":11, "ธันวาคม":12,

    "ม.ค.":1, "ก.พ.":2, "มี.ค.":3,
    "เม.ย.":4, "พ.ค.":5, "มิ.ย.":6,
    "ก.ค.":7, "ส.ค.":8, "ก.ย.":9,
    "ต.ค.":10, "พ.ย.":11, "ธ.ค.":12
}


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
            except:
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
        except:
            pass

    return None


# ============================================================
# 5. FAST SCRAPER
# ============================================================

class ScrapingError(Exception):
    pass


@st.cache_data(ttl=600, show_spinner=False)
def fetch_lottery_data(url):

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Linux; Android 10) "
            "AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36",
        "Accept-Language":
            "th-TH,th;q=0.9,en;q=0.8"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=12
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

            cells = [
                c.get_text(" ", strip=True)
                for c in row.find_all(["td", "th"])
            ]

            text = " ".join(cells)

            if not text:
                continue

            date = normalize_date(text)

            if date is None:
                continue

            # รองรับ 6 หลัก + 2 หลัก
            six = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
            three = re.findall(r"(?<!\d)\d{3}(?!\d)", text)
            two = re.findall(r"(?<!\d)\d{2}(?!\d)", text)

            if six and two:

                rows.append({
                    "Date": date,
                    "Result_6D": six[0],
                    "Result_3D": six[0][-3:],
                    "Result_2D": two[-1]
                })

            elif three and two:

                rows.append({
                    "Date": date,
                    "Result_6D": None,
                    "Result_3D": three[0],
                    "Result_2D": two[-1]
                })

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

                two = re.findall(
                    r"(?<!\d)\d{2}(?!\d)",
                    line
                )

                three = re.findall(
                    r"(?<!\d)\d{3}(?!\d)",
                    line
                )

                if six and two:

                    rows.append({
                        "Date": current_date,
                        "Result_6D": six[0],
                        "Result_3D": six[0][-3:],
                        "Result_2D": two[-1]
                    })

                elif three and two:

                    rows.append({
                        "Date": current_date,
                        "Result_6D": None,
                        "Result_3D": three[0],
                        "Result_2D": two[-1]
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
            .str.extract(r"(\d{3})")[0]
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(r"(\d{2})")[0]
            .str.zfill(2)
        )

        if "Result_6D" in df:

            df["Result_6D"] = (
                df["Result_6D"]
                .astype(str)
                .str.extract(r"(\d{6})")[0]
            )

        df = (
            df.dropna(subset=["Date"])
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        raise ScrapingError(
            f"โหลดข้อมูลไม่สำเร็จ: {exc}"
        )


# ============================================================
# 6. DETECT LOTTERY FORMAT
# ============================================================

def is_thai_6d(df):

    return (
        "Result_6D" in df.columns
        and
        df["Result_6D"].notna().sum() >= 10
    )


# ============================================================
# 7. FAST FEATURE ENGINEERING
# ============================================================

def build_features(df, thai_6d=False):

    w = df.copy()

    # --------------------------------------------------------
    # CREATE POSITION DIGITS
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
                six.str[i].astype(int)
            )

    else:

        three = (
            w["Result_3D"]
            .astype(str)
            .str.zfill(3)
        )

        w["H"] = three.str[0].astype(int)
        w["T"] = three.str[1].astype(int)
        w["O"] = three.str[2].astype(int)

    w["T2"] = (
        w["Result_2D"]
        .astype(str)
        .str.zfill(2)
        .str[0]
        .astype(int)
    )

    w["O2"] = (
        w["Result_2D"]
        .astype(str)
        .str.zfill(2)
        .str[1]
        .astype(int)
    )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    dt = w["Date"].dt

    w["DOW"] = dt.dayofweek
    w["DAY"] = dt.day
    w["MONTH"] = dt.month
    w["DAY_OF_YEAR"] = dt.dayofyear

    w["DOW_SIN"] = np.sin(
        2 * np.pi * w["DOW"] / 7
    )

    w["DOW_COS"] = np.cos(
        2 * np.pi * w["DOW"] / 7
    )

    w["MONTH_SIN"] = np.sin(
        2 * np.pi * w["MONTH"] / 12
    )

    w["MONTH_COS"] = np.cos(
        2 * np.pi * w["MONTH"] / 12
    )

    # --------------------------------------------------------
    # POSITION LIST
    # --------------------------------------------------------

    positions = (
        THAI_POSITIONS
        if thai_6d
        else NORMAL_POSITIONS
    )

    # --------------------------------------------------------
    # FAST FEATURES
    # --------------------------------------------------------

    for pos in positions:

        s = w[pos]
        p = s.shift(1)

        # LAGS
        for lag in (1, 2, 3, 5):

            w[f"{pos}_L{lag}"] = s.shift(lag)

        # rolling
        for window in (5, 10, 20):

            r = p.rolling(
                window,
                min_periods=2
            )

            w[f"{pos}_M{window}"] = r.mean()
            w[f"{pos}_S{window}"] = r.std()

            # important digits only
            for digit in (0, 2, 5, 7):

                w[
                    f"{pos}_F{window}_{digit}"
                ] = (
                    (p == digit)
                    .astype(float)
                    .rolling(
                        window,
                        min_periods=2
                    )
                    .mean()
                )

        # simple momentum
        w[f"{pos}_D1"] = (
            s.shift(1) - s.shift(2)
        )

        w[f"{pos}_D2"] = (
            s.shift(2) - s.shift(3)
        )

        w[f"{pos}_ODD"] = (
            p % 2
        )

        w[f"{pos}_HIGH"] = (
            p >= 5
        ).astype(float)

        w[f"{pos}_MOD3"] = p % 3
        w[f"{pos}_MOD5"] = p % 5

        # cyclic digit
        w[f"{pos}_SIN"] = np.sin(
            2 * np.pi * p / 10
        )

        w[f"{pos}_COS"] = np.cos(
            2 * np.pi * p / 10
        )

        # EWMA
        w[f"{pos}_EWMA3"] = (
            p.ewm(
                span=3,
                adjust=False
            ).mean()
        )

        w[f"{pos}_EWMA7"] = (
            p.ewm(
                span=7,
                adjust=False
            ).mean()
        )

        # repeat
        w[f"{pos}_REPEAT"] = (
            p == s.shift(2)
        ).astype(float)

    # --------------------------------------------------------
    # PREVIOUS DRAW AGGREGATES
    # --------------------------------------------------------

    if thai_6d:

        base = w[
            ["H1", "H2", "H3",
             "H4", "H5", "H6"]
        ].shift(1)

    else:

        base = w[
            ["H", "T", "O"]
        ].shift(1)

    w["PREV_SUM"] = base.sum(axis=1)

    w["PREV_RANGE"] = (
        base.max(axis=1)
        -
        base.min(axis=1)
    )

    w["PREV_MEAN"] = base.mean(axis=1)

    w["PREV_ODD"] = (
        base % 2
    ).sum(axis=1)

    w["PREV_HIGH"] = (
        base >= 5
    ).sum(axis=1)

    w["PREV_UNIQUE"] = (
        base.nunique(axis=1)
    )

    return w.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 8. FEATURE GENERATOR
# ============================================================

def get_features(thai_6d):

    base = [
        "DOW",
        "DAY",
        "MONTH",
        "DAY_OF_YEAR",
        "DOW_SIN",
        "DOW_COS",
        "MONTH_SIN",
        "MONTH_COS",
        "PREV_SUM",
        "PREV_RANGE",
        "PREV_MEAN",
        "PREV_ODD",
        "PREV_HIGH",
        "PREV_UNIQUE"
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
                f"{pos}_M5",
                f"{pos}_M10",
                f"{pos}_M20",
                f"{pos}_S5",
                f"{pos}_S10",
                f"{pos}_S20",
                f"{pos}_D1",
                f"{pos}_D2",
                f"{pos}_ODD",
                f"{pos}_HIGH",
                f"{pos}_MOD3",
                f"{pos}_MOD5",
                f"{pos}_SIN",
                f"{pos}_COS",
                f"{pos}_EWMA3",
                f"{pos}_EWMA7",
                f"{pos}_REPEAT"
            ]
        )

        for window in (5, 10, 20):

            for digit in (0, 2, 5, 7):

                base.append(
                    f"{pos}_F{window}_{digit}"
                )

    return list(dict.fromkeys(base))


# ============================================================
# 9. FAST CONFIG
# ============================================================

def get_adaptive_config(n):

    # เน้นเร็วมากขึ้น

    if n >= 700:

        return {
            "min_train": 120,
            "trees": 80,
            "depth": 8,
            "leaf": 2,
            "selected_features": 24,
            "backtest_points": 20,
            "recent_decay": 0.985
        }

    if n >= 400:

        return {
            "min_train": 100,
            "trees": 65,
            "depth": 7,
            "leaf": 2,
            "selected_features": 22,
            "backtest_points": 16,
            "recent_decay": 0.98
        }

    if n >= 200:

        return {
            "min_train": 80,
            "trees": 55,
            "depth": 6,
            "leaf": 2,
            "selected_features": 20,
            "backtest_points": 12,
            "recent_decay": 0.975
        }

    return {
        "min_train": 50,
        "trees": 40,
        "depth": 5,
        "leaf": 2,
        "selected_features": 18,
        "backtest_points": 8,
        "recent_decay": 0.97
    }


# ============================================================
# 10. MODELS
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
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    if name == "RandomForest":

        return RandomForestClassifier(
            n_estimators=t,
            max_depth=d,
            min_samples_leaf=l,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    return HistGradientBoostingClassifier(
        max_iter=max(35, int(t * .65)),
        max_leaf_nodes=20,
        learning_rate=.04,
        min_samples_leaf=l,
        l2_regularization=2,
        random_state=42
    )


# ============================================================
# 11. FEATURE SELECTION
# ============================================================

def select_features_once(
    X,
    y,
    max_features
):

    cols = list(X.columns)

    if len(cols) <= max_features:
        return cols

    valid = [
        c for c in cols
        if X[c].nunique(dropna=False) > 1
    ]

    if len(valid) <= max_features:
        return valid

    imp = SimpleImputer(
        strategy="median"
    )

    Xi = imp.fit_transform(
        X[valid]
    )

    selector = ExtraTreesClassifier(
        n_estimators=25,
        max_depth=5,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=123
    )

    selector.fit(Xi, y)

    order = np.argsort(
        selector.feature_importances_
    )[::-1]

    return [
        valid[i]
        for i in order[:max_features]
    ]


# ============================================================
# 12. PROBABILITY
# ============================================================

def normalize_probability(p):

    p = np.asarray(
        p,
        dtype=float
    )

    p = np.clip(
        p,
        1e-9,
        None
    )

    s = p.sum()

    if s <= 0:
        return np.ones(10) / 10

    return p / s


# ============================================================
# 13. TRAIN
# ============================================================

def train_models(
    X_train,
    y_train,
    X_test,
    cfg,
    selected_features
):

    imp = SimpleImputer(
        strategy="median"
    )

    A = imp.fit_transform(
        X_train[selected_features]
    )

    B = imp.transform(
        X_test[selected_features]
    )

    predictions = []

    for name in MODEL_NAMES:

        try:

            model = create_model(
                name,
                cfg
            )

            model.fit(
                A,
                y_train
            )

            raw = model.predict_proba(B)[0]

            out = np.zeros(10)

            for cls, prob in zip(
                model.classes_,
                raw
            ):

                cls = int(cls)

                if 0 <= cls <= 9:
                    out[cls] = prob

            predictions.append(
                normalize_probability(out)
            )

        except Exception:
            continue

    if not predictions:
        return np.ones(10) / 10

    return normalize_probability(
        np.mean(
            predictions,
            axis=0
        )
    )


# ============================================================
# 14. FAST WALK-FORWARD
# ============================================================

def strict_walk_forward(
    df_feat,
    pos,
    features,
    cfg
):

    X = df_feat[
        features
    ].astype(float)

    y = df_feat[
        pos
    ].astype(int)

    n = len(df_feat)

    start = cfg["min_train"]

    if n <= start + 2:
        return {
            "tests": 0,
            "scores": {}
        }

    available = n - start

    tests = min(
        cfg["backtest_points"],
        available
    )

    # --------------------------------------------------------
    # จุดทดสอบกระจายทั่วช่วงท้าย
    # ไม่ train ทุกงวด
    # --------------------------------------------------------

    test_indices = np.linspace(
        start,
        n - 1,
        tests,
        dtype=int
    )

    test_indices = np.unique(
        test_indices
    )

    # --------------------------------------------------------
    # เลือก Feature ครั้งเดียวจาก training block
    # --------------------------------------------------------

    selection_end = test_indices[0]

    if y.iloc[:selection_end].nunique() < 2:

        return {
            "tests": 0,
            "scores": {}
        }

    selected = select_features_once(
        X.iloc[:selection_end],
        y.iloc[:selection_end],
        cfg["selected_features"]
    )

    records = []

    for idx in test_indices:

        if idx < cfg["min_train"]:
            continue

        if y.iloc[:idx].nunique() < 2:
            continue

        probs = train_models(
            X.iloc[:idx],
            y.iloc[:idx],
            X.iloc[[idx]],
            cfg,
            selected
        )

        actual = int(
            y.iloc[idx]
        )

        ranking = np.argsort(
            probs
        )[::-1]

        records.append({
            "top1":
                int(actual == ranking[0]),

            "top3":
                int(actual in ranking[:3]),

            "top5":
                int(actual in ranking[:5]),

            "dead7":
                int(actual in np.argsort(probs)[:7]),

            "logloss":
                -np.log(
                    max(
                        probs[actual],
                        1e-9
                    )
                )
        })

    if not records:

        return {
            "tests": 0,
            "scores": {}
        }

    h = pd.DataFrame(
        records
    )

    decay = cfg[
        "recent_decay"
    ] ** (
        len(h)
        -
        np.arange(len(h))
        - 1
    )

    decay /= decay.sum()

    scores = {
        "top1":
            float(
                np.sum(
                    h.top1 * decay
                )
            ),

        "top3":
            float(
                np.sum(
                    h.top3 * decay
                )
            ),

        "top5":
            float(
                np.sum(
                    h.top5 * decay
                )
            ),

        "dead7":
            float(
                np.sum(
                    h.dead7 * decay
                )
            ),

        "logloss":
            float(
                np.sum(
                    h.logloss * decay
                )
            )
    }

    scores["score"] = (
        .35 * scores["top1"]
        +
        .30 * scores["top3"]
        +
        .20 * scores["top5"]
        +
        .10 * (1 / (1 + scores["logloss"]))
        +
        .05 * (1 - scores["dead7"])
    )

    return {
        "tests": len(h),
        "scores": scores
    }


# ============================================================
# 15. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_feat,
    pos,
    features,
    cfg
):

    X = df_feat[
        features
    ].astype(float)

    y = df_feat[
        pos
    ].astype(int)

    # --------------------------------------------------------
    # แถวสุดท้ายคือ target dummy
    # training ใช้เฉพาะข้อมูลจริง
    # --------------------------------------------------------

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

    order = np.argsort(
        probs
    )[::-1]

    hot = [
        (
            int(i),
            float(probs[i])
        )
        for i in order[:5]
    ]

    dead = [
        (
            int(i),
            float(probs[i])
        )
        for i in np.argsort(probs)[:7]
    ]

    confidence = (
        probs[order[0]]
        -
        probs[order[1]]
    )

    return {
        "probabilities": probs,
        "hot": hot,
        "dead": dead,
        "confidence":
            float(confidence),
        "top3":
            float(
                probs[order[:3]].sum()
            ),
        "selected_features":
            selected
    }


# ============================================================
# 16. DISPLAY
# ============================================================

def display_card(
    pos,
    result,
    hot=True
):

    data = (
        result["hot"]
        if hot
        else result["dead"]
    )

    style = (
        "hot"
        if hot
        else "dead"
    )

    nums = " - ".join(
        str(n)
        for n, _ in data
    )

    probs = " | ".join(
        f"{n}: {p*100:.1f}%"
        for n, p in data
    )

    html = f"""
    <div class="{style}-card">

        <div class="position-title">
            {POSITION_LABELS[pos]}
        </div>

        <div class="{style}-number">
            {nums}
        </div>

        <div class="prob-text">
            AI Probability: {probs}
        </div>
    """

    if hot:

        html += f"""
        <div class="confidence">
            📌 Top-1 Gap:
            {result["confidence"]*100:.1f}%
            &nbsp;|&nbsp;
            Top-3:
            {result["top3"]*100:.1f}%
        </div>
        """

    html += """
        <div class="model-badge">
            🤖 AI Ensemble
            ExtraTrees + RandomForest +
            HistGradientBoosting
        </div>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        """
        <div class="main-title">
            🤖 LOTTO AI PRO V8.2 TURBO
        </div>

        <div class="subtitle">
            STRICT WALK-FORWARD •
            LEAKAGE SAFE •
            NO PERSISTENT MEMORY •
            ⚡ TURBO ENGINE
        </div>
        """,
        unsafe_allow_html=True
    )

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
        "🚀 เริ่มวิเคราะห์ V8.2 TURBO",
        type="primary",
        use_container_width=True
    ):
        return

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    with st.spinner(
        "📥 โหลดข้อมูล..."
    ):

        try:

            df = fetch_lottery_data(
                LOTTERY_SOURCES[
                    lottery
                ]
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

            return

    if len(df) < 50:

        st.error(
            f"❌ มีข้อมูล {len(df)} งวด "
            "ต้องมีอย่างน้อย 50 งวด"
        )

        return

    thai_6d = (
        lottery == "หวยไทย"
        and is_thai_6d(df)
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

        if len(df) >= 2:

            gap = (
                df["Date"].iloc[-1]
                -
                df["Date"].iloc[-2]
            ).days

            days_ahead = max(
                int(gap),
                1
            )

        else:

            days_ahead = 7

    else:

        target_dow = DOW_NAMES.index(
            selected_day
        )

        days_ahead = (
            target_dow
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
    # ADD DUMMY TARGET
    # --------------------------------------------------------

    dummy = {
        "Date": target_date,
        "Result_3D": "000",
        "Result_2D": "00"
    }

    if thai_6d:
        dummy["Result_6D"] = "000000"

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
        "⚡ สร้าง Fast Causal Features..."
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

    st.info(
        f"""
        ⚡ V8.2 TURBO |
        ข้อมูล {len(df):,} งวด |
        {"หวยไทย 6 หลัก + 2 หลัก" if thai_6d else "3 หลัก + 2 หลัก"} |
        Features {len(features)} |
        Selected ≤ {cfg["selected_features"]} |
        Trees {cfg["trees"]} |
        WF Tests {cfg["backtest_points"]}
        """
    )

    # --------------------------------------------------------
    # BACKTEST + FINAL
    # --------------------------------------------------------

    backtest = {}
    final = {}

    progress = st.progress(0)
    status = st.empty()

    total = len(positions)

    for i, pos in enumerate(positions):

        status.caption(
            f"⚡ TURBO AI: "
            f"{POSITION_LABELS[pos]}"
        )

        backtest[pos] = strict_walk_forward(
            feat.iloc[:-1],
            pos,
            features,
            cfg
        )

        final[pos] = final_prediction(
            feat,
            pos,
            features,
            cfg
        )

        progress.progress(
            int(
                (i + 1)
                /
                total
                *
                100
            )
        )

    progress.empty()
    status.empty()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    tests = [
        x["tests"]
        for x in backtest.values()
    ]

    st.markdown(
        f"""
        <div class="status-card">

        🤖 <b>LOTTO AI PRO V8.2 TURBO</b><br>

        📊 ข้อมูล:
        {len(df):,} งวด<br>

        📅 เป้าหมาย:
        {target_date.strftime("%d/%m/%Y")}<br>

        🎯 Mode:
        {"6D + 2D" if thai_6d else "3D + 2D"}<br>

        🧠 Selected Features:
        ≤{cfg["selected_features"]}<br>

        🌳 Trees:
        {cfg["trees"]}<br>

        ⚡ Walk-Forward:
        {min(tests)}–{max(tests)} จุด

        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    t1, t2, t3 = st.tabs(
        [
            "🎯 เลขเด่น AI",
            "🛑 เลขดับ 7",
            "📊 Backtest"
        ]
    )

    with t1:

        for pos in positions:

            display_card(
                pos,
                final[pos],
                True
            )

    with t2:

        for pos in positions:

            display_card(
                pos,
                final[pos],
                False
            )

    with t3:

        for pos in positions:

            st.markdown(
                f"### {POSITION_LABELS[pos]}"
            )

            score = (
                backtest[pos]
                .get("scores", {})
            )

            if not score:

                st.warning(
                    "ไม่มี Backtest"
                )

                continue

            st.dataframe(
                pd.DataFrame([
                    {
                        "Metric":
                            "Top-1",
                        "AI":
                            f"{score['top1']*100:.1f}%",
                        "Random":
                            "10%",
                        "Edge":
                            f"{(score['top1']-.10)*100:+.1f}%"
                    },
                    {
                        "Metric":
                            "Top-3",
                        "AI":
                            f"{score['top3']*100:.1f}%",
                        "Random":
                            "30%",
                        "Edge":
                            f"{(score['top3']-.30)*100:+.1f}%"
                    },
                    {
                        "Metric":
                            "Top-5",
                        "AI":
                            f"{score['top5']*100:.1f}%",
                        "Random":
                            "50%",
                        "Edge":
                            f"{(score['top5']-.50)*100:+.1f}%"
                    },
                    {
                        "Metric":
                            "Dead-7",
                        "AI":
                            f"{score['dead7']*100:.1f}%",
                        "Random":
                            "70%",
                        "Edge":
                            f"{(score['dead7']-.70)*100:+.1f}%"
                    },
                    {
                        "Metric":
                            "LogLoss",
                        "AI":
                            f"{score['logloss']:.3f}",
                        "Random":
                            "-",
                        "Edge":
                            "-"
                    }
                ]),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    with st.expander(
        "🎯 สรุปเลขเด่นทั้งหมด"
    ):

        data = []

        for pos in positions:

            sc = (
                backtest[pos]
                .get("scores", {})
            )

            data.append({
                "ตำแหน่ง":
                    POSITION_LABELS[pos],

                "Top-1":
                    final[pos]["hot"][0][0],

                "Probability":
                    f'{final[pos]["hot"][0][1]*100:.1f}%',

                "WF Top-1":
                    f'{sc.get("top1", 0)*100:.1f}%',

                "WF Top-3":
                    f'{sc.get("top3", 0)*100:.1f}%',

                "Confidence":
                    f'{final[pos]["confidence"]*100:.1f}%'
            })

        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":
    main()
