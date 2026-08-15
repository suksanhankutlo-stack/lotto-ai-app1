# ============================================================
# 🤖 LOTTO AI PRO V8.2 TURBO MAX
# ============================================================
# AI ONLY
# STRICT WALK-FORWARD
# NO PERSISTENT MEMORY
# LEAKAGE SAFE
# THAI LOTTERY 6 DIGIT + 2 DIGIT SUPPORT
# FAST / MOBILE OPTIMIZED
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
    page_title="Lotto AI PRO V8.2 TURBO MAX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. SOURCES
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


POSITIONS = ["H", "T", "O", "T2", "O2"]

POSITION_LABELS = {
    "H": "💯 หลักร้อย 3 ตัวบน",
    "T": "🔟 หลักสิบ 3 ตัวบน",
    "O": "1️⃣ หลักหน่วย 3 ตัวบน",
    "T2": "🔽 หลักสิบ 2 ตัวล่าง",
    "O2": "⬇️ หลักหน่วย 2 ตัวล่าง"
}

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
    "RandomForest",
    "HistGradientBoosting"
]


# ============================================================
# 3. CSS
# ============================================================

def inject_css():

    st.markdown("""
    <style>

    .stApp {
        background: #f8fafc;
    }

    .main-title {
        text-align: center;
        font-size: 2.15rem;
        font-weight: 900;
        margin-bottom: 2px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: .9rem;
        margin-bottom: 15px;
    }

    .status-card {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 13px;
        text-align: center;
        color: #1e40af;
        font-weight: 700;
        line-height: 1.7;
    }

    .hot-card {
        background: #f0fdf4;
        border-left: 7px solid #16a34a;
        border-radius: 14px;
        padding: 14px;
        margin: 9px 0;
    }

    .dead-card {
        background: #fef2f2;
        border-left: 7px solid #dc2626;
        border-radius: 14px;
        padding: 14px;
        margin: 9px 0;
    }

    .position-title {
        font-size: 1.1rem;
        font-weight: 900;
        color: #334155;
    }

    .hot-number,
    .dead-number {
        font-size: 2.05rem;
        font-weight: 900;
        letter-spacing: 3px;
        text-align: center;
    }

    .hot-number {
        color: #16a34a;
    }

    .dead-number {
        color: #dc2626;
    }

    .prob-text {
        text-align: center;
        color: #64748b;
        font-size: .8rem;
        margin-top: 4px;
    }

    .model-badge {
        text-align: center;
        background: white;
        border-radius: 9px;
        padding: 6px;
        margin-top: 7px;
        color: #475569;
        font-weight: 700;
    }

    .confidence {
        text-align: center;
        font-size: .85rem;
        font-weight: 800;
        margin-top: 5px;
        color: #334155;
    }

    div.stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 800;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 4. DATE
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


def normalize_date(value):

    if not value:
        return None

    text = str(value).strip()

    for m_name, m_num in THAI_MONTHS.items():

        match = re.search(
            rf"(\d{{1,2}})\s*{re.escape(m_name)}\s*(\d{{4}})",
            text
        )

        if match:

            d = int(match.group(1))
            y = int(match.group(2))

            if y >= 2400:
                y -= 543

            try:
                return pd.Timestamp(y, m_num, d)
            except Exception:
                return None

    # YYYY-MM-DD / DD-MM-YYYY
    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a, b, c = map(int, match.groups())

        if a >= 1000:
            y, m, d = a, b, c
        else:
            d, m, y = a, b, c

        if y < 100:
            y += 2000

        if y >= 2400:
            y -= 543

        try:
            return pd.Timestamp(y, m, d)
        except Exception:
            pass

    return None


# ============================================================
# 5. SCRAPER
# ============================================================

class ScrapingError(Exception):
    pass


class NetworkScrapingError(ScrapingError):
    pass


class HTTPStatusScrapingError(ScrapingError):
    pass


class ParsingScrapingError(ScrapingError):
    pass


def extract_numbers(text):

    return re.findall(r"\b\d{2,6}\b", text)


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def fetch_lottery_data(url, lottery_name):

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Linux; Android 10) "
            "AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36",

        "Accept-Language":
            "th-TH,th;q=0.9,en;q=0.8"
    }

    try:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=12
            )

        except requests.exceptions.RequestException as exc:

            raise NetworkScrapingError(
                f"Network error: {exc}"
            )

        if response.status_code != 200:

            raise HTTPStatusScrapingError(
                f"HTTP {response.status_code}"
            )

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

        extracted = []

        # ----------------------------------------------------
        # A. TABLE
        # ----------------------------------------------------

        for table in content.find_all("table"):

            current_date = None

            for row in table.find_all("tr"):

                cells = " ".join(
                    c.get_text(" ", strip=True)
                    for c in row.find_all(["td", "th"])
                )

                if not cells:
                    continue

                parsed = normalize_date(cells)

                if parsed is not None:
                    current_date = parsed

                nums = extract_numbers(cells)

                # =================================================
                # หวยไทย
                # 6 หลัก + 2 หลัก
                # เช่น 932479 | 69
                # =================================================

                if lottery_name == "หวยไทย":

                    six = [
                        x for x in nums
                        if len(x) == 6
                    ]

                    two = [
                        x for x in nums
                        if len(x) == 2
                    ]

                    if current_date and six and two:

                        extracted.append({
                            "Date": current_date,
                            "Result_6D": six[0],
                            "Result_2D": two[-1]
                        })

                # =================================================
                # หวยประเภททั่วไป
                # 3 หลัก + 2 หลัก
                # =================================================

                else:

                    three = [
                        x for x in nums
                        if len(x) == 3
                    ]

                    two = [
                        x for x in nums
                        if len(x) == 2
                    ]

                    if current_date and three and two:

                        extracted.append({
                            "Date": current_date,
                            "Result_6D": three[0],
                            "Result_2D": two[-1]
                        })

        # ----------------------------------------------------
        # B. TEXT FALLBACK
        # ----------------------------------------------------

        if not extracted:

            lines = [
                x.strip()
                for x in content.get_text(
                    separator="\n"
                ).split("\n")
                if x.strip()
            ]

            current_date = None

            for line in lines:

                parsed = normalize_date(line)

                if parsed is not None:
                    current_date = parsed

                nums = extract_numbers(line)

                if lottery_name == "หวยไทย":

                    six = [
                        x for x in nums
                        if len(x) == 6
                    ]

                    two = [
                        x for x in nums
                        if len(x) == 2
                    ]

                    if current_date and six and two:

                        extracted.append({
                            "Date": current_date,
                            "Result_6D": six[0],
                            "Result_2D": two[-1]
                        })

                else:

                    three = [
                        x for x in nums
                        if len(x) == 3
                    ]

                    two = [
                        x for x in nums
                        if len(x) == 2
                    ]

                    if current_date and three and two:

                        extracted.append({
                            "Date": current_date,
                            "Result_6D": three[0],
                            "Result_2D": two[-1]
                        })

        # ----------------------------------------------------
        # C. VALIDATE
        # ----------------------------------------------------

        if not extracted:

            raise ParsingScrapingError(
                "ไม่พบข้อมูลในรูปแบบที่ระบบรองรับ"
            )

        df = pd.DataFrame(extracted)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Result_6D"] = (
            df["Result_6D"]
            .astype(str)
            .str.extract(r"(\d{6})")[0]
            .str.zfill(6)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(r"(\d{2})")[0]
            .str.zfill(2)
        )

        df = df.dropna()

        df = df[
            df["Result_6D"].str.match(r"^\d{3,6}$")
            &
            df["Result_2D"].str.match(r"^\d{2}$")
        ]

        df = (
            df
            .drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if df.empty:

            raise ParsingScrapingError(
                "ข้อมูลถูกกรองออกทั้งหมด"
            )

        # =====================================================
        # IMPORTANT
        #
        # หวยไทย:
        # 932479
        #
        # 3 ตัวบน = 479
        #
        # ไม่ใช่ 932
        # =====================================================

        if lottery_name == "หวยไทย":

            df["Result_3D"] = (
                df["Result_6D"]
                .str[-3:]
                .str.zfill(3)
            )

        else:

            df["Result_3D"] = (
                df["Result_6D"]
                .str[-3:]
                .str.zfill(3)
            )

        return df[
            [
                "Date",
                "Result_6D",
                "Result_3D",
                "Result_2D"
            ]
        ]

    except ScrapingError:
        raise

    except Exception as exc:

        raise ParsingScrapingError(
            f"Parsing error: {exc}"
        )


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def gap_since_digit(series, digit):

    arr = series.shift(1).to_numpy()

    output = np.zeros(
        len(series),
        dtype=np.float32
    )

    last_seen = -1

    for i, value in enumerate(arr):

        if pd.notna(value):

            if int(value) == digit:
                last_seen = i

        output[i] = (
            i + 1
            if last_seen < 0
            else i - last_seen
        )

    return pd.Series(
        output,
        index=series.index
    )


def build_features(df):

    work = df.copy()

    # --------------------------------------------------------
    # POSITION DIGITS
    # --------------------------------------------------------

    work["H"] = (
        work["Result_3D"]
        .str[0]
        .astype(int)
    )

    work["T"] = (
        work["Result_3D"]
        .str[1]
        .astype(int)
    )

    work["O"] = (
        work["Result_3D"]
        .str[2]
        .astype(int)
    )

    work["T2"] = (
        work["Result_2D"]
        .str[0]
        .astype(int)
    )

    work["O2"] = (
        work["Result_2D"]
        .str[1]
        .astype(int)
    )

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    dt = work["Date"].dt

    work["DOW"] = dt.dayofweek
    work["DAY"] = dt.day
    work["MONTH"] = dt.month
    work["DAY_OF_YEAR"] = dt.dayofyear
    work["WEEK_OF_YEAR"] = (
        dt.isocalendar()
        .week
        .astype(int)
    )

    work["DOW_SIN"] = np.sin(
        2 * np.pi * work["DOW"] / 7
    )

    work["DOW_COS"] = np.cos(
        2 * np.pi * work["DOW"] / 7
    )

    work["MONTH_SIN"] = np.sin(
        2 * np.pi * work["MONTH"] / 12
    )

    work["MONTH_COS"] = np.cos(
        2 * np.pi * work["MONTH"] / 12
    )

    # --------------------------------------------------------
    # PREVIOUS RESULT FEATURES
    # --------------------------------------------------------

    p3 = work[
        ["H", "T", "O"]
    ].shift(1)

    p2 = work[
        ["T2", "O2"]
    ].shift(1)

    work["PREV_SUM3"] = p3.sum(axis=1)
    work["PREV_SUM2"] = p2.sum(axis=1)

    work["PREV_RANGE3"] = (
        p3.max(axis=1)
        -
        p3.min(axis=1)
    )

    work["PREV_MEAN3"] = p3.mean(axis=1)

    work["PREV_HIGH_COUNT"] = (
        p3 >= 5
    ).sum(axis=1)

    work["PREV_ODD_COUNT"] = (
        p3 % 2
    ).sum(axis=1)

    work["PREV_UNIQUE3"] = (
        p3.nunique(axis=1)
    )

    work["PREV_REPEAT3"] = (
        3 - work["PREV_UNIQUE3"]
    )

    # --------------------------------------------------------
    # POSITION FEATURES
    #
    # ลด Feature ลงเพื่อเพิ่มความเร็ว
    # --------------------------------------------------------

    for pos in POSITIONS:

        s = work[pos]
        shifted = s.shift(1)

        # LAGS
        for lag in range(1, 6):

            work[f"{pos}_L{lag}"] = (
                s.shift(lag)
            )

        # ROLLING
        for w in [3, 5, 10]:

            roll = shifted.rolling(
                w,
                min_periods=2
            )

            work[f"{pos}_M{w}"] = (
                roll.mean()
            )

            work[f"{pos}_S{w}"] = (
                roll.std()
            )

            # only useful digits
            for d in [0, 5, 7]:

                work[
                    f"{pos}_F{w}_{d}"
                ] = (
                    (shifted == d)
                    .astype(float)
                    .rolling(
                        w,
                        min_periods=2
                    )
                    .mean()
                )

        # DIFFERENCE
        work[f"{pos}_D1"] = (
            s.shift(1)
            -
            s.shift(2)
        )

        work[f"{pos}_D2"] = (
            s.shift(2)
            -
            s.shift(3)
        )

        # ODD / HIGH
        work[f"{pos}_ODD"] = (
            shifted % 2
        )

        work[f"{pos}_HIGH"] = (
            shifted >= 5
        ).astype(float)

        # MOD
        work[f"{pos}_MOD3"] = (
            shifted % 3
        )

        work[f"{pos}_MOD5"] = (
            shifted % 5
        )

        # MIRROR
        work[f"{pos}_MIRROR"] = (
            9 - shifted
        )

        # SIN/COS DIGIT
        work[f"{pos}_SIN"] = np.sin(
            2 * np.pi * shifted / 10
        )

        work[f"{pos}_COS"] = np.cos(
            2 * np.pi * shifted / 10
        )

        # EWMA
        work[f"{pos}_EWMA3"] = (
            shifted
            .ewm(
                span=3,
                adjust=False
            )
            .mean()
        )

        work[f"{pos}_EWMA7"] = (
            shifted
            .ewm(
                span=7,
                adjust=False
            )
            .mean()
        )

        # REPEAT
        work[f"{pos}_IS_REPEAT"] = (
            shifted
            ==
            s.shift(2)
        ).astype(float)

        # GAP
        for d in [0, 5, 7]:

            work[
                f"{pos}_GAP_{d}"
            ] = gap_since_digit(
                s,
                d
            )

    # --------------------------------------------------------
    # PREVIOUS SUMMARY LAGS
    # --------------------------------------------------------

    for lag in [1, 2, 3]:

        c = work[
            ["H", "T", "O"]
        ].shift(lag)

        work[
            f"PREV_SUM3_L{lag}"
        ] = c.sum(axis=1)

        work[
            f"PREV_RANGE3_L{lag}"
        ] = (
            c.max(axis=1)
            -
            c.min(axis=1)
        )

        work[
            f"PREV_ODD3_L{lag}"
        ] = (
            c % 2
        ).sum(axis=1)

    return work.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 7. COMPACT FEATURE LIST
# ============================================================

FEATURES = [
    "DOW",
    "DAY",
    "MONTH",
    "DAY_OF_YEAR",
    "WEEK_OF_YEAR",
    "DOW_SIN",
    "DOW_COS",
    "MONTH_SIN",
    "MONTH_COS",

    "PREV_SUM3",
    "PREV_SUM2",
    "PREV_RANGE3",
    "PREV_MEAN3",
    "PREV_HIGH_COUNT",
    "PREV_ODD_COUNT",
    "PREV_UNIQUE3",
    "PREV_REPEAT3",
]

for lag in [1, 2, 3]:

    FEATURES.extend([
        f"PREV_SUM3_L{lag}",
        f"PREV_RANGE3_L{lag}",
        f"PREV_ODD3_L{lag}"
    ])


for pos in POSITIONS:

    FEATURES.extend([
        f"{pos}_L{l}"
        for l in range(1, 6)
    ])

    for w in [3, 5, 10]:

        FEATURES.extend([
            f"{pos}_M{w}",
            f"{pos}_S{w}",
            f"{pos}_F{w}_0",
            f"{pos}_F{w}_5",
            f"{pos}_F{w}_7",
        ])

    FEATURES.extend([
        f"{pos}_D1",
        f"{pos}_D2",
        f"{pos}_ODD",
        f"{pos}_HIGH",
        f"{pos}_MOD3",
        f"{pos}_MOD5",
        f"{pos}_MIRROR",
        f"{pos}_SIN",
        f"{pos}_COS",
        f"{pos}_EWMA3",
        f"{pos}_EWMA7",
        f"{pos}_IS_REPEAT",
        f"{pos}_GAP_0",
        f"{pos}_GAP_5",
        f"{pos}_GAP_7",
    ])

FEATURES = list(
    dict.fromkeys(FEATURES)
)


# ============================================================
# 8. FAST CONFIG
# ============================================================

def get_adaptive_config(n):

    # ---------------------------------------------
    # 700+
    # ---------------------------------------------

    if n >= 700:

        return {
            "min_train": 120,
            "trees": 80,
            "depth": 8,
            "leaf": 2,
            "backtest_start": 120,
            "recent_decay": 0.985,
            "max_backtest": 45
        }

    # ---------------------------------------------
    # 400+
    # ---------------------------------------------

    if n >= 400:

        return {
            "min_train": 100,
            "trees": 70,
            "depth": 7,
            "leaf": 2,
            "backtest_start": 100,
            "recent_decay": 0.98,
            "max_backtest": 40
        }

    # ---------------------------------------------
    # 200+
    # ---------------------------------------------

    if n >= 200:

        return {
            "min_train": 80,
            "trees": 60,
            "depth": 6,
            "leaf": 2,
            "backtest_start": 80,
            "recent_decay": 0.975,
            "max_backtest": 35
        }

    # ---------------------------------------------
    # 50+
    # ---------------------------------------------

    return {
        "min_train": 50,
        "trees": 45,
        "depth": 5,
        "leaf": 2,
        "backtest_start": 50,
        "recent_decay": 0.97,
        "max_backtest": 25
    }


# ============================================================
# 9. MODELS
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
        max_iter=max(40, int(t * 0.7)),
        max_leaf_nodes=15,
        learning_rate=0.04,
        min_samples_leaf=l,
        l2_regularization=2.0,
        random_state=42
    )


# ============================================================
# 10. PROBABILITY
# ============================================================

def stabilize_probability(
    probs,
    temperature=0.90
):

    probs = np.asarray(
        probs,
        dtype=float
    )

    probs = np.clip(
        probs,
        1e-9,
        1
    )

    probs /= probs.sum()

    logits = (
        np.log(probs)
        /
        temperature
    )

    logits -= logits.max()

    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum()


# ============================================================
# 11. FAST TRAIN
# ============================================================

def train_and_predict(
    X_train,
    y_train,
    X_test,
    config
):

    # ---------------------------------------------
    # Fixed compact features
    #
    # ไม่ทำ feature selection ทุก Backtest
    # จึงเร็วขึ้นมาก และยังไม่เกิด leakage
    # ---------------------------------------------

    selected = [
        f for f in FEATURES
        if f in X_train.columns
    ]

    imp = SimpleImputer(
        strategy="median"
    )

    Xtr = imp.fit_transform(
        X_train[selected]
    )

    Xte = imp.transform(
        X_test[selected]
    )

    model_probs = {}

    for model_name in MODEL_NAMES:

        try:

            model = create_model(
                model_name,
                config
            )

            model.fit(
                Xtr,
                y_train
            )

            raw = model.predict_proba(
                Xte
            )[0]

            out = np.zeros(
                10,
                dtype=np.float32
            )

            for cls, prob in zip(
                model.classes_,
                raw
            ):

                cls = int(cls)

                if 0 <= cls <= 9:

                    out[cls] = float(prob)

            model_probs[
                model_name
            ] = stabilize_probability(
                out
            )

        except Exception:
            continue

    if not model_probs:
        return None

    # Equal AI ensemble
    ensemble = np.mean(
        list(model_probs.values()),
        axis=0
    )

    ensemble = stabilize_probability(
        ensemble
    )

    return (
        ensemble,
        model_probs,
        selected
    )


# ============================================================
# 12. FAST WALK-FORWARD
# ============================================================

def strict_walk_forward_backtest(
    df_feat,
    pos,
    config
):

    X = df_feat[
        FEATURES
    ].astype(float)

    y = df_feat[
        pos
    ].astype(int)

    n = len(df_feat)

    start = max(
        config["min_train"],
        config["backtest_start"]
    )

    max_bt = config[
        "max_backtest"
    ]

    if n - start > max_bt:

        start = n - max_bt

    history = []

    for t_idx in range(
        start,
        n
    ):

        if y.iloc[
            :t_idx
        ].nunique() < 2:

            continue

        result = train_and_predict(
            X.iloc[:t_idx],
            y.iloc[:t_idx],
            X.iloc[[t_idx]],
            config
        )

        if result is None:
            continue

        ensemble = result[0]

        actual = int(
            y.iloc[t_idx]
        )

        ranking = np.argsort(
            ensemble
        )[::-1]

        history.append({
            "actual": actual,

            "top1": int(
                actual
                in ranking[:1]
            ),

            "top3": int(
                actual
                in ranking[:3]
            ),

            "top5": int(
                actual
                in ranking[:5]
            ),

            "dead7": int(
                actual
                in ranking[-7:]
            ),

            "logloss":
                -np.log(
                    max(
                        ensemble[actual],
                        1e-9
                    )
                ),

            "brier":
                np.sum(
                    (
                        ensemble
                        -
                        np.eye(10)[actual]
                    ) ** 2
                )
        })

    if not history:

        return {
            "scores": {},
            "tests": 0
        }

    hist = pd.DataFrame(
        history
    )

    # ---------------------------------------------
    # Recent weighted score
    # ---------------------------------------------

    decay = config[
        "recent_decay"
    ]

    w = decay ** (
        len(hist)
        -
        np.arange(len(hist))
        -
        1
    )

    w /= w.sum()

    metrics = {}

    for key in [
        "top1",
        "top3",
        "top5",
        "dead7",
        "logloss",
        "brier"
    ]:

        metrics[key] = float(
            np.sum(
                hist[key]
                *
                w
            )
        )

    # Stability
    roll_window = min(
        20,
        max(
            5,
            len(hist)
        )
    )

    rolling = (
        hist["top3"]
        .rolling(
            roll_window,
            min_periods=5
        )
        .mean()
    )

    if len(
        rolling.dropna()
    ):

        metrics["stability"] = float(
            np.clip(
                1.0
                -
                rolling.std(),
                0,
                1
            )
        )

    else:

        metrics["stability"] = 0.0

    # ---------------------------------------------
    # Overall score
    # ---------------------------------------------

    metrics["score"] = (
        0.30 * metrics["top1"]
        +
        0.25 * metrics["top3"]
        +
        0.15 * metrics["top5"]
        +
        0.10 * metrics["stability"]
        +
        0.10 *
        (
            1
            /
            (1 + metrics["logloss"])
        )
        +
        0.10 *
        (
            1
            /
            (1 + metrics["brier"])
        )
    )

    # Raw
    for key in [
        "top1",
        "top3",
        "top5",
        "dead7"
    ]:

        metrics[
            f"raw_{key}"
        ] = float(
            hist[key].mean()
        )

    return {
        "scores": metrics,
        "tests": len(hist),
        "history": history
    }


# ============================================================
# 13. FINAL PREDICTION
# ============================================================

def final_prediction(
    df_feat,
    pos,
    config
):

    X = df_feat[
        FEATURES
    ].astype(float)

    y = df_feat[
        pos
    ].astype(int)

    # --------------------------------------------------------
    # สำคัญ:
    # แถวสุดท้ายเป็น Target Date
    # จึง train เฉพาะข้อมูลก่อนหน้า
    # --------------------------------------------------------

    result = train_and_predict(
        X.iloc[:-1],
        y.iloc[:-1],
        X.iloc[[-1]],
        config
    )

    if result is None:

        ensemble = np.ones(
            10
        ) / 10

        model_probs = {}

        selected = []

    else:

        ensemble, model_probs, selected = result

    ranking = np.argsort(
        ensemble
    )[::-1]

    hot = [
        (
            int(i),
            float(
                ensemble[i]
            )
        )
        for i in ranking[:5]
    ]

    dead = [
        (
            int(i),
            float(
                ensemble[i]
            )
        )
        for i in ranking[-7:][::-1]
    ]

    # Model confidence
    top_values = np.sort(
        ensemble
    )[::-1]

    confidence = float(
        top_values[0]
        -
        top_values[1]
    )

    top3_concentration = float(
        top_values[:3].sum()
    )

    # Model agreement
    ranks = []

    for p in model_probs.values():

        ranks.append(
            set(
                np.argsort(p)[::-1][:5]
            )
        )

    agreements = []

    for i in range(
        len(ranks)
    ):

        for j in range(
            i + 1,
            len(ranks)
        ):

            agreements.append(
                len(
                    ranks[i]
                    &
                    ranks[j]
                )
                /
                5
            )

    agreement = (
        float(
            np.mean(agreements)
        )
        if agreements
        else 0.0
    )

    # Model selected by Top-3 mass
    if model_probs:

        selected_model = max(
            model_probs.keys(),
            key=lambda k:
                np.sort(
                    model_probs[k]
                )[::-1][:3].sum()
        )

    else:

        selected_model = "Ensemble"

    return {
        "model": selected_model,

        "weights": {
            k:
            1 / len(model_probs)
            for k in model_probs
        } if model_probs else {},

        "model_probabilities":
            model_probs,

        "probabilities":
            ensemble,

        "hot":
            hot,

        "dead":
            dead,

        "confidence":
            confidence,

        "top3_concentration":
            top3_concentration,

        "agreement":
            agreement,

        "selected_features":
            selected
    }


# ============================================================
# 14. DISPLAY
# ============================================================

def display_card(
    pos,
    result,
    is_hot=True
):

    data = (
        result["hot"]
        if is_hot
        else result["dead"]
    )

    style = (
        "hot"
        if is_hot
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
            AI Probability
            {"ต่ำสุด" if not is_hot else ""}:
            {probs}
        </div>
    """

    if is_hot:

        html += f"""
        <div class="confidence">

        📌 Top-1 Gap:
        {result["confidence"]*100:.1f}%

        &nbsp;|&nbsp;

        Top-3:
        {result["top3_concentration"]*100:.1f}%

        &nbsp;|&nbsp;

        Agreement:
        {result["agreement"]*100:.1f}%

        </div>
        """

    html += f"""
        <div class="model-badge">
            🤖
            {"Final AI: " + result["model"]
             if is_hot
             else
             "Adaptive Walk-Forward AI"}
        </div>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(
        """
        <div class="main-title">
            🤖 LOTTO AI PRO V8.2 TURBO MAX
        </div>

        <div class="subtitle">
            STRICT WALK-FORWARD • AI ONLY •
            LEAKAGE SAFE • ⚡ FAST
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
        ["อัตโนมัติ"]
        +
        DOW_NAMES
    )

    if not st.button(
        "🚀 เริ่มวิเคราะห์ PRO V8.2 TURBO MAX",
        type="primary",
        use_container_width=True
    ):

        return st.info(
            "เลือกหวยและวันเป้าหมาย แล้วกด 🚀 เริ่มวิเคราะห์"
        )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    with st.spinner(
        "📥 กำลังโหลดข้อมูลย้อนหลัง..."
    ):

        try:

            df = fetch_lottery_data(
                LOTTERY_SOURCES[lottery],
                lottery
            )

        except Exception as exc:

            return st.error(
                f"❌ Error: {exc}"
            )

    if len(df) < 50:

        return st.error(
            f"❌ พบข้อมูลเพียง {len(df)} งวด "
            f"(ต้องการ ≥50)"
        )

    # ========================================================
    # TARGET DATE
    # ========================================================

    if selected_day == "อัตโนมัติ":

        tgt_dow = None

    else:

        tgt_dow = DOW_NAMES.index(
            selected_day
        )

    last_d = pd.Timestamp(
        df["Date"].iloc[-1]
    )

    if tgt_dow is not None:

        days_ahead = (
            tgt_dow
            -
            last_d.dayofweek
        ) % 7

        target_date = (
            last_d
            +
            timedelta(
                days=days_ahead or 7
            )
        )

    else:

        if len(df) >= 2:

            gap = (
                last_d
                -
                pd.Timestamp(
                    df["Date"].iloc[-2]
                )
            ).days

            gap = max(
                gap,
                1
            )

        else:

            gap = 7

        target_date = (
            last_d
            +
            timedelta(
                days=gap
            )
        )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    with st.spinner(
        "🧠 สร้าง Causal Features..."
    ):

        dummy = {
            "Date":
                target_date,
            "Result_6D":
                "000000",
            "Result_3D":
                "000",
            "Result_2D":
                "00"
        }

        ext_df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [dummy]
                )
            ],
            ignore_index=True
        )

        feat_df = build_features(
            ext_df
        )

        config = get_adaptive_config(
            len(df)
        )

    # ========================================================
    # INFO
    # ========================================================

    st.info(
        f"""
        ⚡ **V8.2 TURBO MAX**

        ข้อมูล {len(df):,} งวด
        • Min Train {config["min_train"]}
        • Feature Pool {len(FEATURES)}
        • Trees {config["trees"]}
        • Max Backtest {config["max_backtest"]}
        """
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    backtest_res = {}
    final_res = {}

    progress = st.progress(0)
    status = st.empty()

    for i, pos in enumerate(
        POSITIONS
    ):

        status.caption(
            f"🧠 Walk-Forward: "
            f"{POSITION_LABELS[pos]}"
        )

        backtest_res[pos] = (
            strict_walk_forward_backtest(
                feat_df.iloc[:-1],
                pos,
                config
            )
        )

        progress.progress(
            int(
                (i + 1)
                /
                len(POSITIONS)
                *
                100
            )
        )

    progress.empty()
    status.empty()

    # ========================================================
    # FINAL AI
    # ========================================================

    with st.spinner(
        "🤖 Train Final AI..."
    ):

        for pos in POSITIONS:

            final_res[pos] = (
                final_prediction(
                    feat_df,
                    pos,
                    config
                )
            )

    # ========================================================
    # STATUS
    # ========================================================

    bt_tests = [
        x["tests"]
        for x in backtest_res.values()
    ]

    st.markdown(
        f"""
        <div class="status-card">

        🤖 <b>LOTTO AI PRO V8.2 TURBO MAX</b>

        <br>

        🎯 หวย:
        <b>{lottery}</b>

        <br>

        📊 ข้อมูล:
        <b>{len(df):,} งวด</b>

        <br>

        📅 Target:
        <b>{target_date.strftime("%d/%m/%Y")}</b>

        <br>

        🧩 Features:
        <b>{len(FEATURES)}</b>

        &nbsp;|&nbsp;

        🌳 Trees:
        <b>{config["trees"]}</b>

        <br>

        ⚡ Backtest:
        <b>{min(bt_tests)}–{max(bt_tests)}</b>
        งวด

        </div>

        <br>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # TABS
    # ========================================================

    t1, t2, t3 = st.tabs(
        [
            "🎯 เลขเด่น AI",
            "🛑 เลขดับ 7",
            "📊 Walk-Forward"
        ]
    )

    # ========================================================
    # HOT
    # ========================================================

    with t1:

        for pos in POSITIONS:

            display_card(
                pos,
                final_res[pos],
                True
            )

    # ========================================================
    # DEAD
    # ========================================================

    with t2:

        for pos in POSITIONS:

            display_card(
                pos,
                final_res[pos],
                False
            )

    # ========================================================
    # BACKTEST
    # ========================================================

    with t3:

        for pos in POSITIONS:

            st.markdown(
                f"### {POSITION_LABELS[pos]}"
            )

            scores = (
                backtest_res[pos]
                .get(
                    "scores",
                    {}
                )
            )

            if not scores:

                st.warning(
                    "ไม่มีข้อมูล Backtest เพียงพอ"
                )

                continue

            table = pd.DataFrame([

                {
                    "Metric":
                        "Top-1",

                    "AI":
                        f"{scores['top1']*100:.1f}%",

                    "Random":
                        "10.0%",

                    "Edge":
                        f"{(scores['top1']-.10)*100:+.1f}%"
                },

                {
                    "Metric":
                        "Top-3",

                    "AI":
                        f"{scores['top3']*100:.1f}%",

                    "Random":
                        "30.0%",

                    "Edge":
                        f"{(scores['top3']-.30)*100:+.1f}%"
                },

                {
                    "Metric":
                        "Top-5",

                    "AI":
                        f"{scores['top5']*100:.1f}%",

                    "Random":
                        "50.0%",

                    "Edge":
                        f"{(scores['top5']-.50)*100:+.1f}%"
                },

                {
                    "Metric":
                        "Dead-7",

                    "AI":
                        f"{scores['dead7']*100:.1f}%",

                    "Random":
                        "70.0%",

                    "Edge":
                        f"{(scores['dead7']-.70)*100:+.1f}%"
                },

                {
                    "Metric":
                        "LogLoss / Brier",

                    "AI":
                        f"L: {scores['logloss']:.3f} | "
                        f"B: {scores['brier']:.3f}",

                    "Random":
                        "-",

                    "Edge":
                        "-"
                }

            ])

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    with st.expander(
        "🎯 สรุปเลขเด่นทั้งหมด",
        expanded=True
    ):

        summary = []

        for pos in POSITIONS:

            top = (
                final_res[pos]["hot"][0]
            )

            score = (
                backtest_res[pos]
                .get(
                    "scores",
                    {}
                )
                .get(
                    "top1",
                    0
                )
            )

            summary.append({

                "ตำแหน่ง":
                    POSITION_LABELS[pos],

                "Top-1":
                    top[0],

                "AI Probability":
                    f"{top[1]*100:.1f}%",

                "AI Model":
                    final_res[pos]["model"],

                "WF Top-1":
                    f"{score*100:.1f}%"

            })

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
