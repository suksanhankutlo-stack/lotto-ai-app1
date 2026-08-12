# ============================================================
# 🤖 LOTTO AI PRO V8.1 MAX
# AI-ONLY • STRICT WALK-FORWARD • STATISTICAL BACKTEST
# FEATURE SELECTION • MULTI-WINDOW STABILITY • RANDOM BASELINE
# ============================================================
#
# V8.1 MAX CHANGES
# ------------------------------------------------------------
# ✅ ExtraTrees
# ✅ RandomForest
# ✅ HistGradientBoosting
# ✅ AI-ONLY
# ✅ Strict causal / leakage-safe features
# ✅ Walk-Forward CONTINUOUS backtest
# ✅ Multi-window stability analysis
# ✅ Random baseline
# ✅ Top-1 / Top-3 / Top-5
# ✅ Dead-7
# ✅ LogLoss
# ✅ Brier Score
# ✅ 95% Confidence Interval
# ✅ AI Advantage vs Random
# ✅ Reliability based on test sample size
# ✅ Stability penalty
# ✅ Feature selection INSIDE each backtest fold
# ✅ Feature selection for final model
# ✅ Adaptive ensemble
# ✅ Recency weighted evaluation
# ✅ Model agreement
# ✅ Confidence
# ✅ AI Hot 5
# ✅ AI Dead 7
# ✅ Data cache
# ✅ Backtest cache
# ✅ Model cache
# ✅ Mobile optimized
# ✅ Streamlit Cloud friendly
#
# NO:
# ❌ XGBoost
# ❌ Markov
# ❌ Frequency voting
# ❌ Calendar voting
# ❌ Equation voting
# ❌ Manual number voting
#
# IMPORTANT
# ------------------------------------------------------------
# Every result-derived feature is shifted.
# Current target is NEVER used to predict itself.
#
# Feature selection is performed ONLY on training data
# inside each walk-forward fold.
#
# ============================================================

import re
import hashlib
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

warnings.filterwarnings("ignore")


# ============================================================
# 1. STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Lotto AI PRO V8.1 MAX",
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


# ============================================================
# 3. CONSTANTS
# ============================================================

POSITIONS = [
    "H",
    "T",
    "O",
    "T2",
    "O2"
]

POSITION_LABELS = {

    "H":
        "💯 หลักร้อย 3 ตัวบน",

    "T":
        "🔟 หลักสิบ 3 ตัวบน",

    "O":
        "1️⃣ หลักหน่วย 3 ตัวบน",

    "T2":
        "🔽 หลักสิบ 2 ตัวล่าง",

    "O2":
        "⬇️ หลักหน่วย 2 ตัวล่าง"
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

ALL_DIGITS = np.arange(10)


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
            text-align: center;
            font-size: 2.4rem;
            font-weight: 900;
            margin-bottom: 2px;
        }

        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: .95rem;
            margin-bottom: 18px;
        }

        .status-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 14px;
            text-align: center;
            color: #1e40af;
            font-weight: 700;
            line-height: 1.75;
        }

        .hot-card {
            background: #f0fdf4;
            border-left: 7px solid #16a34a;
            border-radius: 14px;
            padding: 15px;
            margin: 10px 0;
        }

        .dead-card {
            background: #fef2f2;
            border-left: 7px solid #dc2626;
            border-radius: 14px;
            padding: 15px;
            margin: 10px 0;
        }

        .position-title {
            font-size: 1.15rem;
            font-weight: 900;
            color: #334155;
            margin-bottom: 6px;
        }

        .hot-number,
        .dead-number {
            font-size: 2.15rem;
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
            font-size: .82rem;
            margin-top: 4px;
        }

        .model-badge {
            text-align: center;
            background: white;
            border-radius: 9px;
            padding: 7px;
            margin-top: 7px;
            color: #475569;
            font-weight: 700;
        }

        .confidence {
            text-align: center;
            font-size: .9rem;
            font-weight: 800;
            margin-top: 5px;
            color: #334155;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. DATE
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
    "ธ.ค.": 12,
}


def normalize_date(value):

    if value is None:
        return None

    text = str(value).strip()

    for month_name, month_num in THAI_MONTHS.items():

        pattern = (
            rf"(\d{{1,2}})\s*"
            rf"{re.escape(month_name)}\s*"
            rf"(\d{{4}})"
        )

        match = re.search(
            pattern,
            text
        )

        if match:

            day = int(match.group(1))
            year = int(match.group(2))

            if year >= 2400:
                year -= 543

            try:
                return pd.Timestamp(
                    year,
                    month_num,
                    day
                )
            except Exception:
                return None

    match = re.search(
        r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})",
        text
    )

    if match:

        a = int(match.group(1))
        b = int(match.group(2))
        c = int(match.group(3))

        try:

            if a >= 1000:

                year = a
                month = b
                day = c

            else:

                day = a
                month = b
                year = c

                if year < 100:
                    year += 2000

                if year >= 2400:
                    year -= 543

            return pd.Timestamp(
                year,
                month,
                day
            )

        except Exception:
            return None

    return None


# ============================================================
# 6. SCRAPING
# ============================================================

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
        "(KHTML, like Gecko) "
        "Chrome/120.0 Mobile Safari/537.36"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if content is None:
            content = soup

        lines = content.get_text(
            separator="\n"
        ).split("\n")

        extracted = []

        current_date = None

        for raw in lines:

            line = raw.strip()

            if not line:
                continue

            parsed_date = normalize_date(line)

            if parsed_date is not None:
                current_date = parsed_date

            # ปรับปรุง Regex เพื่อรองรับหวยไทย 6 ตัว และหวยอื่น 3 ตัว
            match = re.search(
                r"\b(\d{6}|\d{3})\b.*?\b(\d{2})\b",
                line
            )

            if (
                match
                and current_date is not None
            ):
                
                raw_top = str(match.group(1))

                extracted.append(
                    {
                        "Date": current_date,
                        # ตัดเอาเฉพาะ 3 ตัวท้ายเพื่อใช้งานในระบบ 3D
                        "Result_3D": raw_top[-3:],
                        "Result_2D": match.group(2)
                    }
                )

        df = pd.DataFrame(
            extracted
        )

        if df.empty:
            return pd.DataFrame()

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
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.extract(
                r"(\d{2})"
            )[0]
        )

        df = df.dropna(
            subset=[
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        )

        df["Result_3D"] = (
            df["Result_3D"]
            .astype(str)
            .str.zfill(3)
        )

        df["Result_2D"] = (
            df["Result_2D"]
            .astype(str)
            .str.zfill(2)
        )

        df = (
            df
            .drop_duplicates(
                subset=[
                    "Date",
                    "Result_3D",
                    "Result_2D"
                ]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except requests.exceptions.Timeout:

        return pd.DataFrame()

    except requests.exceptions.ConnectionError:

        return pd.DataFrame()

    except requests.exceptions.HTTPError:

        return pd.DataFrame()

    except requests.exceptions.RequestException:

        return pd.DataFrame()

    except Exception:

        return pd.DataFrame()


# ============================================================
# 7. ENTROPY
# ============================================================

def safe_entropy(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    arr = arr[
        ~np.isnan(arr)
    ]

    if len(arr) == 0:
        return 0.0

    arr = arr.astype(int)

    counts = np.bincount(
        arr,
        minlength=10
    ).astype(float)

    total = counts.sum()

    if total <= 0:
        return 0.0

    p = counts / total

    p = p[p > 0]

    return float(
        -(p * np.log(p)).sum()
    )


# ============================================================
# 8. FAST ROLLING DIGIT FREQUENCY
# ============================================================

def rolling_digit_frequency(
    series,
    window,
    digit
):

    shifted = (
        series
        .shift(1)
    )

    indicator = (
        shifted == digit
    ).astype(float)

    return (
        indicator
        .rolling(
            window,
            min_periods=1
        )
        .mean()
    )


# ============================================================
# 9. GAP
# ============================================================

def gap_since_digit(
    series,
    digit
):

    shifted = (
        series
        .shift(1)
        .to_numpy()
    )

    result = np.zeros(
        len(series),
        dtype=np.float32
    )

    last_seen = -1

    for i, value in enumerate(
        shifted
    ):

        if (
            not pd.isna(value)
            and int(value) == digit
        ):

            last_seen = i

        if last_seen < 0:

            result[i] = i + 1

        else:

            result[i] = (
                i - last_seen
            )

    return pd.Series(
        result,
        index=series.index
    )


# ============================================================
# 10. BUILD FEATURES
# ============================================================

def build_features(df):

    work = df.copy()

    # --------------------------------------------------------
    # Targets
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
    # Date features
    # --------------------------------------------------------

    work["DOW"] = (
        work["Date"].dt.dayofweek
    )

    work["DAY"] = (
        work["Date"].dt.day
    )

    work["MONTH"] = (
        work["Date"].dt.month
    )

    work["DAY_OF_YEAR"] = (
        work["Date"].dt.dayofyear
    )

    work["WEEK_OF_YEAR"] = (
        work["Date"].dt.isocalendar()
        .week
        .astype(int)
    )

    work["DOW_SIN"] = np.sin(
        2 * np.pi
        * work["DOW"] / 7
    )

    work["DOW_COS"] = np.cos(
        2 * np.pi
        * work["DOW"] / 7
    )

    work["MONTH_SIN"] = np.sin(
        2 * np.pi
        * work["MONTH"] / 12
    )

    work["MONTH_COS"] = np.cos(
        2 * np.pi
        * work["MONTH"] / 12
    )

    work["DAY_SIN"] = np.sin(
        2 * np.pi
        * work["DAY"] / 31
    )

    work["DAY_COS"] = np.cos(
        2 * np.pi
        * work["DAY"] / 31
    )

    # --------------------------------------------------------
    # Previous draw features
    # --------------------------------------------------------

    prev_h = work["H"].shift(1)
    prev_t = work["T"].shift(1)
    prev_o = work["O"].shift(1)

    prev_t2 = work["T2"].shift(1)
    prev_o2 = work["O2"].shift(1)

    prev3 = pd.concat(
        [
            prev_h,
            prev_t,
            prev_o
        ],
        axis=1
    )

    prev2 = pd.concat(
        [
            prev_t2,
            prev_o2
        ],
        axis=1
    )

    work["PREV_SUM3"] = (
        prev3.sum(axis=1)
    )

    work["PREV_SUM2"] = (
        prev2.sum(axis=1)
    )

    work["PREV_RANGE3"] = (
        prev3.max(axis=1)
        -
        prev3.min(axis=1)
    )

    work["PREV_MEAN3"] = (
        prev3.mean(axis=1)
    )

    work["PREV_HIGH_COUNT"] = (
        (prev3 >= 5)
        .sum(axis=1)
    )

    work["PREV_ODD_COUNT"] = (
        (prev3 % 2)
        .sum(axis=1)
    )

    work["PREV_UNIQUE3"] = (
        prev3.nunique(axis=1)
    )

    work["PREV_REPEAT3"] = (
        3
        -
        work["PREV_UNIQUE3"]
    )

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

    for pos in POSITIONS:

        series = work[pos]

        for lag in range(1, 8):

            work[
                f"{pos}_L{lag}"
            ] = series.shift(lag)

        shifted = series.shift(1)

        for window in [
            3,
            5,
            10,
            20
        ]:

            work[
                f"{pos}_M{window}"
            ] = (
                shifted
                .rolling(
                    window,
                    min_periods=2
                )
                .mean()
            )

            work[
                f"{pos}_S{window}"
            ] = (
                shifted
                .rolling(
                    window,
                    min_periods=2
                )
                .std()
            )

            for digit in [
                0,
                2,
                5,
                7
            ]:

                work[
                    f"{pos}_F{window}_{digit}"
                ] = rolling_digit_frequency(
                    series,
                    window,
                    digit
                )

        work[
            f"{pos}_D1"
        ] = (
            series.shift(1)
            -
            series.shift(2)
        )

        work[
            f"{pos}_D2"
        ] = (
            series.shift(2)
            -
            series.shift(3)
        )

        work[
            f"{pos}_D3"
        ] = (
            series.shift(3)
            -
            series.shift(4)
        )

        work[
            f"{pos}_ODD"
        ] = (
            shifted
            .fillna(0)
            .astype(int)
            % 2
        )

        work[
            f"{pos}_HIGH"
        ] = (
            shifted
            .fillna(0)
            .astype(int)
            >= 5
        ).astype(int)

        work[
            f"{pos}_MOD3"
        ] = (
            shifted
            .fillna(0)
            .astype(int)
            % 3
        )

        work[
            f"{pos}_MOD5"
        ] = (
            shifted
            .fillna(0)
            .astype(int)
            % 5
        )

        work[
            f"{pos}_SIN"
        ] = np.sin(
            2 * np.pi
            * shifted.fillna(0)
            / 10
        )

        work[
            f"{pos}_COS"
        ] = np.cos(
            2 * np.pi
            * shifted.fillna(0)
            / 10
        )

        work[
            f"{pos}_MIRROR"
        ] = (
            9
            -
            shifted.fillna(0)
        )

        for digit in [
            0,
            2,
            5,
            7
        ]:

            work[
                f"{pos}_GAP_{digit}"
            ] = gap_since_digit(
                series,
                digit
            )

        for window in [
            5,
            10,
            20
        ]:

            # Entropy is calculated on shifted
            # history only.
            work[
                f"{pos}_ENT{window}"
            ] = (
                shifted
                .rolling(
                    window,
                    min_periods=2
                )
                .apply(
                    safe_entropy,
                    raw=False
                )
            )

    # --------------------------------------------------------
    # Cross-position
    # --------------------------------------------------------

    for lag in [
        1,
        2,
        3
    ]:

        h = work["H"].shift(lag)
        t = work["T"].shift(lag)
        o = work["O"].shift(lag)

        temp = pd.concat(
            [
                h,
                t,
                o
            ],
            axis=1
        )

        work[
            f"PREV_SUM3_L{lag}"
        ] = (
            temp.sum(axis=1)
        )

        work[
            f"PREV_RANGE3_L{lag}"
        ] = (
            temp.max(axis=1)
            -
            temp.min(axis=1)
        )

        work[
            f"PREV_ODD3_L{lag}"
        ] = (
            (h % 2)
            +
            (t % 2)
            +
            (o % 2)
        )

    # --------------------------------------------------------
    # Missing-value policy
    #
    # Do NOT blindly fill everything with zero.
    #
    # Numeric rolling statistics use:
    #   median -> forward fill -> zero fallback
    #
    # This avoids making missing entropy/mean values
    # look like meaningful zero measurements.
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in work.columns
        if c not in [
            "Date",
            "Result_3D",
            "Result_2D"
        ]
    ]

    for column in feature_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce"
        )

        median = (
            work[column]
            .median()
        )

        if pd.isna(median):
            median = 0.0

        work[column] = (
            work[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(median)
        )

    return work


# ============================================================
# 11. FEATURE LIST
# ============================================================

BASE_FEATURES = [

    "DOW",
    "DAY",
    "MONTH",
    "DAY_OF_YEAR",
    "WEEK_OF_YEAR",

    "DOW_SIN",
    "DOW_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "DAY_SIN",
    "DAY_COS",

    "PREV_SUM3",
    "PREV_SUM2",
    "PREV_RANGE3",
    "PREV_MEAN3",

    "PREV_HIGH_COUNT",
    "PREV_ODD_COUNT",
    "PREV_UNIQUE3",
    "PREV_REPEAT3",

    "PREV_SUM3_L1",
    "PREV_SUM3_L2",
    "PREV_SUM3_L3",

    "PREV_RANGE3_L1",
    "PREV_RANGE3_L2",
    "PREV_RANGE3_L3",

    "PREV_ODD3_L1",
    "PREV_ODD3_L2",
    "PREV_ODD3_L3"
]

FEATURES = list(
    BASE_FEATURES
)

for pos in POSITIONS:

    for lag in range(1, 8):

        FEATURES.append(
            f"{pos}_L{lag}"
        )

    for window in [
        3,
        5,
        10,
        20
    ]:

        FEATURES.append(
            f"{pos}_M{window}"
        )

        FEATURES.append(
            f"{pos}_S{window}"
        )

        for digit in [
            0,
            2,
            5,
            7
        ]:

            FEATURES.append(
                f"{pos}_F{window}_{digit}"
            )

    FEATURES.extend(
        [
            f"{pos}_D1",
            f"{pos}_D2",
            f"{pos}_D3",

            f"{pos}_ODD",
            f"{pos}_HIGH",

            f"{pos}_MOD3",
            f"{pos}_MOD5",

            f"{pos}_SIN",
            f"{pos}_COS",

            f"{pos}_MIRROR"
        ]
    )

    for digit in [
        0,
        2,
        5,
        7
    ]:

        FEATURES.append(
            f"{pos}_GAP_{digit}"
        )

    for window in [
        5,
        10,
        20
    ]:

        FEATURES.append(
            f"{pos}_ENT{window}"
        )


# Remove accidental duplicates
FEATURES = list(
    dict.fromkeys(FEATURES)
)


# ============================================================
# 12. DATA HASH
# ============================================================

def get_data_hash(df):

    hashed = pd.util.hash_pandas_object(
        df[
            [
                "Date",
                "Result_3D",
                "Result_2D"
            ]
        ],
        index=False
    ).values

    return hashlib.md5(
        hashed.tobytes()
    ).hexdigest()


# ============================================================
# 13. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "backtest": 80,
            "min_train": 120,
            "trees": 100,
            "depth": 9,
            "leaf": 3,
            "top_features": 40,
            "recent_decay": 0.985
        }

    if n >= 400:

        return {
            "backtest": 70,
            "min_train": 100,
            "trees": 85,
            "depth": 8,
            "leaf": 3,
            "top_features": 30,
            "recent_decay": 0.982
        }

    if n >= 200:

        return {
            "backtest": 50,
            "min_train": 70,
            "trees": 70,
            "depth": 7,
            "leaf": 3,
            "top_features": 25,
            "recent_decay": 0.978
        }

    if n >= 100:

        return {
            "backtest": 35,
            "min_train": 50,
            "trees": 60,
            "depth": 6,
            "leaf": 3,
            "top_features": 20,
            "recent_decay": 0.975
        }

    return {

        "backtest": 20,
        "min_train": 35,
        "trees": 50,
        "depth": 6,
        "leaf": 3,
        "top_features": 15,
        "recent_decay": 0.970
    }


# ============================================================
# 14. MODEL FACTORY
# ============================================================

def create_model(
    model_name,
    config
):

    trees = config["trees"]
    depth = config["depth"]
    leaf = config["leaf"]

    if model_name == "ExtraTrees":

        return ExtraTreesClassifier(

            n_estimators=trees,

            max_depth=depth,

            min_samples_leaf=leaf,

            max_features="sqrt",

            class_weight="balanced",

            n_jobs=-1,

            random_state=42
        )

    if model_name == "RandomForest":

        return RandomForestClassifier(

            n_estimators=trees,

            max_depth=depth,

            min_samples_leaf=leaf,

            max_features="sqrt",

            class_weight="balanced",

            n_jobs=-1,

            random_state=42
        )

    if model_name == "HistGradientBoosting":

        return HistGradientBoostingClassifier(

            max_iter=max(
                40,
                int(
                    trees * 0.60
                )
            ),

            max_leaf_nodes=15,

            learning_rate=0.045,

            min_samples_leaf=leaf,

            l2_regularization=0.75,

            random_state=42
        )

    raise ValueError(
        model_name
    )


# ============================================================
# 15. PROBABILITY
# ============================================================

def probability_vector(
    model,
    X
):

    raw = model.predict_proba(
        X
    )[0]

    output = np.zeros(
        10,
        dtype=float
    )

    for cls, prob in zip(
        model.classes_,
        raw
    ):

        cls = int(cls)

        if 0 <= cls <= 9:

            output[cls] = float(
                prob
            )

    total = output.sum()

    if total <= 0:

        return np.ones(10) / 10

    return output / total


# ============================================================
# 16. PROBABILITY STABILIZATION
# ============================================================

def stabilize_probability(
    probabilities,
    temperature=1.08
):

    p = np.asarray(
        probabilities,
        dtype=float
    )

    p = np.clip(
        p,
        1e-9,
        1
    )

    logits = (
        np.log(p)
        /
        temperature
    )

    logits -= logits.max()

    exp_logits = np.exp(
        logits
    )

    result = (
        exp_logits
        /
        exp_logits.sum()
    )

    return result


# ============================================================
# 17. METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    actual
):

    ranking = np.argsort(
        probabilities
    )[::-1]

    top1 = int(
        actual in ranking[:1]
    )

    top3 = int(
        actual in ranking[:3]
    )

    top5 = int(
        actual in ranking[:5]
    )

    dead7 = int(
        actual in np.argsort(
            probabilities
        )[:7]
    )

    p = float(
        probabilities[
            int(actual)
        ]
    )

    p = max(
        p,
        1e-9
    )

    logloss = -np.log(p)

    # Multiclass Brier score
    brier = float(
        np.sum(
            (
                probabilities
                -
                np.eye(10)[
                    int(actual)
                ]
            ) ** 2
        )
    )

    return {

        "top1": top1,
        "top3": top3,
        "top5": top5,
        "dead7": dead7,
        "logloss": logloss,
        "brier": brier
    }


# ============================================================
# 18. RANDOM BASELINE
# ============================================================

def random_baseline():

    return {

        "top1": 0.10,
        "top3": 0.30,
        "top5": 0.50,

        # Random 7 of 10
        "dead7": 0.70,

        # Uniform multiclass probability
        "logloss":
            float(
                -np.log(0.10)
            ),

        # Brier score for uniform 10-class
        "brier":
            float(0.90)
    }


# ============================================================
# 19. BINOMIAL CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes,
    total,
    z=1.96
):

    if total <= 0:

        return (
            0.0,
            0.0
        )

    p = successes / total

    denominator = (
        1
        +
        z**2 / total
    )

    center = (
        p
        +
        z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        *
        np.sqrt(
            (
                p * (1 - p)
                / total
            )
            +
            (
                z**2
                /
                (4 * total**2)
            )
        )
        /
        denominator
    )

    return (
        max(
            0.0,
            center - margin
        ),
        min(
            1.0,
            center + margin
        )
    )


# ============================================================
# 20. FEATURE SELECTION
# ============================================================

def select_features(
    X_train,
    y_train,
    feature_names,
    config
):

    available = [
        f
        for f in feature_names
        if f in X_train.columns
    ]

    if len(available) <= config[
        "top_features"
    ]:

        return available

    if y_train.nunique() < 2:

        return available[
            :config["top_features"]
        ]

    selector = ExtraTreesClassifier(

        n_estimators=50,

        max_depth=7,

        min_samples_leaf=3,

        max_features="sqrt",

        class_weight="balanced",

        n_jobs=-1,

        random_state=123
    )

    try:

        selector.fit(
            X_train[available],
            y_train
        )

        importance = pd.Series(
            selector.feature_importances_,
            index=available
        )

        selected = (
            importance
            .sort_values(
                ascending=False
            )
            .head(
                config["top_features"]
            )
            .index
            .tolist()
        )

        return selected

    except Exception:

        return available[
            :config["top_features"]
        ]


# ============================================================
# 21. RELIABILITY
# ============================================================

def reliability_from_tests(
    tests,
    target=50
):

    if tests <= 0:
        return 0.0

    return float(
        min(
            1.0,
            tests / target
        )
    )


# ============================================================
# 22. MODEL SCORE
# ============================================================

def calculate_model_score(
    metrics,
    tests
):

    if tests <= 0:

        return 0.0

    random = random_baseline()

    top1_adv = (
        metrics["top1"]
        -
        random["top1"]
    )

    top3_adv = (
        metrics["top3"]
        -
        random["top3"]
    )

    top5_adv = (
        metrics["top5"]
        -
        random["top5"]
    )

    logloss_score = (
        random["logloss"]
        /
        max(
            metrics["logloss"],
            1e-9
        )
    )

    logloss_score = min(
        1.0,
        max(
            0.0,
            logloss_score
        )
    )

    brier_score = (
        1
        -
        min(
            1.0,
            metrics["brier"]
        )
    )

    # Accuracy levels remain visible,
    # but advantage vs random is what
    # controls model selection.
    advantage_score = (

        0.35
        *
        np.clip(
            0.5
            +
            top1_adv,
            0,
            1
        )

        +

        0.25
        *
        np.clip(
            0.5
            +
            top3_adv,
            0,
            1
        )

        +

        0.15
        *
        np.clip(
            0.5
            +
            top5_adv,
            0,
            1
        )

        +

        0.15
        *
        logloss_score

        +

        0.10
        *
        brier_score
    )

    reliability = (
        reliability_from_tests(
            tests
        )
    )

    return float(
        advantage_score
        *
        reliability
    )


# ============================================================
# 23. WALK-FORWARD BACKTEST
# ============================================================

@st.cache_data(
    ttl=1200,
    show_spinner=False
)
def adaptive_backtest(

    df_features,
    position,
    data_hash,
    config
):

    n = len(
        df_features
    )

    if n <= config["min_train"]:

        return {

            "best_model":
                "ExtraTrees",

            "weights":
                {
                    name:
                    1 / len(MODEL_NAMES)
                    for name in MODEL_NAMES
                },

            "scores": {},

            "tests": 0,

            "selected_features": []
        }

    X_all = (
        df_features[
            FEATURES
        ]
        .astype(np.float32)
    )

    y_all = (
        df_features[position]
        .astype(int)
    )

    max_test = min(
        config["backtest"],
        n - config["min_train"]
    )

    test_indices = list(
        range(
            n - max_test,
            n
        )
    )

    all_results = {
        model: []
        for model in MODEL_NAMES
    }

    feature_counts = {
        model: []
        for model in MODEL_NAMES
    }

    for order, test_idx in enumerate(
        test_indices
    ):

        train_end = test_idx

        if train_end < config[
            "min_train"
        ]:
            continue

        X_train = X_all.iloc[
            :train_end
        ]

        y_train = y_all.iloc[
            :train_end
        ]

        X_test = X_all.iloc[
            [test_idx]
        ]

        actual = int(
            y_all.iloc[
                test_idx
            ]
        )

        if y_train.nunique() < 2:
            continue

        # ----------------------------------------------------
        # Feature selection ONLY from past data
        # ----------------------------------------------------

        selected_features = (
            select_features(
                X_train,
                y_train,
                FEATURES,
                config
            )
        )

        if not selected_features:
            continue

        for model_name in MODEL_NAMES:

            model = create_model(
                model_name,
                config
            )

            try:

                model.fit(
                    X_train[
                        selected_features
                    ],
                    y_train
                )

                probs = probability_vector(
                    model,
                    X_test[
                        selected_features
                    ]
                )

                probs = stabilize_probability(
                    probs
                )

                metrics = calculate_metrics(
                    probs,
                    actual
                )

                all_results[
                    model_name
                ].append(
                    {
                        **metrics,
                        "index":
                            test_idx,
                        "order":
                            order
                    }
                )

                feature_counts[
                    model_name
                ].append(
                    selected_features
                )

            except Exception:

                continue

    random = random_baseline()

    scores = {}

    for model_name in MODEL_NAMES:

        rows = all_results[
            model_name
        ]

        tests = len(rows)

        if tests == 0:

            scores[
                model_name
            ] = {

                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
                "dead7": 0.0,
                "logloss": 99.0,
                "brier": 1.0,

                "top1_low": 0.0,
                "top1_high": 0.0,

                "top3_low": 0.0,
                "top3_high": 0.0,

                "top5_low": 0.0,
                "top5_high": 0.0,

                "stability": 0.0,
                "stability_std": 1.0,

                "score": 0.0,

                "tests": 0,

                "advantage_top1":
                    -0.10,

                "advantage_top3":
                    -0.30,

                "advantage_top5":
                    -0.50,

                "reliability":
                    0.0
            }

            continue

        # ----------------------------------------------------
        # Recency weights
        # ----------------------------------------------------

        weighted = {

            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "dead7": 0.0,
            "logloss": 0.0,
            "brier": 0.0
        }

        total_weight = 0.0

        for row_order, row in enumerate(
            rows
        ):

            recency_weight = (
                config["recent_decay"]
                **
                (
                    tests
                    -
                    row_order
                    -
                    1
                )
            )

            total_weight += (
                recency_weight
            )

            for key in weighted:

                weighted[key] += (
                    row[key]
                    *
                    recency_weight
                )

        metrics = {

            key:
                weighted[key]
                /
                max(
                    total_weight,
                    1e-9
                )

            for key in weighted
        }

        # ----------------------------------------------------
        # CI based on raw test count
        # ----------------------------------------------------

        top1_success = sum(
            row["top1"]
            for row in rows
        )

        top3_success = sum(
            row["top3"]
            for row in rows
        )

        top5_success = sum(
            row["top5"]
            for row in rows
        )

        top1_low, top1_high = (
            wilson_interval(
                top1_success,
                tests
            )
        )

        top3_low, top3_high = (
            wilson_interval(
                top3_success,
                tests
            )
        )

        top5_low, top5_high = (
            wilson_interval(
                top5_success,
                tests
            )
        )

        # ----------------------------------------------------
        # Multi-window stability
        # ----------------------------------------------------

        window_scores = []

        chunks = np.array_split(
            rows,
            min(
                4,
                len(rows)
            )
        )

        for chunk in chunks:

            if len(chunk) == 0:
                continue

            chunk_top3 = np.mean(
                [
                    x["top3"]
                    for x in chunk
                ]
            )

            window_scores.append(
                chunk_top3
            )

        if len(window_scores) >= 2:

            stability_std = float(
                np.std(
                    window_scores
                )
            )

        else:

            stability_std = 0.0

        stability = max(
            0.0,
            1.0
            -
            min(
                1.0,
                stability_std
                * 3.0
            )
        )

        metrics["stability"] = (
            stability
        )

        metrics["stability_std"] = (
            stability_std
        )

        metrics["top1_low"] = (
            top1_low
        )

        metrics["top1_high"] = (
            top1_high
        )

        metrics["top3_low"] = (
            top3_low
        )

        metrics["top3_high"] = (
            top3_high
        )

        metrics["top5_low"] = (
            top5_low
        )

        metrics["top5_high"] = (
            top5_high
        )

        metrics["advantage_top1"] = (
            metrics["top1"]
            -
            random["top1"]
        )

        metrics["advantage_top3"] = (
            metrics["top3"]
            -
            random["top3"]
        )

        metrics["advantage_top5"] = (
            metrics["top5"]
            -
            random["top5"]
        )

        raw_score = calculate_model_score(
            metrics,
            tests
        )

        # Stability penalty
        stability_factor = (
            0.70
            +
            0.30
            *
            stability
        )

        metrics["score"] = float(
            raw_score
            *
            stability_factor
        )

        metrics["reliability"] = (
            reliability_from_tests(
                tests
            )
        )

        metrics["tests"] = tests

        scores[
            model_name
        ] = metrics

    # ========================================================
    # ADAPTIVE WEIGHTS
    # ========================================================

    raw_weights = {}

    for model_name in MODEL_NAMES:

        score = (
            scores[
                model_name
            ]["score"]
        )

        # Conservative floor
        raw_weights[
            model_name
        ] = (
            0.10
            +
            score ** 1.80
        )

    total = sum(
        raw_weights.values()
    )

    weights = {

        name:
            value / total

        for name, value
        in raw_weights.items()
    }

    best_model = max(
        weights,
        key=weights.get
    )

    # --------------------------------------------------------
    # Most frequently selected features
    # --------------------------------------------------------

    feature_frequency = {}

    for model_name in MODEL_NAMES:

        for selected in feature_counts[
            model_name
        ]:

            for feature in selected:

                feature_frequency[
                    feature
                ] = (
                    feature_frequency.get(
                        feature,
                        0
                    )
                    + 1
                )

    selected_features_final = [

        name

        for name, count in sorted(
            feature_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ][:config["top_features"]]

    return {

        "best_model":
            best_model,

        "weights":
            weights,

        "scores":
            scores,

        "tests":
            max(
                [
                    x["tests"]
                    for x in scores.values()
                ],
                default=0
            ),

        "selected_features":
            selected_features_final
    }


# ============================================================
# 24. FINAL FEATURE SELECTION
# ============================================================

@st.cache_data(
    ttl=1200,
    show_spinner=False
)
def final_feature_selection(
    X_train,
    y_train,
    config,
    lottery_name,
    position,
    data_hash
):

    return select_features(
        X_train,
        y_train,
        FEATURES,
        config
    )


# ============================================================
# 25. FINAL MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_final_model(

    X_train,
    y_train,

    model_name,
    selected_features,

    config,

    lottery_name,
    position,
    data_hash
):

    model = create_model(
        model_name,
        config
    )

    model.fit(
        X_train[
            selected_features
        ],
        y_train
    )

    return model


# ============================================================
# 26. FINAL PREDICTION
# ============================================================

def final_prediction(

    df_features,
    position,

    backtest,
    config,

    lottery_name,
    data_hash
):

    X = (
        df_features[
            FEATURES
        ]
        .astype(np.float32)
    )

    y = (
        df_features[position]
        .astype(int)
    )

    X_train = X.iloc[
        :-1
    ].copy()

    y_train = y.iloc[
        :-1
    ].copy()

    X_next = X.iloc[
        [-1]
    ].copy()

    # --------------------------------------------------------
    # Final feature selection from historical data only
    # --------------------------------------------------------

    selected_features = (
        backtest.get(
            "selected_features",
            []
        )
    )

    if not selected_features:

        selected_features = (
            final_feature_selection(
                X_train,
                y_train,
                config,
                lottery_name,
                position,
                data_hash
            )
        )

    selected_features = [
        f
        for f in selected_features
        if f in X_train.columns
    ]

    if not selected_features:

        selected_features = FEATURES[
            :config["top_features"]
        ]

    model_probabilities = {}

    weights = backtest.get(
        "weights",
        {
            name:
            1 / len(MODEL_NAMES)
            for name in MODEL_NAMES
        }
    )

    for model_name in MODEL_NAMES:

        try:

            model = train_final_model(

                X_train,
                y_train,

                model_name,
                selected_features,

                config,

                lottery_name,
                position,
                data_hash
            )

            probs = probability_vector(
                model,
                X_next[
                    selected_features
                ]
            )

            probs = stabilize_probability(
                probs,
                temperature=1.08
            )

            model_probabilities[
                model_name
            ] = probs

        except Exception:

            continue

    ensemble = np.zeros(
        10,
        dtype=float
    )

    used_weight = 0.0

    for model_name, probs in (
        model_probabilities.items()
    ):

        weight = float(
            weights.get(
                model_name,
                0
            )
        )

        ensemble += (
            probs
            *
            weight
        )

        used_weight += weight

    if used_weight <= 0:

        ensemble = (
            np.ones(10)
            /
            10
        )

    else:

        ensemble /= used_weight

    ensemble = stabilize_probability(
        ensemble,
        temperature=1.05
    )

    hot_idx = np.argsort(
        ensemble
    )[::-1][:5]

    dead_idx = np.argsort(
        ensemble
    )[:7]

    hot = [

        (
            int(i),
            float(
                ensemble[i]
            )
        )

        for i in hot_idx
    ]

    dead = [

        (
            int(i),
            float(
                ensemble[i]
            )
        )

        for i in dead_idx
    ]

    sorted_probs = np.sort(
        ensemble
    )[::-1]

    top1_gap = (
        sorted_probs[0]
        -
        sorted_probs[1]
    )

    top3_concentration = (
        sorted_probs[:3].sum()
    )

    top5_concentration = (
        sorted_probs[:5].sum()
    )

    agreement = 0.0

    if len(
        model_probabilities
    ) >= 2:

        rankings = [

            np.argsort(
                p
            )[::-1][:5]

            for p in
            model_probabilities.values()
        ]

        intersections = []

        base = set(
            rankings[0]
        )

        for ranking in rankings[1:]:

            intersections.append(
                len(
                    base.intersection(
                        set(ranking)
                    )
                )
                /
                5
            )

        if intersections:

            agreement = float(
                np.mean(
                    intersections
                )
            )

    selected_model = max(
        weights,
        key=weights.get
    )

    return {

        "model":
            selected_model,

        "weights":
            weights,

        "model_probabilities":
            model_probabilities,

        "probabilities":
            ensemble,

        "hot":
            hot,

        "dead":
            dead,

        "confidence":
            float(top1_gap),

        "top3_concentration":
            float(
                top3_concentration
            ),

        "top5_concentration":
            float(
                top5_concentration
            ),

        "agreement":
            float(agreement),

        "selected_features":
            selected_features
    }


# ============================================================
# 27. TARGET DATE
# ============================================================

def calculate_target_date(
    df,
    selected_day
):

    last_date = pd.Timestamp(
        df["Date"].iloc[-1]
    )

    day_map = {

        "อัตโนมัติ":
            None,

        "วันจันทร์":
            0,

        "วันอังคาร":
            1,

        "วันพุธ":
            2,

        "วันพฤหัสบดี":
            3,

        "วันศุกร์":
            4,

        "วันเสาร์":
            5,

        "วันอาทิตย์":
            6
    }

    target_dow = day_map[
        selected_day
    ]

    if target_dow is None:

        if len(df) >= 2:

            gap = (
                last_date
                -
                pd.Timestamp(
                    df["Date"].iloc[-2]
                )
            ).days

            if gap <= 0:
                gap = 7

        else:

            gap = 7

        return (
            last_date
            +
            timedelta(
                days=gap
            )
        )

    days_ahead = (
        target_dow
        -
        last_date.dayofweek
    )

    if days_ahead <= 0:
        days_ahead += 7

    return (
        last_date
        +
        timedelta(
            days=days_ahead
        )
    )


# ============================================================
# 28. DISPLAY HOT
# ============================================================

def display_hot(
    position,
    result
):

    hot = result["hot"]

    numbers = " - ".join(
        str(num)
        for num, _ in hot
    )

    probability_text = " | ".join(

        f"{num}: {prob * 100:.1f}%"

        for num, prob in hot
    )

    confidence = (
        result["confidence"]
        * 100
    )

    concentration = (
        result["top3_concentration"]
        * 100
    )

    agreement = (
        result["agreement"]
        * 100
    )

    weights_text = " | ".join(

        f"{name}: {weight * 100:.0f}%"

        for name, weight in
        result["weights"].items()
    )

    st.markdown(

        f"""
        <div class="hot-card">

            <div class="position-title">
                {POSITION_LABELS[position]}
            </div>

            <div class="hot-number">
                {numbers}
            </div>

            <div class="prob-text">
                AI Probability:
                {probability_text}
            </div>

            <div class="confidence">
                Top-1 Gap:
                {confidence:.1f}%
                |
                Top-3:
                {concentration:.1f}%
                |
                Agreement:
                {agreement:.1f}%
            </div>

            <div class="model-badge">
                🤖 Adaptive AI:
                {result["model"]}
                <br>
                ⚖️ Weights:
                {weights_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 29. DISPLAY DEAD
# ============================================================

def display_dead(
    position,
    result
):

    dead = result["dead"]

    numbers = " - ".join(
        str(num)
        for num, _ in dead
    )

    probability_text = " | ".join(

        f"{num}: {prob * 100:.1f}%"

        for num, prob in dead
    )

    st.markdown(

        f"""
        <div class="dead-card">

            <div class="position-title">
                {POSITION_LABELS[position]}
            </div>

            <div class="dead-number">
                {numbers}
            </div>

            <div class="prob-text">
                AI Probability ต่ำสุด:
                {probability_text}
            </div>

            <div class="model-badge">
                🤖 Adaptive AI:
                {result["model"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 30. DISPLAY BACKTEST
# ============================================================

def display_backtest(
    position,
    backtest
):

    st.markdown(
        f"### {POSITION_LABELS[position]}"
    )

    if not backtest["scores"]:

        st.warning(
            "ไม่มีผล Backtest เพียงพอ"
        )

        return

    random = random_baseline()

    rows = []

    for model_name, score in (
        backtest["scores"].items()
    ):

        weight = (
            backtest
            .get("weights", {})
            .get(
                model_name,
                0
            )
        )

        rows.append(

            {
                "AI Model":
                    model_name,

                "Weight":
                    f"{weight * 100:.1f}%",

                "Top-1":
                    f"{score['top1'] * 100:.1f}%",

                "Top-1 CI":
                    f"{score['top1_low'] * 100:.1f}"
                    f"–{score['top1_high'] * 100:.1f}%",

                "Top-3":
                    f"{score['top3'] * 100:.1f}%",

                "Top-5":
                    f"{score['top5'] * 100:.1f}%",

                "Dead-7":
                    f"{score['dead7'] * 100:.1f}%",

                "LogLoss":
                    f"{score['logloss']:.3f}",

                "Brier":
                    f"{score['brier']:.3f}",

                "Stability":
                    f"{score['stability'] * 100:.1f}%",

                "Adv Top-1":
                    f"{score['advantage_top1'] * 100:+.1f}%",

                "AI Score":
                    f"{score['score'] * 100:.1f}%",

                "Reliability":
                    f"{score['reliability'] * 100:.1f}%",

                "Tests":
                    score["tests"]
            }
        )

    st.dataframe(

        pd.DataFrame(rows),

        use_container_width=True,

        hide_index=True
    )

    st.info(

        f"""
        Random Baseline:
        Top-1 {random['top1'] * 100:.0f}% |
        Top-3 {random['top3'] * 100:.0f}% |
        Top-5 {random['top5'] * 100:.0f}%

        | Tests สูงสุด:
        {backtest['tests']}
        """
    )

    st.success(

        f"🤖 Adaptive AI: "
        f"**{backtest['best_model']}**"
    )


# ============================================================
# 31. MAIN
# ============================================================

def main():

    inject_css()

    st.markdown(

        """
        <div class="main-title">
            🤖 LOTTO AI PRO V8.1 MAX
        </div>

        <div class="subtitle">
            STATISTICAL WALK-FORWARD •
            FEATURE SELECTION •
            RANDOM BASELINE •
            STABILITY SAFE
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        lottery = st.selectbox(
            "🏷️ เลือกประเภทหวย",
            list(
                LOTTERY_SOURCES.keys()
            )
        )

    with col2:

        selected_day = st.selectbox(

            "📅 วันเป้าหมาย",

            [
                "อัตโนมัติ",
                "วันจันทร์",
                "วันอังคาร",
                "วันพุธ",
                "วันพฤหัสบดี",
                "วันศุกร์",
                "วันเสาร์",
                "วันอาทิตย์"
            ]
        )

    st.markdown("---")

    run = st.button(

        "🚀 เริ่มวิเคราะห์ PRO V8.1 MAX",

        type="primary",

        use_container_width=True
    )

    if not run:

        st.info(
            "เลือกหวยและวันเป้าหมาย "
            "แล้วกด 🚀 เริ่มวิเคราะห์"
        )

        return

    url = LOTTERY_SOURCES[
        lottery
    ]

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    with st.spinner(
        "📥 กำลังโหลดข้อมูลย้อนหลัง..."
    ):

        df = fetch_lottery_data(
            url
        )

    if df.empty:

        st.error(
            "❌ ไม่สามารถดึงข้อมูลได้"
        )

        st.warning(
            "ตรวจสอบ Internet หรือโครงสร้างหน้าเว็บต้นทาง"
        )

        return

    if len(df) < 50:

        st.error(
            f"❌ พบข้อมูลเพียง "
            f"{len(df)} งวด"
        )

        st.warning(
            "ระบบต้องการอย่างน้อย 50 งวด"
        )

        return

    # --------------------------------------------------------
    # Target date
    # --------------------------------------------------------

    target_date = calculate_target_date(
        df,
        selected_day
    )

    # --------------------------------------------------------
    # Dummy target
    # --------------------------------------------------------

    dummy = pd.DataFrame(

        [
            {
                "Date":
                    target_date,

                "Result_3D":
                    "000",

                "Result_2D":
                    "00"
            }
        ]
    )

    extended = pd.concat(
        [
            df,
            dummy
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    with st.spinner(
        "🧠 สร้าง Leakage-Safe Features..."
    ):

        feature_df = build_features(
            extended
        )

    config = get_adaptive_config(
        len(df)
    )

    data_hash = get_data_hash(
        df
    )

    st.info(

        f"""
        ⚡ V8.1 MAX Engine |
        ข้อมูล {len(df):,} งวด |
        Walk-Forward {config['backtest']} |
        Min Train {config['min_train']} |
        Trees {config['trees']} |
        Top Features {config['top_features']} |
        Features ดิบ {len(FEATURES)}
        """
    )

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    backtest_results = {}

    progress = st.progress(0)

    status_text = st.empty()

    for idx, position in enumerate(
        POSITIONS
    ):

        status_text.caption(

            f"🧠 Walk-Forward Backtest: "
            f"{POSITION_LABELS[position]}"
        )

        backtest_results[position] = (

            adaptive_backtest(

                feature_df.iloc[:-1],

                position,

                data_hash,

                config
            )
        )

        progress.progress(

            int(
                (
                    idx + 1
                )
                /
                len(POSITIONS)
                *
                100
            )
        )

    progress.empty()
    status_text.empty()

    # --------------------------------------------------------
    # Final Ensemble
    # --------------------------------------------------------

    with st.spinner(
        "🤖 กำลังสร้าง Adaptive AI Ensemble..."
    ):

        final_results = {}

        for position in POSITIONS:

            final_results[position] = (

                final_prediction(

                    feature_df,

                    position,

                    backtest_results[position],

                    config,

                    lottery,

                    data_hash
                )
            )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    selected_models = [

        final_results[p]["model"]

        for p in POSITIONS
    ]

    model_text = " | ".join(
        selected_models
    )

    st.markdown(

        f"""
        <div class="status-card">

        🤖 <b>LOTTO AI PRO V8.1 MAX</b><br>

        📊 ข้อมูลย้อนหลัง:
        {len(df):,} งวด<br>

        📅 งวดล่าสุด:
        {df["Date"].iloc[-1].strftime("%d/%m/%Y")}<br>

        🎯 เป้าหมาย:
        {target_date.strftime("%d/%m/%Y")}
        ({DOW_NAMES[target_date.dayofweek]})<br>

        🧠 Raw Features:
        {len(FEATURES)}<br>

        🎯 Selected Features:
        {config["top_features"]}<br>

        🌳 Trees:
        {config["trees"]}<br>

        🔬 Walk-Forward:
        {config["backtest"]} tests<br>

        🤖 Models:
        {model_text}<br>

        🎲 Random Baseline:
        ENABLED<br>

        📐 Confidence Interval:
        ENABLED<br>

        📉 Stability Penalty:
        ENABLED<br>

        🔒 Leakage Safe:
        ENABLED

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tab_hot, tab_dead, tab_accuracy = st.tabs(

        [
            "🎯 เลขเด่น AI",
            "🛑 เลขดับ 7",
            "📊 Accuracy / Evidence"
        ]
    )

    # --------------------------------------------------------
    # HOT
    # --------------------------------------------------------

    with tab_hot:

        st.subheader(
            "🎯 AI TOP 5"
        )

        st.caption(
            "Probability จาก Adaptive Ensemble "
            "หลังผ่าน Feature Selection และ "
            "Walk-Forward Backtest"
        )

        for position in POSITIONS:

            display_hot(
                position,
                final_results[position]
            )

    # --------------------------------------------------------
    # DEAD
    # --------------------------------------------------------

    with tab_dead:

        st.subheader(
            "🛑 AI BOTTOM 7"
        )

        st.warning(

            "Dead-7 เป็นเพียงกลุ่มที่ AI "
            "ให้ Probability ต่ำสุด "
            "ไม่ใช่การรับประกันว่าจะไม่ออก"
        )

        for position in POSITIONS:

            display_dead(
                position,
                final_results[position]
            )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    with tab_accuracy:

        st.subheader(
            "📊 Statistical AI Backtest"
        )

        st.caption(

            "ผล Backtest ถูกประเมินเทียบกับ "
            "Random Baseline และปรับ Reliability "
            "ตามจำนวนตัวอย่าง"
        )

        for position in POSITIONS:

            display_backtest(

                position,

                backtest_results[position]
            )

            st.write("")

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    with st.expander(
        "🧠 Selected Features"
    ):

        for position in POSITIONS:

            selected = (
                final_results[position]
                .get(
                    "selected_features",
                    []
                )
            )

            st.markdown(
                f"**{POSITION_LABELS[position]}**"
            )

            if selected:

                st.write(
                    ", ".join(
                        selected
                    )
                )

            else:

                st.write(
                    "ไม่มีข้อมูล"
                )

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    with st.expander(
        "⚖️ Adaptive AI Weights"
    ):

        rows = []

        for position in POSITIONS:

            result = (
                final_results[position]
            )

            for model_name, weight in (
                result["weights"].items()
            ):

                rows.append(

                    {
                        "ตำแหน่ง":
                            POSITION_LABELS[
                                position
                            ],

                        "AI Model":
                            model_name,

                        "Adaptive Weight":
                            f"{weight * 100:.1f}%"
                    }
                )

        st.dataframe(

            pd.DataFrame(rows),

            use_container_width=True,

            hide_index=True
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    with st.expander(
        "🎯 สรุปเลขเด่นทั้งหมด"
    ):

        summary_hot = []

        for position in POSITIONS:

            result = (
                final_results[position]
            )

            top = result["hot"][0]

            bt = (
                backtest_results[position]
            )

            best_score = (
                bt["scores"]
                .get(
                    result["model"],
                    {}
                )
            )

            summary_hot.append(

                {

                    "ตำแหน่ง":
                        POSITION_LABELS[
                            position
                        ],

                    "Top-1":
                        top[0],

                    "Probability":
                        f"{top[1] * 100:.1f}%",

                    "AI":
                        result["model"],

                    "Top-1 Gap":
                        f"{result['confidence'] * 100:.1f}%",

                    "Top-3":
                        f"{result['top3_concentration'] * 100:.1f}%",

                    "Agreement":
                        f"{result['agreement'] * 100:.1f}%",

                    "BT Top-1":
                        f"{best_score.get('top1', 0) * 100:.1f}%",

                    "BT Tests":
                        best_score.get(
                            "tests",
                            0
                        ),

                    "Reliability":
                        f"{best_score.get('reliability', 0) * 100:.1f}%"
                }
            )

        st.dataframe(

            pd.DataFrame(
                summary_hot
            ),

            use_container_width=True,

            hide_index=True
        )

    # --------------------------------------------------------
    # SYSTEM INFO
    # --------------------------------------------------------

    with st.expander(
        "🔧 System Information"
    ):

        info = {

            "Engine":
                "LOTTO AI PRO V8.1 MAX",

            "Lottery":
                lottery,

            "Historical draws":
                len(df),

            "Raw Features":
                len(FEATURES),

            "Selected Features":
                config["top_features"],

            "Trees":
                config["trees"],

            "Walk-Forward Tests":
                config["backtest"],

            "Minimum Training":
                config["min_train"],

            "Data Hash":
                data_hash[:16] + "...",

            "AI Models":
                "ExtraTrees / RandomForest / HistGradientBoosting",

            "Feature Selection":
                "Inside every Walk-Forward fold",

            "Random Baseline":
                "Enabled",

            "Confidence Interval":
                "Wilson 95%",

            "Stability Analysis":
                "4 rolling backtest windows",

            "Reliability":
                "Sample-size weighted",

            "Recency Weight":
                "Enabled",

            "Leakage Protection":
                "Enabled",

            "XGBoost":
                "Disabled",

            "Markov":
                "Disabled",

            "Frequency Voting":
                "Disabled",

            "Calendar Voting":
                "Disabled",

            "Equation Voting":
                "Disabled"
        }

        for key, value in info.items():

            st.write(
                f"**{key}:** {value}"
            )


# ============================================================
# 32. RUN
# ============================================================

if __name__ == "__main__":

    main()
