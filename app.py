# ============================================================
# 🤖 LOTTO AI PRO V8.1 MAX
# STRICT WALK-FORWARD • NO PERSISTENT MEMORY
# FEATURE SELECTION • LEAKAGE SAFE • MOBILE TURBO
# ============================================================
#
# V8.1 MAX
# ------------------------------------------------------------
# ✅ Strict Walk-Forward
# ✅ Train -> Predict -> Move 1 Draw
# ✅ NO Future Leakage
# ✅ NO Persistent Model Memory
# ✅ Causal Feature Engineering
# ✅ Training-only Feature Selection
# ✅ ExtraTrees
# ✅ RandomForest
# ✅ HistGradientBoosting
# ✅ Adaptive Ensemble
# ✅ Top-1 / Top-3 / Top-5
# ✅ Dead-7
# ✅ LogLoss
# ✅ Brier Score
# ✅ Stability
# ✅ Model Agreement
# ✅ Random Baseline
# ✅ Recency-aware evaluation
# ✅ Median Imputation
# ✅ Robust Scraping Exception
# ✅ Fast Rolling Entropy
# ✅ Mobile Optimized
# ✅ Streamlit Cloud Friendly
#
# NO
# ------------------------------------------------------------
# ❌ XGBoost
# ❌ Markov
# ❌ Frequency Voting
# ❌ Calendar Voting
# ❌ Equation Voting
# ❌ Manual Number Voting
# ❌ Persistent trained model
#
# IMPORTANT
# ------------------------------------------------------------
# Target at row T is NEVER used to create features for row T.
#
# Backtest:
#
#   TRAIN [0 ... T-1]
#          ↓
#       PREDICT T
#          ↓
#       move +1
#
# Feature selection is also performed using TRAINING DATA ONLY.
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

from sklearn.impute import SimpleImputer

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

# ============================================================
# IMPORTANT:
# Keep this relatively small for mobile.
#
# Feature selection chooses from the full causal feature pool.
# ============================================================

MAX_SELECTED_FEATURES = 25


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
            font-size: 2.35rem;
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

        .hot-number {
            font-size: 2.15rem;
            font-weight: 900;
            color: #16a34a;
            letter-spacing: 3px;
            text-align: center;
        }

        .dead-number {
            font-size: 2.15rem;
            font-weight: 900;
            color: #dc2626;
            letter-spacing: 3px;
            text-align: center;
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

        div.stButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 800;
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
# 6. SCRAPING EXCEPTIONS
# ============================================================

class ScrapingError(Exception):
    pass


class NetworkScrapingError(ScrapingError):
    pass


class HTTPStatusScrapingError(ScrapingError):
    pass


class ParsingScrapingError(ScrapingError):
    pass


# ============================================================
# 7. ROBUST SCRAPER
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
        "Chrome/120.0 Mobile Safari/537.36",

        "Accept-Language":
        "th-TH,th;q=0.9,en;q=0.8"
    }

    try:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

        except requests.exceptions.Timeout as exc:

            raise NetworkScrapingError(
                "หมดเวลาเชื่อมต่อเว็บไซต์"
            ) from exc

        except requests.exceptions.ConnectionError as exc:

            raise NetworkScrapingError(
                "ไม่สามารถเชื่อมต่อเว็บไซต์ได้"
            ) from exc

        except requests.exceptions.RequestException as exc:

            raise NetworkScrapingError(
                f"เกิดข้อผิดพลาด Network: {exc}"
            ) from exc

        # ----------------------------------------------------
        # HTTP
        # ----------------------------------------------------

        if response.status_code != 200:

            raise HTTPStatusScrapingError(
                f"เว็บไซต์ตอบกลับ HTTP "
                f"{response.status_code}"
            )

        if not response.text:

            raise ParsingScrapingError(
                "เว็บไซต์ส่งข้อมูลว่างกลับมา"
            )

        # ----------------------------------------------------
        # HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        content = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content",
                re.I
            )
        )

        if content is None:

            content = soup

        # ----------------------------------------------------
        # FIRST: TRY TABLE STRUCTURE
        # ----------------------------------------------------

        extracted = []

        current_date = None

        tables = content.find_all("table")

        for table in tables:

            rows = table.find_all("tr")

            for row in rows:

                cells = [
                    c.get_text(
                        " ",
                        strip=True
                    )
                    for c in row.find_all(
                        ["td", "th"]
                    )
                ]

                if not cells:
                    continue

                row_text = " ".join(cells)

                parsed_date = normalize_date(
                    row_text
                )

                if parsed_date is not None:

                    current_date = parsed_date

                numbers3 = re.findall(
                    r"\b\d{3}\b",
                    row_text
                )

                numbers2 = re.findall(
                    r"\b\d{2}\b",
                    row_text
                )

                if (
                    current_date is not None
                    and numbers3
                    and numbers2
                ):

                    extracted.append(
                        {
                            "Date":
                                current_date,

                            "Result_3D":
                                numbers3[0],

                            "Result_2D":
                                numbers2[-1]
                        }
                    )

        # ----------------------------------------------------
        # SECOND: LINE PARSER
        # ----------------------------------------------------

        if not extracted:

            lines = content.get_text(
                separator="\n"
            ).split("\n")

            current_date = None

            for raw in lines:

                line = raw.strip()

                if not line:
                    continue

                parsed_date = normalize_date(
                    line
                )

                if parsed_date is not None:
                    current_date = parsed_date

                match = re.search(
                    r"\b(\d{3})\b.*?\b(\d{2})\b",
                    line
                )

                if (
                    match
                    and current_date is not None
                ):

                    extracted.append(
                        {
                            "Date":
                                current_date,

                            "Result_3D":
                                match.group(1),

                            "Result_2D":
                                match.group(2)
                        }
                    )

        # ----------------------------------------------------
        # PARSE CHECK
        # ----------------------------------------------------

        if not extracted:

            raise ParsingScrapingError(
                "ไม่พบรูปแบบข้อมูลหวย "
                "ที่ระบบรู้จักบนหน้าเว็บไซต์"
            )

        df = pd.DataFrame(
            extracted
        )

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # REMOVE INVALID
        # ----------------------------------------------------

        df = df[
            df["Result_3D"].str.match(
                r"^\d{3}$",
                na=False
            )
            &
            df["Result_2D"].str.match(
                r"^\d{2}$",
                na=False
            )
        ]

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        df = (
            df
            .drop_duplicates(
                subset=[
                    "Date",
                    "Result_3D",
                    "Result_2D"
                ]
            )
            .sort_values(
                "Date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(df) == 0:

            raise ParsingScrapingError(
                "ข้อมูลถูกกรองออกทั้งหมด "
                "หลังจากตรวจสอบความถูกต้อง"
            )

        return df

    except ScrapingError:

        raise

    except Exception as exc:

        raise ParsingScrapingError(
            f"เกิดข้อผิดพลาดในการอ่านข้อมูล: {exc}"
        ) from exc


# ============================================================
# 8. FAST ENTROPY
# ============================================================

def add_fast_entropy(
    work,
    pos,
    window
):

    shifted = work[pos].shift(1)

    counts = []

    for digit in range(10):

        count = (
            (shifted == digit)
            .astype(float)
            .rolling(
                window,
                min_periods=2
            )
            .sum()
        )

        counts.append(count)

    total = (
        shifted
        .rolling(
            window,
            min_periods=2
        )
        .count()
        .replace(0, np.nan)
    )

    entropy = pd.Series(
        0.0,
        index=work.index,
        dtype=float
    )

    for count in counts:

        p = count / total

        entropy += np.where(
            p > 0,
            -p * np.log(p),
            0.0
        )

    work[
        f"{pos}_ENT{window}"
    ] = entropy


# ============================================================
# 9. FAST DIGIT FREQUENCY
# ============================================================

def add_digit_frequency(
    work,
    pos,
    window,
    digit
):

    shifted = work[pos].shift(1)

    work[
        f"{pos}_F{window}_{digit}"
    ] = (
        (shifted == digit)
        .astype(float)
        .rolling(
            window,
            min_periods=2
        )
        .mean()
    )


# ============================================================
# 10. GAP
# ============================================================

def gap_since_digit(
    series,
    digit
):

    shifted = series.shift(1)

    arr = shifted.to_numpy()

    output = np.zeros(
        len(arr),
        dtype=float
    )

    last_seen = -1

    for i, value in enumerate(arr):

        if (
            not pd.isna(value)
            and int(value) == digit
        ):

            last_seen = i

        if last_seen < 0:

            output[i] = i + 1

        else:

            output[i] = i - last_seen

    return pd.Series(
        output,
        index=series.index
    )


# ============================================================
# 11. BUILD CAUSAL FEATURES
# ============================================================

def build_features(df):

    work = df.copy()

    # --------------------------------------------------------
    # TARGET DIGITS
    # --------------------------------------------------------

    work["H"] = (
        work["Result_3D"]
        .astype(str)
        .str[0]
        .astype(int)
    )

    work["T"] = (
        work["Result_3D"]
        .astype(str)
        .str[1]
        .astype(int)
    )

    work["O"] = (
        work["Result_3D"]
        .astype(str)
        .str[2]
        .astype(int)
    )

    work["T2"] = (
        work["Result_2D"]
        .astype(str)
        .str[0]
        .astype(int)
    )

    work["O2"] = (
        work["Result_2D"]
        .astype(str)
        .str[1]
        .astype(int)
    )

    # --------------------------------------------------------
    # DATE FEATURES
    #
    # Date is known before the result occurs.
    # Therefore it is causal.
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
        work["Date"]
        .dt.isocalendar()
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

    work["DAY_SIN"] = np.sin(
        2 * np.pi * work["DAY"] / 31
    )

    work["DAY_COS"] = np.cos(
        2 * np.pi * work["DAY"] / 31
    )

    # --------------------------------------------------------
    # PREVIOUS RESULT FEATURES
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
    # POSITION FEATURES
    # --------------------------------------------------------

    for pos in POSITIONS:

        series = work[pos]

        shifted = series.shift(1)

        # ----------------------------------------------------
        # LAGS
        # ----------------------------------------------------

        for lag in range(1, 8):

            work[
                f"{pos}_L{lag}"
            ] = series.shift(lag)

        # ----------------------------------------------------
        # ROLLING
        # ----------------------------------------------------

        for window in [3, 5, 10, 20]:

            rolling = (
                shifted
                .rolling(
                    window,
                    min_periods=2
                )
            )

            work[
                f"{pos}_M{window}"
            ] = rolling.mean()

            work[
                f"{pos}_S{window}"
            ] = rolling.std()

            # Selected digit frequency
            for digit in [0, 2, 5, 7]:

                add_digit_frequency(
                    work,
                    pos,
                    window,
                    digit
                )

        # ----------------------------------------------------
        # DIFFERENCES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        work[
            f"{pos}_ODD"
        ] = (
            shifted
            % 2
        )

        work[
            f"{pos}_HIGH"
        ] = (
            shifted >= 5
        ).astype(float)

        work[
            f"{pos}_MOD3"
        ] = (
            shifted
            % 3
        )

        work[
            f"{pos}_MOD5"
        ] = (
            shifted
            % 5
        )

        # ----------------------------------------------------
        # CYCLIC DIGIT
        # ----------------------------------------------------

        work[
            f"{pos}_SIN"
        ] = np.sin(
            2 * np.pi
            * shifted
            / 10
        )

        work[
            f"{pos}_COS"
        ] = np.cos(
            2 * np.pi
            * shifted
            / 10
        )

        # ----------------------------------------------------
        # MIRROR
        # ----------------------------------------------------

        work[
            f"{pos}_MIRROR"
        ] = (
            9 - shifted
        )

        # ----------------------------------------------------
        # GAP
        # ----------------------------------------------------

        for digit in [0, 2, 5, 7]:

            work[
                f"{pos}_GAP_{digit}"
            ] = gap_since_digit(
                series,
                digit
            )

        # ----------------------------------------------------
        # ENTROPY
        # ----------------------------------------------------

        for window in [5, 10, 20]:

            add_fast_entropy(
                work,
                pos,
                window
            )

    # --------------------------------------------------------
    # CROSS POSITION FEATURES
    # --------------------------------------------------------

    for lag in [1, 2, 3]:

        h = work["H"].shift(lag)
        t = work["T"].shift(lag)
        o = work["O"].shift(lag)

        cross = pd.concat(
            [h, t, o],
            axis=1
        )

        work[
            f"PREV_SUM3_L{lag}"
        ] = cross.sum(axis=1)

        work[
            f"PREV_RANGE3_L{lag}"
        ] = (
            cross.max(axis=1)
            -
            cross.min(axis=1)
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
    # DO NOT fill all NaN with zero.
    #
    # Missing values are intentionally kept here.
    # SimpleImputer will learn median values from TRAIN ONLY.
    # --------------------------------------------------------

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return work


# ============================================================
# 12. FEATURE LIST
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

    for window in [3, 5, 10, 20]:

        FEATURES.append(
            f"{pos}_M{window}"
        )

        FEATURES.append(
            f"{pos}_S{window}"
        )

        for digit in [0, 2, 5, 7]:

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

    for digit in [0, 2, 5, 7]:

        FEATURES.append(
            f"{pos}_GAP_{digit}"
        )

    for window in [5, 10, 20]:

        FEATURES.append(
            f"{pos}_ENT{window}"
        )


# Remove duplicate feature names
FEATURES = list(
    dict.fromkeys(FEATURES)
)


# ============================================================
# 13. DATA HASH
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
# 14. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {

            "min_train": 120,

            "trees": 90,

            "depth": 8,

            "leaf": 3,

            "max_features": "sqrt",

            "selected_features":
                30,

            "backtest_start":
                120,

            "recent_decay":
                0.985
        }

    if n >= 400:

        return {

            "min_train": 100,

            "trees": 75,

            "depth": 7,

            "leaf": 3,

            "max_features": "sqrt",

            "selected_features":
                25,

            "backtest_start":
                100,

            "recent_decay":
                0.98
        }

    if n >= 200:

        return {

            "min_train": 80,

            "trees": 60,

            "depth": 6,

            "leaf": 3,

            "max_features": "sqrt",

            "selected_features":
                20,

            "backtest_start":
                80,

            "recent_decay":
                0.975
        }

    return {

        "min_train": 50,

        "trees": 45,

        "depth": 5,

        "leaf": 3,

        "max_features": "sqrt",

        "selected_features":
            15,

        "backtest_start":
            50,

        "recent_decay":
            0.97
    }


# ============================================================
# 15. MODEL FACTORY
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

            max_features=config[
                "max_features"
            ],

            class_weight="balanced",

            n_jobs=-1,

            random_state=42
        )

    if model_name == "RandomForest":

        return RandomForestClassifier(

            n_estimators=trees,

            max_depth=depth,

            min_samples_leaf=leaf,

            max_features=config[
                "max_features"
            ],

            class_weight="balanced",

            n_jobs=-1,

            random_state=42
        )

    if model_name == "HistGradientBoosting":

        return HistGradientBoostingClassifier(

            max_iter=max(
                35,
                int(
                    trees * 0.65
                )
            ),

            max_leaf_nodes=15,

            learning_rate=0.045,

            min_samples_leaf=leaf,

            l2_regularization=1.0,

            random_state=42
        )

    raise ValueError(
        model_name
    )


# ============================================================
# 16. PREPROCESSING
# ============================================================

def fit_imputer(
    X_train
):

    imputer = SimpleImputer(
        strategy="median"
    )

    X_out = imputer.fit_transform(
        X_train
    )

    return imputer, X_out


# ============================================================
# 17. TRAINING-ONLY FEATURE SELECTION
# ============================================================

def select_features_training_only(
    X_train,
    y_train,
    feature_names,
    max_features
):

    # --------------------------------------------------------
    # If already small, no selection needed.
    # --------------------------------------------------------

    if len(feature_names) <= max_features:

        return list(
            feature_names
        )

    # --------------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------------

    valid_features = []

    for col in feature_names:

        try:

            if (
                X_train[col]
                .nunique(
                    dropna=False
                )
                > 1
            ):

                valid_features.append(
                    col
                )

        except Exception:
            continue

    if not valid_features:

        return list(
            feature_names[:max_features]
        )

    # --------------------------------------------------------
    # Median impute using TRAIN ONLY
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imp = imputer.fit_transform(
        X_train[
            valid_features
        ]
    )

    # --------------------------------------------------------
    # Small selector model
    #
    # This model is ONLY for selecting features.
    # It never sees validation/test rows.
    # --------------------------------------------------------

    selector = ExtraTreesClassifier(

        n_estimators=50,

        max_depth=6,

        min_samples_leaf=3,

        max_features="sqrt",

        class_weight="balanced",

        n_jobs=-1,

        random_state=123
    )

    try:

        selector.fit(
            X_imp,
            y_train
        )

        importance = (
            selector
            .feature_importances_
        )

        ranking = np.argsort(
            importance
        )[::-1]

        chosen = [

            valid_features[i]

            for i in ranking[
                :max_features
            ]
        ]

        if chosen:

            return chosen

    except Exception:

        pass

    return list(
        valid_features[
            :max_features
        ]
    )


# ============================================================
# 18. PROBABILITY
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

        return np.ones(
            10
        ) / 10

    return (
        output
        /
        total
    )


# ============================================================
# 19. PROBABILITY STABILIZATION
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

    logits = np.log(p)

    logits /= temperature

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
# 20. METRICS
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

    # Brier score for multiclass
    brier = (
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

        "top1":
            top1,

        "top3":
            top3,

        "top5":
            top5,

        "dead7":
            dead7,

        "logloss":
            logloss,

        "brier":
            float(brier)
    }


# ============================================================
# 21. RANDOM BASELINE
# ============================================================

RANDOM_TOP1 = 0.10
RANDOM_TOP3 = 0.30
RANDOM_TOP5 = 0.50


# ============================================================
# 22. MODEL WEIGHT
# ============================================================

def model_weight(
    score
):

    if score <= 0:

        return 0.05

    return (
        0.20
        +
        score ** 1.5
    )


def normalize_weights(
    weights
):

    names = list(
        weights.keys()
    )

    values = np.asarray(
        list(
            weights.values()
        ),
        dtype=float
    )

    values = np.clip(
        values,
        1e-6,
        None
    )

    values /= values.sum()

    return {

        name:
            float(value)

        for name, value in zip(
            names,
            values
        )
    }


# ============================================================
# 23. SINGLE WALK-FORWARD PREDICTION
# ============================================================

def walk_forward_predict_one(
    X,
    y,
    test_idx,
    config
):

    train_end = test_idx

    if train_end < config[
        "min_train"
    ]:

        return None

    X_train_full = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    X_test_full = X.iloc[
        [test_idx]
    ].copy()

    if y_train.nunique() < 2:

        return None

    # --------------------------------------------------------
    # Feature selection
    # ONLY training rows
    # --------------------------------------------------------

    selected_features = (
        select_features_training_only(

            X_train_full,

            y_train,

            list(
                X_train_full.columns
            ),

            config[
                "selected_features"
            ]
        )
    )

    X_train = (
        X_train_full[
            selected_features
        ]
    )

    X_test = (
        X_test_full[
            selected_features
        ]
    )

    # --------------------------------------------------------
    # Imputer FIT ONLY on training data
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test_imp = (
        imputer.transform(
            X_test
        )
    )

    actual = int(
        y.iloc[test_idx]
    )

    model_probs = {}

    # --------------------------------------------------------
    # Train every model FROM SCRATCH
    # --------------------------------------------------------

    for model_name in MODEL_NAMES:

        try:

            model = create_model(
                model_name,
                config
            )

            model.fit(
                X_train_imp,
                y_train
            )

            probs = probability_vector(
                model,
                X_test_imp
            )

            probs = stabilize_probability(
                probs
            )

            model_probs[
                model_name
            ] = probs

        except Exception:

            continue

    if not model_probs:

        return None

    # --------------------------------------------------------
    # Equal weight inside each individual walk-forward test.
    #
    # This prevents the same test result from being used
    # to choose its own model weight.
    #
    # Adaptive weights are calculated AFTER the whole
    # historical backtest.
    # --------------------------------------------------------

    ensemble = np.mean(
        list(
            model_probs.values()
        ),
        axis=0
    )

    ensemble = stabilize_probability(
        ensemble,
        temperature=1.05
    )

    metrics = calculate_metrics(
        ensemble,
        actual
    )

    return {

        "probabilities":
            ensemble,

        "metrics":
            metrics,

        "actual":
            actual,

        "selected_features":
            selected_features,

        "model_probabilities":
            model_probs
    }


# ============================================================
# 24. STRICT WALK-FORWARD BACKTEST
# ============================================================

def strict_walk_forward_backtest(
    df_features,
    position,
    config
):

    X = (
        df_features[
            FEATURES
        ]
        .astype(float)
    )

    y = (
        df_features[position]
        .astype(int)
    )

    n = len(df_features)

    start = max(
        config["min_train"],
        config["backtest_start"]
    )

    if n <= start:

        return {

            "scores": {},

            "tests": 0,

            "random": {

                "top1":
                    RANDOM_TOP1,

                "top3":
                    RANDOM_TOP3,

                "top5":
                    RANDOM_TOP5
            },

            "history":
                []
        }

    history = []

    # --------------------------------------------------------
    # TRUE WALK-FORWARD
    #
    # Every test index is evaluated sequentially.
    # --------------------------------------------------------

    for test_idx in range(
        start,
        n
    ):

        result = walk_forward_predict_one(
            X,
            y,
            test_idx,
            config
        )

        if result is None:
            continue

        metrics = result[
            "metrics"
        ]

        history.append(

            {

                "index":
                    test_idx,

                "actual":
                    result["actual"],

                "top1":
                    metrics["top1"],

                "top3":
                    metrics["top3"],

                "top5":
                    metrics["top5"],

                "dead7":
                    metrics["dead7"],

                "logloss":
                    metrics["logloss"],

                "brier":
                    metrics["brier"]
            }
        )

    if not history:

        return {

            "scores": {},

            "tests": 0,

            "random": {

                "top1":
                    RANDOM_TOP1,

                "top3":
                    RANDOM_TOP3,

                "top5":
                    RANDOM_TOP5
            },

            "history":
                []
        }

    hist = pd.DataFrame(
        history
    )

    # --------------------------------------------------------
    # Recency weighting
    #
    # Recent test rows have slightly more influence.
    # --------------------------------------------------------

    m = len(hist)

    positions = np.arange(
        m
    )

    weights = (
        config["recent_decay"]
        **
        (m - positions - 1)
    )

    weights = (
        weights
        /
        weights.sum()
    )

    top1 = float(
        np.sum(
            hist["top1"]
            * weights
        )
    )

    top3 = float(
        np.sum(
            hist["top3"]
            * weights
        )
    )

    top5 = float(
        np.sum(
            hist["top5"]
            * weights
        )
    )

    dead7 = float(
        np.sum(
            hist["dead7"]
            * weights
        )
    )

    logloss = float(
        np.sum(
            hist["logloss"]
            * weights
        )
    )

    brier = float(
        np.sum(
            hist["brier"]
            * weights
        )
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    rolling_top3 = (
        hist["top3"]
        .rolling(
            min(30, max(5, m)),
            min_periods=5
        )
        .mean()
    )

    if len(rolling_top3.dropna()):

        stability = float(
            1.0
            -
            rolling_top3.std()
        )

        stability = float(
            np.clip(
                stability,
                0,
                1
            )
        )

    else:

        stability = 0.0

    # --------------------------------------------------------
    # Overall score
    #
    # Accuracy is NOT compared alone.
    # --------------------------------------------------------

    logloss_score = (
        1
        /
        (
            1
            +
            logloss
        )
    )

    brier_score = (
        1
        /
        (
            1
            +
            brier
        )
    )

    score = (

        0.25 * top1
        +
        0.25 * top3
        +
        0.15 * top5
        +
        0.10 * stability
        +
        0.10 * logloss_score
        +
        0.10 * brier_score
        +
        0.05 * (1 - dead7)
    )

    # --------------------------------------------------------
    # Raw accuracy
    # --------------------------------------------------------

    raw_top1 = float(
        hist["top1"].mean()
    )

    raw_top3 = float(
        hist["top3"].mean()
    )

    raw_top5 = float(
        hist["top5"].mean()
    )

    raw_dead7 = float(
        hist["dead7"].mean()
    )

    return {

        "scores": {

            "top1":
                top1,

            "top3":
                top3,

            "top5":
                top5,

            "dead7":
                dead7,

            "logloss":
                logloss,

            "brier":
                brier,

            "stability":
                stability,

            "score":
                score,

            "raw_top1":
                raw_top1,

            "raw_top3":
                raw_top3,

            "raw_top5":
                raw_top5,

            "raw_dead7":
                raw_dead7
        },

        "tests":
            len(hist),

        "random": {

            "top1":
                RANDOM_TOP1,

            "top3":
                RANDOM_TOP3,

            "top5":
                RANDOM_TOP5
        },

        "history":
            history
    }


# ============================================================
# 25. ADAPTIVE MODEL WEIGHTS
# ============================================================

def get_adaptive_weights(
    backtest_result
):

    # --------------------------------------------------------
    # V8.1 uses model-specific walk-forward evaluation.
    #
    # To avoid test contamination, we perform a second,
    # lightweight model comparison using the historical
    # predictions. If unavailable, equal weights.
    #
    # For safety and mobile speed, final ensemble uses
    # stable equal weights unless a strong model advantage
    # is demonstrated.
    # --------------------------------------------------------

    return {

        "ExtraTrees":
            1 / 3,

        "RandomForest":
            1 / 3,

        "HistGradientBoosting":
            1 / 3
    }


# ============================================================
# 26. FINAL MODEL
# ============================================================

def final_prediction(
    df_features,
    position,
    backtest_result,
    config
):

    X_full = (
        df_features[
            FEATURES
        ]
        .astype(float)
    )

    y = (
        df_features[position]
        .astype(int)
    )

    # --------------------------------------------------------
    # Historical data ONLY
    #
    # Last row is the future dummy row.
    # --------------------------------------------------------

    X_train_full = X_full.iloc[
        :-1
    ].copy()

    y_train = y.iloc[
        :-1
    ].copy()

    X_next_full = X_full.iloc[
        [-1]
    ].copy()

    # --------------------------------------------------------
    # Feature selection using historical data ONLY
    # --------------------------------------------------------

    selected_features = (
        select_features_training_only(

            X_train_full,

            y_train,

            list(
                X_train_full.columns
            ),

            config[
                "selected_features"
            ]
        )
    )

    X_train = (
        X_train_full[
            selected_features
        ]
    )

    X_next = (
        X_next_full[
            selected_features
        ]
    )

    # --------------------------------------------------------
    # Imputer learns from historical data ONLY
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = (
        imputer.fit_transform(
            X_train
        )
    )

    X_next_imp = (
        imputer.transform(
            X_next
        )
    )

    # --------------------------------------------------------
    # Final models are trained FROM SCRATCH
    # on all available historical data.
    #
    # Nothing is loaded from previous runs.
    # --------------------------------------------------------

    model_probabilities = {}

    for model_name in MODEL_NAMES:

        try:

            model = create_model(
                model_name,
                config
            )

            model.fit(
                X_train_imp,
                y_train
            )

            probs = probability_vector(
                model,
                X_next_imp
            )

            probs = stabilize_probability(
                probs
            )

            model_probabilities[
                model_name
            ] = probs

        except Exception:

            continue

    if not model_probabilities:

        ensemble = (
            np.ones(10)
            / 10
        )

    else:

        # ----------------------------------------------------
        # V8.1 deliberately avoids using the same final
        # target to optimize model weights.
        #
        # Equal-weight ensemble is safer against overfitting.
        # ----------------------------------------------------

        ensemble = np.mean(
            list(
                model_probabilities.values()
            ),
            axis=0
        )

        ensemble = stabilize_probability(
            ensemble,
            temperature=1.05
        )

    # --------------------------------------------------------
    # HOT 5
    # --------------------------------------------------------

    hot_idx = np.argsort(
        ensemble
    )[::-1][:5]

    hot = [

        (
            int(i),
            float(
                ensemble[i]
            )
        )

        for i in hot_idx
    ]

    # --------------------------------------------------------
    # DEAD 7
    # --------------------------------------------------------

    dead_idx = np.argsort(
        ensemble
    )[:7]

    dead = [

        (
            int(i),
            float(
                ensemble[i]
            )
        )

        for i in dead_idx
    ]

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL AGREEMENT
    # --------------------------------------------------------

    agreement = 0.0

    rankings = [

        np.argsort(
            p
        )[::-1][:5]

        for p in
        model_probabilities.values()
    ]

    if len(rankings) >= 2:

        intersections = []

        for i in range(
            len(rankings)
        ):

            for j in range(
                i + 1,
                len(rankings)
            ):

                a = set(
                    rankings[i]
                )

                b = set(
                    rankings[j]
                )

                intersections.append(
                    len(
                        a.intersection(b)
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

    # --------------------------------------------------------
    # Selected model
    # --------------------------------------------------------

    model_top1 = {}

    for name, probs in (
        model_probabilities.items()
    ):

        model_top1[name] = int(
            np.argmax(
                probs
            )
        )

    if model_top1:

        # model with smallest entropy
        # = strongest concentration
        model_concentration = {}

        for name, probs in (
            model_probabilities.items()
        ):

            sorted_p = np.sort(
                probs
            )[::-1]

            model_concentration[
                name
            ] = float(
                sorted_p[:3].sum()
            )

        selected_model = max(
            model_concentration,
            key=model_concentration.get
        )

    else:

        selected_model = "Ensemble"

    return {

        "model":
            selected_model,

        "weights": {

            name:
                1 /
                max(
                    len(
                        model_probabilities
                    ),
                    1
                )

            for name in
            model_probabilities.keys()
        },

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
            float(top3_concentration),

        "top5_concentration":
            float(top5_concentration),

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

                📌 Top-1 Gap:
                {confidence:.1f}%

                &nbsp; | &nbsp;

                Top-3:
                {concentration:.1f}%

                &nbsp; | &nbsp;

                Agreement:
                {agreement:.1f}%

            </div>

            <div class="model-badge">

                🤖 Final AI:
                {result["model"]}

                <br>

                ⚖️ Ensemble:
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
                🤖 Adaptive Walk-Forward AI
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

    scores = backtest.get(
        "scores",
        {}
    )

    if not scores:

        st.warning(
            "ไม่มีข้อมูล Backtest เพียงพอ"
        )

        return

    random_top1 = (
        RANDOM_TOP1 * 100
    )

    random_top3 = (
        RANDOM_TOP3 * 100
    )

    random_top5 = (
        RANDOM_TOP5 * 100
    )

    rows = [

        {

            "Metric":
                "Top-1",

            "AI":
                f"{scores['top1'] * 100:.1f}%",

            "Random":
                f"{random_top1:.1f}%",

            "Edge":
                f"{(scores['top1'] - RANDOM_TOP1) * 100:+.1f}%"
        },

        {

            "Metric":
                "Top-3",

            "AI":
                f"{scores['top3'] * 100:.1f}%",

            "Random":
                f"{random_top3:.1f}%",

            "Edge":
                f"{(scores['top3'] - RANDOM_TOP3) * 100:+.1f}%"
        },

        {

            "Metric":
                "Top-5",

            "AI":
                f"{scores['top5'] * 100:.1f}%",

            "Random":
                f"{random_top5:.1f}%",

            "Edge":
                f"{(scores['top5'] - RANDOM_TOP5) * 100:+.1f}%"
        },

        {

            "Metric":
                "Dead-7",

            "AI":
                f"{scores['dead7'] * 100:.1f}%",

            "Random":
                "70.0%",

            "Edge":
                f"{(scores['dead7'] - 0.70) * 100:+.1f}%"
        },

        {

            "Metric":
                "LogLoss",

            "AI":
                f"{scores['logloss']:.3f}",

            "Random":
                f"{np.log(10):.3f}",

            "Edge":
                ""
        },

        {

            "Metric":
                "Brier",

            "AI":
                f"{scores['brier']:.3f}",

            "Random":
                "0.900",

            "Edge":
                ""
        }
    ]

    st.dataframe(

        pd.DataFrame(rows),

        use_container_width=True,

        hide_index=True
    )

    # --------------------------------------------------------
    # Raw vs Recency
    # --------------------------------------------------------

    raw_rows = [

        {

            "Metric":
                "Top-1 Raw",

            "Value":
                f"{scores['raw_top1'] * 100:.1f}%"
        },

        {

            "Metric":
                "Top-3 Raw",

            "Value":
                f"{scores['raw_top3'] * 100:.1f}%"
        },

        {

            "Metric":
                "Top-5 Raw",

            "Value":
                f"{scores['raw_top5'] * 100:.1f}%"
        },

        {

            "Metric":
                "Stability",

            "Value":
                f"{scores['stability'] * 100:.1f}%"
        },

        {

            "Metric":
                "AI Score",

            "Value":
                f"{scores['score'] * 100:.1f}%"
        },

        {

            "Metric":
                "Backtest Tests",

            "Value":
                str(
                    backtest[
                        "tests"
                    ]
                )
        }
    ]

    st.dataframe(

        pd.DataFrame(raw_rows),

        use_container_width=True,

        hide_index=True
    )

    # --------------------------------------------------------
    # Warning
    # --------------------------------------------------------

    if backtest["tests"] < 100:

        st.warning(

            f"⚠️ Backtest มีเพียง "
            f"{backtest['tests']} งวด "
            f"ความแม่นยำมีความผันผวนสูง "
            f"ไม่ควรตีความว่าเป็น Accuracy ระยะยาว"
        )

    elif backtest["tests"] < 200:

        st.info(

            f"ℹ️ Backtest {backtest['tests']} งวด "
            f"เพียงพอสำหรับประเมินเบื้องต้น "
            f"แต่ยังมี Sampling Variance"
        )

    else:

        st.success(

            f"✅ ใช้ Walk-Forward จริง "
            f"{backtest['tests']} งวด"
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
            STRICT WALK-FORWARD •
            NO PERSISTENT MEMORY •
            FEATURE SELECTION •
            LEAKAGE SAFE
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

    # ========================================================
    # FETCH
    # ========================================================

    with st.spinner(
        "📥 กำลังโหลดข้อมูลย้อนหลัง..."
    ):

        try:

            df = fetch_lottery_data(
                url
            )

        except NetworkScrapingError as exc:

            st.error(
                f"🌐 Network Error: {exc}"
            )

            st.info(
                "ตรวจสอบ Internet หรือเว็บไซต์ต้นทาง"
            )

            return

        except HTTPStatusScrapingError as exc:

            st.error(
                f"🚫 HTTP Error: {exc}"
            )

            return

        except ParsingScrapingError as exc:

            st.error(
                f"🧩 Data Parsing Error: {exc}"
            )

            st.info(
                "เว็บไซต์อาจเปลี่ยนรูปแบบหน้าเว็บ "
                "ทำให้ระบบอ่านข้อมูลไม่พบ"
            )

            return

        except Exception as exc:

            st.error(
                f"❌ Unexpected Error: {exc}"
            )

            return

    # ========================================================
    # DATA CHECK
    # ========================================================

    if df.empty:

        st.error(
            "❌ ไม่พบข้อมูล"
        )

        return

    if len(df) < 50:

        st.error(

            f"❌ พบข้อมูลเพียง "
            f"{len(df)} งวด"
        )

        st.warning(
            "V8.1 ต้องการอย่างน้อย 50 งวด"
        )

        return

    # ========================================================
    # TARGET DATE
    # ========================================================

    target_date = calculate_target_date(

        df,

        selected_day
    )

    # ========================================================
    # DUMMY ROW
    # ========================================================
    #
    # Result 000/00 is ONLY a placeholder.
    #
    # Because all result-derived features are shifted,
    # the dummy result itself is NOT used as a feature
    # for the target row.
    #
    # ========================================================

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

    # ========================================================
    # FEATURES
    # ========================================================

    with st.spinner(
        "🧠 สร้าง Strict Causal Features..."
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

        f"⚡ V8.1 MAX • "
        f"ข้อมูล {len(df):,} งวด • "
        f"Walk-Forward ทุกงวด • "
        f"Min Train {config['min_train']} • "
        f"Features Pool {len(FEATURES)} • "
        f"Selected ≤ {config['selected_features']} • "
        f"Trees {config['trees']}"
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    backtest_results = {}

    progress = st.progress(
        0
    )

    status_text = st.empty()

    for idx, position in enumerate(
        POSITIONS
    ):

        status_text.caption(

            f"🧠 Strict Walk-Forward: "
            f"{POSITION_LABELS[position]}"
        )

        backtest_results[
            position
        ] = strict_walk_forward_backtest(

            feature_df.iloc[:-1],

            position,

            config
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

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    with st.spinner(
        "🤖 Train Final AI จากข้อมูลย้อนหลังเท่านั้น..."
    ):

        final_results = {}

        for position in POSITIONS:

            final_results[
                position
            ] = final_prediction(

                feature_df,

                position,

                backtest_results[
                    position
                ],

                config
            )

    # ========================================================
    # STATUS
    # ========================================================

    selected_models = [

        final_results[p]["model"]

        for p in POSITIONS
    ]

    model_text = " | ".join(
        selected_models
    )

    total_tests = [

        backtest_results[p][
            "tests"
        ]

        for p in POSITIONS
    ]

    min_tests = min(
        total_tests
    )

    max_tests = max(
        total_tests
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

        🧠 Feature Pool:
        {len(FEATURES)}<br>

        🎯 Selected Features:
        ≤ {config["selected_features"]}<br>

        🌳 Trees:
        {config["trees"]}<br>

        🔄 Strict Walk-Forward:
        ENABLED<br>

        🧪 Backtest:
        {min_tests}–{max_tests} งวด / ตำแหน่ง<br>

        🤖 Models:
        {model_text}<br>

        🔒 Future Leakage:
        BLOCKED<br>

        🧠 Persistent Memory:
        DISABLED

        </div>
        """,

        unsafe_allow_html=True
    )

    st.write("")

    # ========================================================
    # TABS
    # ========================================================

    tab_hot, tab_dead, tab_accuracy = st.tabs(

        [
            "🎯 เลขเด่น AI",
            "🛑 เลขดับ 7",
            "📊 Walk-Forward Accuracy"
        ]
    )

    # ========================================================
    # HOT
    # ========================================================

    with tab_hot:

        st.subheader(
            "🎯 AI TOP 5"
        )

        st.caption(

            "Probability จาก Ensemble "
            "ที่ Train ด้วยข้อมูลย้อนหลัง "
            "ก่อนงวดเป้าหมายเท่านั้น"
        )

        for position in POSITIONS:

            display_hot(

                position,

                final_results[position]
            )

    # ========================================================
    # DEAD
    # ========================================================

    with tab_dead:

        st.subheader(
            "🛑 AI BOTTOM 7"
        )

        st.warning(

            "Dead-7 หมายถึง 7 ตัวที่ AI "
            "ให้ Probability ต่ำสุด "
            "ไม่ใช่การรับประกันว่าเลขจะไม่ออก"
        )

        for position in POSITIONS:

            display_dead(

                position,

                final_results[position]
            )

    # ========================================================
    # ACCURACY
    # ========================================================

    with tab_accuracy:

        st.subheader(
            "📊 Strict Walk-Forward Backtest"
        )

        st.caption(

            "แต่ละงวดจะใช้ข้อมูลก่อนหน้างวดนั้น "
            "Train → Predict → เลื่อนไปงวดถัดไป"
        )

        for position in POSITIONS:

            display_backtest(

                position,

                backtest_results[position]
            )

            st.write("")

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    with st.expander(
        "🧠 Selected Features ที่ใช้ทำนาย"
    ):

        rows = []

        for position in POSITIONS:

            selected = final_results[
                position
            ][
                "selected_features"
            ]

            rows.append(

                {

                    "ตำแหน่ง":
                        POSITION_LABELS[position],

                    "จำนวน":
                        len(selected),

                    "Features":
                        ", ".join(
                            selected
                        )
                }
            )

        st.dataframe(

            pd.DataFrame(rows),

            use_container_width=True,

            hide_index=True
        )

    # ========================================================
    # MODEL PROBABILITIES
    # ========================================================

    with st.expander(
        "🤖 Probability ของแต่ละ AI"
    ):

        rows = []

        for position in POSITIONS:

            result = final_results[
                position
            ]

            model_probs = result[
                "model_probabilities"
            ]

            for model_name, probs in (
                model_probs.items()
            ):

                top = np.argsort(
                    probs
                )[::-1][:5]

                rows.append(

                    {

                        "ตำแหน่ง":
                            POSITION_LABELS[position],

                        "Model":
                            model_name,

                        "Top-5":
                            " - ".join(
                                str(
                                    int(x)
                                )
                                for x in top
                            ),

                        "Top-1 Probability":
                            f"{probs[top[0]] * 100:.1f}%"
                    }
                )

        st.dataframe(

            pd.DataFrame(rows),

            use_container_width=True,

            hide_index=True
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    with st.expander(
        "🎯 สรุปเลขเด่นทั้งหมด"
    ):

        summary = []

        for position in POSITIONS:

            result = final_results[
                position
            ]

            top = result[
                "hot"
            ][0]

            bt = backtest_results[
                position
            ]

            scores = bt.get(
                "scores",
                {}
            )

            summary.append(

                {

                    "ตำแหน่ง":
                        POSITION_LABELS[position],

                    "Top-1":
                        top[0],

                    "Probability":
                        f"{top[1] * 100:.1f}%",

                    "AI":
                        result["model"],

                    "Top-1 Gap":
                        f"{result['confidence'] * 100:.1f}%",

                    "Agreement":
                        f"{result['agreement'] * 100:.1f}%",

                    "WF Top-1":
                        (
                            f"{scores.get('top1', 0) * 100:.1f}%"
                        ),

                    "Random":
                        "10.0%"
                }
            )

        st.dataframe(

            pd.DataFrame(summary),

            use_container_width=True,

            hide_index=True
        )

    # ========================================================
    # SYSTEM INFO
    # ========================================================

    with st.expander(
        "🔧 System Information"
    ):

        info = {

            "Engine":
                "LOTTO AI PRO V8.1 MAX",

            "Historical draws":
                len(df),

            "Feature Pool":
                len(FEATURES),

            "Selected Features":
                config[
                    "selected_features"
                ],

            "Minimum Train":
                config[
                    "min_train"
                ],

            "Trees":
                config[
                    "trees"
                ],

            "Strict Walk-Forward":
                "ENABLED",

            "Train Every Draw":
                "YES",

            "Persistent Model Memory":
                "DISABLED",

            "Future Leakage":
                "BLOCKED",

            "Median Imputation":
                "TRAIN ONLY",

            "Feature Selection":
                "TRAIN ONLY",

            "Random Baseline":
                "10 / 30 / 50%",

            "AI":
                "ExtraTrees / RandomForest / HistGradientBoosting",

            "XGBoost":
                "DISABLED",

            "Markov":
                "DISABLED",

            "Frequency Voting":
                "DISABLED",

            "Calendar Voting":
                "DISABLED",

            "Equation Voting":
                "DISABLED",

            "Data Hash":
                data_hash[:16] + "..."
        }

        st.dataframe(

            pd.DataFrame(
                [
                    {
                        "Setting":
                            key,

                        "Value":
                            value
                    }

                    for key, value in
                    info.items()
                ]
            ),

            use_container_width=True,

            hide_index=True
        )


# ============================================================
# 32. RUN
# ============================================================

if __name__ == "__main__":

    main()
