from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "jpx"

prices_path = DATA_DIR / "train_files" / "stock_prices.csv"

print("Project directory:", PROJECT_DIR)
print("Reading:", prices_path)

if not prices_path.exists():
    raise FileNotFoundError(
        f"找不到文件：{prices_path}\n"
        "请检查 JPX 压缩包是否已经解压，以及是否存在 train_files 文件夹。"
    )

prices = pd.read_csv(
    prices_path,
    parse_dates=["Date"],
)

print("Shape:", prices.shape)
print(prices.head())
print(prices.dtypes)
print("Date range:", prices["Date"].min(), "to", prices["Date"].max())
print("Number of securities:", prices["SecuritiesCode"].nunique())

import numpy as np
import matplotlib.pyplot as plt


REPORT_DIR = PROJECT_DIR / "reports" / "data_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 主键和 RowId 唯一性
# ============================================================

key_columns = ["Date", "SecuritiesCode"]

duplicate_keys = prices.duplicated(key_columns).sum()
duplicate_rowids = prices["RowId"].duplicated().sum()

print("\n=== KEY CHECK ===")
print("Duplicate Date × SecuritiesCode:", duplicate_keys)
print("Duplicate RowId:", duplicate_rowids)

if duplicate_keys != 0:
    raise ValueError("Date × SecuritiesCode 存在重复记录")

if duplicate_rowids != 0:
    raise ValueError("RowId 存在重复记录")


# ============================================================
# 2. 每日股票数量 / universe 变化
# ============================================================

daily_universe = (
    prices.groupby("Date")
    .agg(
        n_rows=("RowId", "size"),
        n_securities=("SecuritiesCode", "nunique"),
        missing_target=("Target", lambda x: x.isna().sum()),
        supervision_count=("SupervisionFlag", "sum"),
    )
    .reset_index()
)

print("\n=== DAILY UNIVERSE ===")
print(daily_universe["n_securities"].describe())

print("\nFirst 10 dates:")
print(daily_universe.head(10))

print("\nLast 10 dates:")
print(daily_universe.tail(10))

print("\nDates with minimum universe:")
print(
    daily_universe.loc[
        daily_universe["n_securities"]
        == daily_universe["n_securities"].min()
    ].head(20)
)

print("\nDates with maximum universe:")
print(
    daily_universe.loc[
        daily_universe["n_securities"]
        == daily_universe["n_securities"].max()
    ].head(20)
)

daily_universe.to_csv(
    REPORT_DIR / "daily_universe.csv",
    index=False,
)

plt.figure(figsize=(12, 5))
plt.plot(
    daily_universe["Date"],
    daily_universe["n_securities"],
)
plt.title("Number of Securities by Date")
plt.xlabel("Date")
plt.ylabel("Number of securities")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "daily_universe.png",
    dpi=150,
)
plt.close()


# ============================================================
# 3. 缺失值
# ============================================================

missing_report = (
    prices.isna()
    .agg(["sum", "mean"])
    .T
    .rename(
        columns={
            "sum": "missing_count",
            "mean": "missing_ratio",
        }
    )
    .sort_values("missing_ratio", ascending=False)
)

print("\n=== MISSING VALUES ===")
print(missing_report)

missing_report.to_csv(
    REPORT_DIR / "missing_values.csv"
)


# ============================================================
# 4. OHLC 和成交量基础检查
# ============================================================

complete_ohlc = prices[
    ["Open", "High", "Low", "Close"]
].notna().all(axis=1)

bad_ohlc = complete_ohlc & (
    (prices["Low"] > prices["High"])
    | (prices["Open"] < prices["Low"])
    | (prices["Open"] > prices["High"])
    | (prices["Close"] < prices["Low"])
    | (prices["Close"] > prices["High"])
)

non_positive_close = (
    prices["Close"].notna()
    & (prices["Close"] <= 0)
)

negative_volume = prices["Volume"] < 0

print("\n=== PRICE SANITY CHECK ===")
print("Invalid OHLC rows:", int(bad_ohlc.sum()))
print("Non-positive Close rows:", int(non_positive_close.sum()))
print("Negative Volume rows:", int(negative_volume.sum()))

if bad_ohlc.any():
    prices.loc[
        bad_ohlc,
        [
            "RowId",
            "Date",
            "SecuritiesCode",
            "Open",
            "High",
            "Low",
            "Close",
        ],
    ].to_csv(
        REPORT_DIR / "invalid_ohlc_rows.csv",
        index=False,
    )


# ============================================================
# 5. AdjustmentFactor 检查
# ============================================================

adjustment_events = prices.loc[
    prices["AdjustmentFactor"].ne(1)
    & prices["AdjustmentFactor"].notna(),
    [
        "Date",
        "SecuritiesCode",
        "Close",
        "AdjustmentFactor",
    ],
].copy()

print("\n=== ADJUSTMENT FACTOR ===")
print("Rows where AdjustmentFactor != 1:", len(adjustment_events))

print("\nAdjustmentFactor distribution:")
print(
    adjustment_events["AdjustmentFactor"]
    .value_counts()
    .sort_index()
    .head(30)
)

adjustment_events.to_csv(
    REPORT_DIR / "adjustment_events.csv",
    index=False,
)


# ============================================================
# 6. ExpectedDividend 和 SupervisionFlag
# ============================================================

print("\n=== SPECIAL FIELDS ===")
print(
    "ExpectedDividend non-null rows:",
    int(prices["ExpectedDividend"].notna().sum()),
)

print(
    "SupervisionFlag=True rows:",
    int(prices["SupervisionFlag"].sum()),
)

print(
    "Stocks ever under supervision:",
    prices.loc[
        prices["SupervisionFlag"],
        "SecuritiesCode",
    ].nunique(),
)


# ============================================================
# 7. Target 缺失的位置
# ============================================================

missing_target_by_date = (
    prices.groupby("Date")["Target"]
    .agg(
        rows="size",
        missing=lambda x: x.isna().sum(),
    )
    .reset_index()
)

missing_target_by_date["missing_ratio"] = (
    missing_target_by_date["missing"]
    / missing_target_by_date["rows"]
)

print("\n=== TARGET MISSING BY DATE ===")
print(
    missing_target_by_date.loc[
        missing_target_by_date["missing"] > 0
    ].tail(20)
)

missing_target_by_date.to_csv(
    REPORT_DIR / "target_missing_by_date.csv",
    index=False,
)


# ============================================================
# 8. 手工验证 Target
# 选择一个历史上没有复权事件的股票
# ============================================================

security_stats = (
    prices.groupby("SecuritiesCode")
    .agg(
        n_rows=("RowId", "size"),
        adjustment_events=(
            "AdjustmentFactor",
            lambda x: x.ne(1).sum(),
        ),
    )
)

candidate_codes = security_stats.loc[
    security_stats["adjustment_events"] == 0
].sort_values(
    "n_rows",
    ascending=False,
)

if candidate_codes.empty:
    raise ValueError("没有找到无复权事件的股票")

sample_code = int(candidate_codes.index[0])

sample = (
    prices.loc[
        prices["SecuritiesCode"] == sample_code,
        [
            "Date",
            "SecuritiesCode",
            "Close",
            "Target",
        ],
    ]
    .sort_values("Date")
    .copy()
)

sample["close_t_plus_1"] = sample["Close"].shift(-1)
sample["close_t_plus_2"] = sample["Close"].shift(-2)

sample["manual_target"] = (
    sample["close_t_plus_2"]
    / sample["close_t_plus_1"]
    - 1
)

sample["target_difference"] = (
    sample["Target"]
    - sample["manual_target"]
)

target_comparison = sample.dropna(
    subset=["Target", "manual_target"]
)

print("\n=== TARGET VALIDATION ===")
print("Sample SecuritiesCode:", sample_code)
print("Compared rows:", len(target_comparison))

print(
    target_comparison[
        [
            "Date",
            "Close",
            "close_t_plus_1",
            "close_t_plus_2",
            "Target",
            "manual_target",
            "target_difference",
        ]
    ].head(20)
)

print("\nAbsolute target difference:")
print(
    target_comparison[
        "target_difference"
    ].abs().describe()
)

print(
    "Maximum absolute difference:",
    target_comparison[
        "target_difference"
    ].abs().max(),
)

target_comparison.to_csv(
    REPORT_DIR / "target_validation_sample.csv",
    index=False,
)


# ============================================================
# 9. 总结
# ============================================================

print("\n=== AUDIT COMPLETE ===")
print("Reports saved to:", REPORT_DIR)