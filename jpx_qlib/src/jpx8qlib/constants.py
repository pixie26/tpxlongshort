BASE_FEATURE_COLUMNS = [
    "SecuritiesCode",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "AdjustmentFactor",
    "ExpectedDividend",
    "SupervisionFlag",
]

DERIVED_FEATURE_COLUMNS = [
    "Amplitude",
    "OpenCloseReturn",
    "Return",
    "Volatility10",
    "Volatility30",
    "Volatility50",
    "CloseSMA3",
    "CloseSMA5",
    "CloseSMA10",
    "CloseSMA30",
    "ReturnSMA3",
    "ReturnSMA5",
    "ReturnSMA10",
    "ReturnSMA30",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
LABEL_COLUMN = "Target"
INDEX_COLUMNS = ["Date", "SecuritiesCode"]
