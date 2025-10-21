import re
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Narration", "Ref/Cheque No.", "Debit", "Credit", "Balance"]


def _clean_description(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    desc = desc.strip()
    desc = re.sub(r"\s+", " ", desc)
    desc = re.sub(r"[#_*]", " ", desc)
    return desc

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Coerce types
    df["Narration"] = df["Narration"].astype(str).apply(_clean_description)
    df["Ref/Cheque No."] = df["Ref/Cheque No."].astype(str).apply(_clean_description)
    df["Debit"] = df["Debit"].astype(str).str.replace(",", "").replace("", "0").replace("-", "0").astype(float)
    df["Credit"] = df["Credit"].astype(str).str.replace(",", "").replace("", "0").replace("-", "0").astype(float)
    df["Balance"] = df["Balance"].astype(str).str.replace(",", "").replace("", "0").replace("-", "0").astype(float)

    # Basic de-dup
    df.drop_duplicates(subset=["Date", "Narration", "Ref/Cheque No.", "Debit", "Credit", "Balance"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df









