"""Reading tabular files, inferring column kinds, descriptive statistics and chart data."""
import io
import json
import re
import warnings

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("The file has no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names must be unique.")
    return df


def df_records(df: pd.DataFrame) -> list:
    """JSON-safe records (NaN -> null, timestamps -> ISO strings)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


_DATE_HINT = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def infer_column_kind(s: pd.Series) -> str:
    """One of: numeric, categorical, boolean, datetime, text."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    non_null = s.dropna()
    if non_null.empty:
        return "categorical"
    as_str = non_null.astype(str)
    sample = as_str.sample(min(len(as_str), 500), random_state=0)
    if pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.95:
        return "numeric"
    if sample.str.match(_DATE_HINT).mean() >= 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return "datetime"
    if as_str.str.len().mean() > 40 and as_str.nunique() / len(as_str) > 0.5:
        return "text"
    return "categorical"


def file_details(df: pd.DataFrame, meta: dict) -> dict:
    kinds = {c: infer_column_kind(df[c]) for c in df.columns}
    total_cells = int(df.size)
    missing = int(df.isna().sum().sum())
    return {
        "file_name": meta["file_name"],
        "file_type": meta["file_type"],
        "file_size_bytes": meta["file_size_bytes"],
        "stored_path": meta["stored_path"],
        "uploaded_at": meta["uploaded_at"],
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "missing_cells": missing,
        "missing_pct": round(100 * missing / total_cells, 2) if total_cells else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "edited_cells": meta.get("edited_cells", 0),
        "column_types": [
            {"name": c, "pandas_dtype": str(df[c].dtype), "kind": kinds[c]} for c in df.columns
        ],
    }


def _num(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), 4)


def describe(df: pd.DataFrame) -> list:
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        kind = infer_column_kind(s)
        row = {
            "column": c,
            "kind": kind,
            "count": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2) if n else 0,
            "unique": int(s.nunique(dropna=True)),
        }
        if kind == "numeric":
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x):
                q = x.quantile([0.25, 0.5, 0.75])
                row.update(
                    mean=_num(x.mean()), std=_num(x.std()), min=_num(x.min()),
                    p25=_num(q[0.25]), median=_num(q[0.5]), p75=_num(q[0.75]),
                    max=_num(x.max()), skew=_num(x.skew()) if len(x) > 2 else None,
                )
        else:
            vc = s.astype(str)[s.notna()].value_counts()
            if len(vc):
                row.update(top=str(vc.index[0]), top_freq=int(vc.iloc[0]))
        out.append(row)
    return out


def chart_data(df: pd.DataFrame, max_bins: int = 20, max_categories: int = 12) -> list:
    charts = []
    for c in df.columns:
        s = df[c]
        kind = infer_column_kind(s)
        if kind == "numeric":
            x = pd.to_numeric(s, errors="coerce").dropna()
            if x.empty:
                continue
            bins = int(min(max_bins, max(1, x.nunique())))
            counts, edges = np.histogram(x, bins=bins)
            labels = [f"{_fmt(edges[i])} – {_fmt(edges[i + 1])}" for i in range(len(counts))]
            charts.append({"column": c, "kind": kind, "chart": "histogram",
                           "labels": labels, "values": counts.tolist()})
        elif kind == "datetime":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(s, errors="coerce", format="mixed").dropna()
            if d.empty:
                continue
            vc = d.dt.to_period("M").value_counts().sort_index()
            if len(vc) > 36:
                vc = d.dt.to_period("Y").value_counts().sort_index()
            charts.append({"column": c, "kind": kind, "chart": "timeline",
                           "labels": [str(p) for p in vc.index], "values": vc.astype(int).tolist()})
        else:
            vc = s.astype(str)[s.notna()].value_counts()
            labels = [str(v) for v in vc.index[:max_categories]]
            values = vc.iloc[:max_categories].astype(int).tolist()
            if len(vc) > max_categories:
                labels.append(f"Other ({len(vc) - max_categories})")
                values.append(int(vc.iloc[max_categories:].sum()))
            charts.append({"column": c, "kind": kind, "chart": "bar", "labels": labels, "values": values})
        charts[-1]["missing"] = int(s.isna().sum())
    return charts


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}"
