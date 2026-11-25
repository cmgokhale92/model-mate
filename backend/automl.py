"""BACKEND JOB 1 — the AutoML training engine.

Steps
 1. Clean: drop rows with no target, exact duplicates, ID-like / constant / mostly-empty columns.
 2. Understand modalities: numeric, categorical, boolean, datetime (expanded to parts), free text (TF-IDF).
 3. Understand the target: binary / multiclass classification or regression; detect class imbalance.
 4. Choose a cross-validation strategy (stratified K-fold for classification, K-fold for regression,
    fewer folds + a selection sample for large data).
 5. Cross-validate a set of candidate models (incl. a naive baseline) on a shared preprocessing pipeline.
 6. Tune the best family with a randomized search, keep whichever scores higher.
 7. Refit on all rows, compute permutation feature importance, dump the full pipeline with joblib.
"""
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .profiling import infer_column_kind

RANDOM_STATE = 42
POSITIVE_LABELS = {"1", "1.0", "yes", "y", "true", "t", "churn", "churned", "positive", "pos"}
_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|(customer|policy|client|account)[ _]?(no|num|number|id)$)", re.I)


# --------------------------------------------------------------------------- preprocessing
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerces raw columns to their learned kinds so training and prediction see identical inputs."""

    def __init__(self, kinds=None):
        self.kinds = kinds or {}

    def fit(self, X, y=None):
        return self

    def output_columns(self):
        num, cat, txt = [], [], []
        for c, k in self.kinds.items():
            if k == "numeric":
                num.append(c)
            elif k == "datetime":
                num += [f"{c}__year", f"{c}__month", f"{c}__dayofweek", f"{c}__days_since_2000"]
            elif k == "text":
                txt.append(c)
            else:
                cat.append(c)
        return num, cat, txt

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = {}
        for c, k in self.kinds.items():
            s = X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
            if k == "numeric":
                out[c] = pd.to_numeric(s, errors="coerce").astype(float)
            elif k == "datetime":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = pd.to_datetime(s, errors="coerce", format="mixed")
                out[f"{c}__year"] = d.dt.year.astype(float)
                out[f"{c}__month"] = d.dt.month.astype(float)
                out[f"{c}__dayofweek"] = d.dt.dayofweek.astype(float)
                out[f"{c}__days_since_2000"] = (d - pd.Timestamp("2000-01-01")).dt.days.astype(float)
            elif k == "text":
                out[c] = s.fillna("").astype(str)
            else:  # categorical / boolean
                out[c] = s.map(lambda v: np.nan if pd.isna(v) else _canon(v)).astype(object)
        return pd.DataFrame(out, index=X.index)


def _canon(v):
    """Make 1, 1.0 and '1' the same category; strip whitespace."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        v = int(v)
    return str(v).strip()


def build_preprocessor(kinds: dict) -> Pipeline:
    fe = FeatureEngineer(kinds)
    num, cat, txt = fe.output_columns()
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=25,
                                     sparse_output=False)),
        ]), cat))
    for c in txt:
        transformers.append((f"txt_{c}", TfidfVectorizer(max_features=100, stop_words="english"), c))
    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return Pipeline([("features", fe), ("encode", ct)])


# --------------------------------------------------------------------------- data understanding
def detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "binary" if y.nunique() == 2 else "multiclass"
    n_unique = y.nunique()
    is_int_like = np.allclose(y, np.round(y))
    if n_unique == 2:
        return "binary"
    if is_int_like and n_unique <= 10 and n_unique / len(y) < 0.05:
        return "multiclass"
    return "regression"


def choose_positive_class(y: pd.Series):
    classes = list(pd.Series(y.unique()).sort_values(key=lambda s: s.astype(str)))
    for c in classes:
        if _canon(c).lower() in POSITIVE_LABELS:
            return c
    return y.value_counts().idxmin()  # minority class is usually the event of interest


def clean_and_profile(df: pd.DataFrame, target: str, log):
    decisions, dropped = [], []
    n0 = len(df)
    df = df[df[target].notna()].copy()
    if len(df) < n0:
        decisions.append(f"Dropped {n0 - len(df)} rows with a missing target value.")
    n1 = len(df)
    df = df.drop_duplicates()
    if len(df) < n1:
        decisions.append(f"Dropped {n1 - len(df)} exact duplicate rows.")
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} usable rows after cleaning; at least 20 are needed.")

    kinds = {}
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        miss = s.isna().mean()
        nunique = s.nunique(dropna=True)
        kind = infer_column_kind(s)
        if miss > 0.6:
            dropped.append({"column": c, "reason": f"{miss:.0%} missing values"})
        elif nunique <= 1:
            dropped.append({"column": c, "reason": "constant value"})
        elif _ID_NAME.search(c) and nunique / len(s) > 0.9:
            dropped.append({"column": c, "reason": "identifier column (unique per row)"})
        elif kind == "categorical" and nunique / len(s) > 0.95 and len(s) > 50:
            dropped.append({"column": c, "reason": "identifier-like text (unique per row)"})
        else:
            kinds[c] = kind
    if not kinds:
        raise ValueError("No usable feature columns remain after cleaning.")
    for d in dropped:
        decisions.append(f"Dropped column '{d['column']}': {d['reason']}.")
    counts = pd.Series(kinds).value_counts().to_dict()
    decisions.append("Feature modalities: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + ".")
    log("Cleaned data and detected feature types", 10)
    return df, kinds, dropped, decisions


# --------------------------------------------------------------------------- candidates
def candidates(task: str, class_weight):
    if task == "regression":
        return {
            "Baseline (mean)": DummyRegressor(),
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                   random_state=RANDOM_STATE),
            "Extra Trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                               random_state=RANDOM_STATE),
            "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        }
    return {
        "Baseline (majority class)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=class_weight),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                class_weight=class_weight, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                            class_weight=class_weight, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight=class_weight,
                                                            random_state=RANDOM_STATE),
    }


SEARCH_SPACES = {
    "Logistic Regression": {"model__C": loguniform(1e-3, 1e2)},
    "Ridge Regression": {"model__alpha": loguniform(1e-3, 1e2)},
    "Random Forest": {"model__n_estimators": randint(150, 400), "model__max_depth": [None, 6, 10, 16],
                      "model__min_samples_leaf": randint(1, 8), "model__max_features": ["sqrt", 0.5, 1.0]},
    "Extra Trees": {"model__n_estimators": randint(150, 400), "model__max_depth": [None, 6, 10, 16],
                    "model__min_samples_leaf": randint(1, 8), "model__max_features": ["sqrt", 0.5, 1.0]},
    "Gradient Boosting": {"model__learning_rate": loguniform(0.02, 0.3), "model__max_iter": randint(100, 400),
                          "model__max_leaf_nodes": randint(15, 63), "model__l2_regularization": uniform(0, 1),
                          "model__min_samples_leaf": randint(10, 60)},
}

SCORING = {
    "binary": ({"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1", "precision": "precision",
                "recall": "recall"}, "roc_auc"),
    "multiclass": ({"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
                   "f1_macro"),
    "regression": ({"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}, "r2"),
}


# --------------------------------------------------------------------------- main entry point
def train(df: pd.DataFrame, target: str, tune_iter: int = 8, selection_rows: int = 30000, log=None):
    """Returns (bundle, report). `bundle` is what gets joblib-dumped; `report` goes to the DB."""
    log = log or (lambda msg, pct=None: None)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    log("Starting training job", 2)

    df, kinds, dropped, decisions = clean_and_profile(df, target, log)
    y_raw = df[target]
    X = df[list(kinds)]
    task = detect_task(y_raw)

    classes, positive_class, class_weight = None, None, None
    if task == "binary":
        positive_class = choose_positive_class(y_raw)
        negative = [c for c in y_raw.unique() if c != positive_class][0]
        classes = [negative, positive_class]
        y = (y_raw == positive_class).astype(int).to_numpy()
        rate = y.mean()
        decisions.append(f"Task: binary classification. Positive class = '{positive_class}' ({rate:.1%} of rows).")
        if rate < 0.2 or rate > 0.8:
            class_weight = "balanced"
            decisions.append("Classes are imbalanced, so models use balanced class weights.")
    elif task == "multiclass":
        classes = sorted(y_raw.unique(), key=lambda v: str(v))
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).to_numpy()
        decisions.append(f"Task: multiclass classification with {len(classes)} classes.")
        if pd.Series(y).value_counts(normalize=True).min() < 0.05:
            class_weight = "balanced"
            decisions.append("Some classes are rare, so models use balanced class weights.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        decisions.append("Task: regression (numeric target).")

    scoring, primary = SCORING[task]

    # --- cross-validation strategy
    n = len(X)
    n_splits = 5 if n < 20000 else 3
    if task != "regression":
        min_class = int(pd.Series(y).value_counts().min())
        if min_class < 2:
            raise ValueError("Every target class needs at least 2 rows.")
        n_splits = max(2, min(n_splits, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        decisions.append(f"Cross-validation: stratified {n_splits}-fold (keeps class ratios in every fold).")
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        decisions.append(f"Cross-validation: shuffled {n_splits}-fold.")

    X_sel, y_sel = X, y
    if n > selection_rows:
        X_sel, _, y_sel, _ = train_test_split(
            X, y, train_size=selection_rows, random_state=RANDOM_STATE,
            stratify=y if task != "regression" else None)
        decisions.append(f"Model selection ran on a {selection_rows:,}-row sample; the final model uses all {n:,} rows.")
    decisions.append(f"Primary selection metric: {primary}.")

    # --- candidate evaluation
    pre = build_preprocessor(kinds)
    leaderboard = []
    cands = candidates(task, class_weight)
    for i, (name, est) in enumerate(cands.items()):
        log(f"Cross-validating {name}", 15 + int(50 * i / len(cands)))
        pipe = Pipeline([("prep", clone(pre)), ("model", est)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(pipe, X_sel, y_sel, cv=cv, scoring=scoring, error_score=np.nan)
        leaderboard.append({"model": name, **_summarise(res, scoring), "tuned": False})
    leaderboard.sort(key=lambda r: -np.inf if r[primary] is None else r[primary], reverse=True)
    best = leaderboard[0]
    best_name = best["model"]
    best_pipe = Pipeline([("prep", clone(pre)), ("model", clone(cands[best_name]))])
    log(f"Best candidate: {best_name} ({primary} = {best[primary]:.4f})", 65)

    # --- tuning
    if tune_iter > 0 and best_name in SEARCH_SPACES:
        log(f"Tuning {best_name} hyper-parameters ({tune_iter} trials)", 70)
        search = RandomizedSearchCV(best_pipe, SEARCH_SPACES[best_name], n_iter=tune_iter, cv=cv,
                                    scoring=scoring, refit=primary, random_state=RANDOM_STATE,
                                    error_score=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_sel, y_sel)
        idx = search.best_index_
        tuned = {"model": f"{best_name} (tuned)", "tuned": True}
        for k in scoring:
            m, s = search.cv_results_[f"mean_test_{k}"][idx], search.cv_results_[f"std_test_{k}"][idx]
            tuned[k], tuned[f"{k}_std"] = _fix_sign(k, m), round(float(s), 4)
        leaderboard.append(tuned)
        leaderboard.sort(key=lambda r: -np.inf if r[primary] is None else r[primary], reverse=True)
        if tuned[primary] is not None and tuned[primary] > best[primary]:
            best_pipe = search.best_estimator_
            best = tuned
            params = {k.replace("model__", ""): _jsonable(v) for k, v in search.best_params_.items()}
            decisions.append(f"Tuning improved {best_name}: {params}.")
        else:
            decisions.append(f"Tuning did not beat the default {best_name} settings; kept defaults.")

    # --- final fit
    log(f"Refitting {best['model']} on all {n:,} rows", 85)
    final = clone(best_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X, y)

    log("Computing feature importance", 92)
    importance = _importance(final, X, y, primary)

    schema = build_schema(X, kinds)
    metrics = {k: best[k] for k in scoring}
    report = {
        "task_type": task,
        "algorithm": best["model"],
        "primary_metric": primary,
        "primary_score": best[primary],
        "metrics": metrics,
        "leaderboard": leaderboard,
        "feature_importance": importance,
        "dropped_columns": dropped,
        "decisions": decisions,
        "schema": schema,
        "n_rows": n,
        "n_features": len(kinds),
    }
    bundle = {
        "pipeline": final,
        "task": task,
        "target": target,
        "classes": [_jsonable(c) for c in classes] if classes is not None else None,
        "positive_class": _jsonable(positive_class),
        "feature_columns": list(kinds),
        "schema": schema,
    }
    log("Training complete", 100)
    return bundle, report


# --------------------------------------------------------------------------- helpers
def _fix_sign(metric, value):
    if value is None or not np.isfinite(value):
        return None
    return round(float(-value if metric in ("rmse", "mae") else value), 4)


def _summarise(res, scoring):
    out = {}
    for k in scoring:
        vals = res[f"test_{k}"]
        out[k] = _fix_sign(k, np.nanmean(vals)) if not np.all(np.isnan(vals)) else None
        out[f"{k}_std"] = round(float(np.nanstd(vals)), 4) if not np.all(np.isnan(vals)) else None
    out["fit_seconds"] = round(float(np.sum(res["fit_time"])), 2)
    return out


def _importance(pipe, X, y, primary, max_rows=1500):
    if len(X) > max_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    scorer = {"rmse": "neg_root_mean_squared_error"}.get(primary, primary)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = permutation_importance(pipe, X, y, scoring=scorer, n_repeats=3, random_state=RANDOM_STATE)
    except Exception:  # importance is a nice-to-have; never fail training over it
        return []
    vals = np.clip(r.importances_mean, 0, None)
    total = vals.sum() or 1.0
    items = sorted(zip(X.columns, vals), key=lambda t: -t[1])
    return [{"feature": f, "importance": round(float(v / total), 4)} for f, v in items[:15]]


def build_schema(X: pd.DataFrame, kinds: dict) -> list:
    schema = []
    for c, k in kinds.items():
        s = X[c]
        entry = {"name": c, "kind": k, "required": True, "nullable": bool(s.isna().any())}
        non_null = s.dropna()
        entry["example"] = _jsonable(non_null.iloc[0]) if len(non_null) else None
        if k == "numeric":
            x = pd.to_numeric(s, errors="coerce")
            entry["min"], entry["max"] = _jsonable(x.min()), _jsonable(x.max())
        elif k in ("categorical", "boolean"):
            vals = non_null.map(_canon).value_counts().index.tolist()
            entry["allowed_values"] = vals[:25]
            entry["n_categories"] = len(vals)
        schema.append(entry)
    return schema


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v if isinstance(v, (int, str, bool, list, dict)) else str(v)


# --------------------------------------------------------------------------- prediction
def validate_input(df: pd.DataFrame, schema: list):
    missing = [f["name"] for f in schema if f["name"] not in df.columns]
    if missing:
        raise ValueError("The file is missing required column(s): " + ", ".join(missing))


def predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    validate_input(df, bundle["schema"])
    pipe = bundle["pipeline"]
    X = df[bundle["feature_columns"]]
    target = bundle["target"]
    out = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bundle["task"] == "regression":
            out[f"predicted_{target}"] = pipe.predict(X)
            return out
        proba = pipe.predict_proba(X)
    classes = bundle["classes"]
    pred_idx = proba.argmax(axis=1)
    out[f"predicted_{target}"] = [classes[i] for i in pred_idx]
    out["prediction_confidence"] = proba.max(axis=1).round(4)
    if bundle["task"] == "binary":
        pos = proba[:, 1]
        out[f"probability_{target}_{bundle['positive_class']}"] = pos.round(4)
        out["priority_rank"] = pd.Series(pos).rank(ascending=False, method="first").astype(int).to_numpy()
    else:
        for i, c in enumerate(classes):
            out[f"probability_{c}"] = proba[:, i].round(4)
    return out
