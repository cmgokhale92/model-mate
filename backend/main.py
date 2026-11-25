"""ModelMate API + static frontend.

Run:  uvicorn backend.main:app --reload     then open http://127.0.0.1:8000
"""
import io
import json
import pickle
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import automl, db
from .config import FRONTEND_DIR, MAX_UPLOAD_MB, MODEL_DIR, SELECTION_SAMPLE_ROWS, TUNE_ITER, UPLOAD_DIR
from .profiling import SUPPORTED_EXTENSIONS, chart_data, describe, df_records, file_details, read_table

app = FastAPI(title="ModelMate", version="0.1.0")
db.init_db()

JOBS: dict = {}            # job_id -> job state (in-memory; fine for an MVP single process)
MODEL_CACHE: dict = {}     # model_id -> loaded joblib bundle
_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== datasets
def _ds_paths(ds_id: str):
    return UPLOAD_DIR / f"{ds_id}.pkl", UPLOAD_DIR / f"{ds_id}.meta.json"


def _load_ds(ds_id: str):
    pkl, meta = _ds_paths(ds_id)
    if not pkl.exists():
        raise HTTPException(404, "Dataset not found. Please upload the file again.")
    with open(pkl, "rb") as f:
        df = pickle.load(f)
    return df, json.loads(meta.read_text())


def _save_ds(ds_id: str, df: pd.DataFrame, meta: dict):
    pkl, meta_p = _ds_paths(ds_id)
    with open(pkl, "wb") as f:
        pickle.dump(df, f)
    meta_p.write_text(json.dumps(meta))


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        df = read_table(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    ds_id = uuid.uuid4().hex[:12]
    stored = UPLOAD_DIR / f"{ds_id}_{Path(file.filename).name}"
    stored.write_bytes(content)
    meta = {
        "id": ds_id, "file_name": file.filename, "file_type": ext.lstrip(".").upper(),
        "file_size_bytes": len(content), "stored_path": str(stored), "uploaded_at": _now(), "edited_cells": 0,
    }
    _save_ds(ds_id, df, meta)
    return {"dataset_id": ds_id, "columns": list(df.columns), "details": file_details(df, meta)}


@app.get("/api/datasets/{ds_id}/rows")
def dataset_rows(ds_id: str, offset: int = 0, limit: int = 50):
    df, _ = _load_ds(ds_id)
    limit = max(1, min(limit, 500))
    page = df.iloc[offset: offset + limit]
    return {"total": len(df), "offset": offset, "columns": list(df.columns),
            "row_index": [int(i) for i in range(offset, offset + len(page))], "rows": df_records(page)}


class CellEdit(BaseModel):
    row: int
    column: str
    value: Optional[Any] = None


@app.patch("/api/datasets/{ds_id}/cells")
def edit_cells(ds_id: str, edits: List[CellEdit]):
    df, meta = _load_ds(ds_id)
    for e in edits:
        if e.column not in df.columns:
            raise HTTPException(400, f"Unknown column '{e.column}'.")
        if not 0 <= e.row < len(df):
            raise HTTPException(400, f"Row {e.row} is out of range.")
        val = e.value
        if isinstance(val, str) and val.strip() == "":
            val = None
        col = df[e.column]
        if val is None:
            if pd.api.types.is_integer_dtype(col) or pd.api.types.is_bool_dtype(col):
                df[e.column] = col.astype(object if pd.api.types.is_bool_dtype(col) else float)
            val = np.nan
        elif pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.isna(num):
                raise HTTPException(400, f"Column '{e.column}' is numeric; '{val}' is not a number.")
            if pd.api.types.is_integer_dtype(col) and not float(num).is_integer():
                df[e.column] = col.astype(float)
            val = num
        elif pd.api.types.is_datetime64_any_dtype(col):
            val = pd.to_datetime(val, errors="coerce")
        else:
            if not pd.api.types.is_object_dtype(col):
                df[e.column] = col.astype(object)
            val = str(val)
        df.iat[e.row, df.columns.get_loc(e.column)] = val
    meta["edited_cells"] = meta.get("edited_cells", 0) + len(edits)
    _save_ds(ds_id, df, meta)
    return {"ok": True, "edited_cells": meta["edited_cells"]}


@app.get("/api/datasets/{ds_id}/details")
def dataset_details(ds_id: str):
    df, meta = _load_ds(ds_id)
    return file_details(df, meta)


@app.get("/api/datasets/{ds_id}/stats")
def dataset_stats(ds_id: str):
    df, _ = _load_ds(ds_id)
    return {"stats": describe(df)}


@app.get("/api/datasets/{ds_id}/charts")
def dataset_charts(ds_id: str):
    df, _ = _load_ds(ds_id)
    return {"charts": chart_data(df)}


# =========================================================================== training (BACKEND JOB 1)
class TrainRequest(BaseModel):
    dataset_id: str
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    target: str


def _run_training(job_id: str, req: TrainRequest):
    job = JOBS[job_id]

    def log(msg, pct=None):
        with _lock:
            job["log"].append({"time": _now(), "message": msg})
            if pct is not None:
                job["progress"] = pct

    try:
        job["status"] = "running"
        df, meta = _load_ds(req.dataset_id)
        bundle, report = automl.train(df, req.target, tune_iter=TUNE_ITER,
                                      selection_rows=SELECTION_SAMPLE_ROWS, log=log)
        model_id = uuid.uuid4().hex[:10]
        artifact = MODEL_DIR / f"{model_id}.joblib"
        bundle.update(model_id=model_id, name=req.name, created_at=_now())
        joblib.dump(bundle, artifact)
        db.insert_model({
            "id": model_id, "name": req.name.strip(), "description": req.description.strip(),
            "target": req.target, "artifact_path": str(artifact), "dataset_name": meta["file_name"],
            "created_at": bundle["created_at"],
            **{k: report[k] for k in ("task_type", "algorithm", "primary_metric", "primary_score", "n_rows",
                                       "n_features", "schema", "metrics", "leaderboard", "feature_importance",
                                       "dropped_columns", "decisions")},
        })
        # keep a few example rows so users can download a ready-to-fill template
        cols = [f["name"] for f in report["schema"]]
        df[cols].head(5).to_csv(MODEL_DIR / f"{model_id}.template.csv", index=False)
        with _lock:
            job.update(status="completed", progress=100, model_id=model_id, report=report)
    except Exception as e:
        traceback.print_exc()
        with _lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["log"].append({"time": _now(), "message": f"Failed: {e}"})


@app.post("/api/train", status_code=202)
def start_training(req: TrainRequest):
    df, _ = _load_ds(req.dataset_id)
    if req.target not in df.columns:
        raise HTTPException(400, f"Target column '{req.target}' not found in the dataset.")
    if df[req.target].nunique(dropna=True) < 2:
        raise HTTPException(400, "The target column needs at least two distinct values.")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0, "log": [], "model_id": None,
                    "error": None, "report": None, "created_at": _now()}
    threading.Thread(target=_run_training, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    with _lock:
        return json.loads(json.dumps(job))


# =========================================================================== models
_PUBLIC_FIELDS = ("id", "name", "description", "target", "task_type", "algorithm", "primary_metric",
                  "primary_score", "dataset_name", "n_rows", "n_features", "schema", "created_at")


@app.get("/api/models")
def list_models():
    return {"models": [{k: m[k] for k in _PUBLIC_FIELDS} for m in db.list_models()]}


def _get_model_or_404(model_id: str):
    m = db.get_model(model_id)
    if not m:
        raise HTTPException(404, "Model not found.")
    return m


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    m = _get_model_or_404(model_id)
    m["loaded"] = model_id in MODEL_CACHE
    return m


def _load_bundle(model_id: str):
    if model_id not in MODEL_CACHE:
        m = _get_model_or_404(model_id)
        path = Path(m["artifact_path"])
        if not path.exists():
            raise HTTPException(410, "Model artifact file is missing on disk.")
        MODEL_CACHE[model_id] = joblib.load(path)
    return MODEL_CACHE[model_id]


@app.post("/api/models/{model_id}/load")
def load_model(model_id: str):
    m = _get_model_or_404(model_id)
    _load_bundle(model_id)
    return {"loaded": True, "model_id": model_id, "schema": m["schema"]}


@app.get("/api/models/{model_id}/template")
def model_template(model_id: str):
    m = _get_model_or_404(model_id)
    path = MODEL_DIR / f"{model_id}.template.csv"
    if not path.exists():
        pd.DataFrame(columns=[f["name"] for f in m["schema"]]).to_csv(path, index=False)
    return FileResponse(path, media_type="text/csv", filename=f"{_slug(m['name'])}_input_template.csv")


@app.post("/api/models/{model_id}/predict")
async def predict(model_id: str, file: UploadFile = File(...)):
    m = _get_model_or_404(model_id)
    bundle = _load_bundle(model_id)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    try:
        df = read_table(await file.read(), file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    try:
        out = automl.predict(bundle, df)
    except ValueError as e:
        raise HTTPException(400, str(e))

    pred_col = f"predicted_{m['target']}"
    summary = {"rows": len(out), "prediction_column": pred_col}
    if m["task_type"] == "regression":
        summary.update(mean=round(float(out[pred_col].mean()), 4), min=round(float(out[pred_col].min()), 4),
                       max=round(float(out[pred_col].max()), 4))
    else:
        summary["distribution"] = {str(k): int(v) for k, v in out[pred_col].value_counts().items()}

    stem = f"{_slug(m['name'])}_predictions"
    if ext == ".csv":
        buf = io.BytesIO(out.to_csv(index=False).encode("utf-8"))
        media, fname = "text/csv", f"{stem}.csv"
    else:
        buf = io.BytesIO()
        out.to_excel(buf, index=False)
        buf.seek(0)
        media, fname = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{stem}.xlsx"
    return StreamingResponse(buf, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{fname}"',
        "X-Prediction-Summary": json.dumps(summary),
        "Access-Control-Expose-Headers": "Content-Disposition, X-Prediction-Summary",
    })


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    m = _get_model_or_404(model_id)
    MODEL_CACHE.pop(model_id, None)
    for p in (Path(m["artifact_path"]), MODEL_DIR / f"{model_id}.template.csv"):
        if p.exists():
            p.unlink()
    db.delete_model(model_id)
    return {"deleted": model_id}


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower() or "model"


# =========================================================================== frontend
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
