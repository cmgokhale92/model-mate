"""End-to-end test: upload -> profile -> edit -> train (BACKEND JOB 1) -> list models -> predict.

Run:  python -m pytest -v
Uses a temporary data directory so it never touches your real models.
"""
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data"

# isolate storage + keep tuning short before the app is imported
os.environ["MODELMATE_DATA_DIR"] = tempfile.mkdtemp(prefix="modelmate_test_")
os.environ["MODELMATE_TUNE_ITER"] = "2"
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from backend import automl  # noqa: E402
from backend.main import app  # noqa: E402

client = TestClient(app)


def _upload(path: Path):
    with open(path, "rb") as f:
        r = client.post("/api/datasets", files={"file": (path.name, f, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def _wait(job_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    pytest.fail("training timed out")


def test_full_churn_workflow():
    # --- Screen 1: upload + the four tabs
    up = _upload(SAMPLE / "insurance_churn_train.csv")
    ds = up["dataset_id"]
    assert up["details"]["rows"] == 3000
    assert "churn" in up["columns"]

    rows = client.get(f"/api/datasets/{ds}/rows?offset=0&limit=10").json()
    assert len(rows["rows"]) == 10

    stats = {s["column"]: s for s in client.get(f"/api/datasets/{ds}/stats").json()["stats"]}
    assert stats["age"]["kind"] == "numeric" and stats["age"]["min"] >= 21
    assert stats["churn"]["unique"] == 2

    charts = {c["column"]: c for c in client.get(f"/api/datasets/{ds}/charts").json()["charts"]}
    assert charts["age"]["chart"] == "histogram"
    assert charts["policy_type"]["chart"] == "bar"
    assert charts["last_contact_date"]["chart"] == "timeline"

    # editable fields: a valid edit is persisted, a bad numeric edit is rejected
    assert client.patch(f"/api/datasets/{ds}/cells", json=[{"row": 0, "column": "age", "value": "45"}]).status_code == 200
    assert client.get(f"/api/datasets/{ds}/rows?offset=0&limit=1").json()["rows"][0]["age"] == 45
    assert client.patch(f"/api/datasets/{ds}/cells", json=[{"row": 0, "column": "age", "value": "abc"}]).status_code == 400

    # --- BACKEND JOB 1
    r = client.post("/api/train", json={"dataset_id": ds, "name": "Churn test model",
                                         "description": "pytest", "target": "churn"})
    assert r.status_code == 202
    job = _wait(r.json()["job_id"])
    assert job["status"] == "completed", job.get("error")
    report = job["report"]
    assert report["task_type"] == "binary"
    assert report["primary_metric"] == "roc_auc"
    assert report["primary_score"] > 0.70, "model should clearly beat chance on the sample data"
    assert {"column": "customer_id", "reason": "identifier column (unique per row)"} in report["dropped_columns"]
    baseline = next(x for x in report["leaderboard"] if x["model"].startswith("Baseline"))
    assert report["primary_score"] > baseline["roc_auc"]

    # --- Screen 2: model registry row (name, description, schema, artifact)
    models = client.get("/api/models").json()["models"]
    m = next(x for x in models if x["id"] == job["model_id"])
    assert m["name"] == "Churn test model" and m["description"] == "pytest"
    schema_cols = {f["name"] for f in m["schema"]}
    assert "churn" not in schema_cols and "customer_id" not in schema_cols
    detail = client.get(f"/api/models/{m['id']}").json()
    assert Path(detail["artifact_path"]).exists()
    assert client.post(f"/api/models/{m['id']}/load").json()["loaded"] is True

    template = client.get(f"/api/models/{m['id']}/template")
    assert template.status_code == 200
    assert set(pd.read_csv(io.StringIO(template.text)).columns) == schema_cols

    # --- predict on unseen customers -> downloadable CSV
    with open(SAMPLE / "insurance_churn_new_customers.csv", "rb") as f:
        pr = client.post(f"/api/models/{m['id']}/predict", files={"file": ("new.csv", f, "text/csv")})
    assert pr.status_code == 200, pr.text
    assert "attachment" in pr.headers["content-disposition"]
    out = pd.read_csv(io.StringIO(pr.text))
    assert len(out) == 300
    for col in ("predicted_churn", "prediction_confidence", "probability_churn_Yes", "priority_rank"):
        assert col in out.columns
    assert set(out["predicted_churn"]) <= {"Yes", "No"}
    assert out["probability_churn_Yes"].between(0, 1).all()
    assert sorted(out["priority_rank"]) == list(range(1, 301))
    assert json.loads(pr.headers["x-prediction-summary"])["rows"] == 300

    # predictions are genuinely informative on held-out ground truth
    from sklearn.metrics import roc_auc_score
    truth = pd.read_csv(SAMPLE / "insurance_churn_new_customers_truth.csv")
    merged = out.merge(truth, on="customer_id")
    assert roc_auc_score(merged["churn"] == "Yes", merged["probability_churn_Yes"]) > 0.70

    # wrong schema -> clear 400 error
    bad = pd.read_csv(SAMPLE / "insurance_churn_new_customers.csv").drop(columns=["age", "auto_renew"])
    pr = client.post(f"/api/models/{m['id']}/predict",
                     files={"file": ("bad.csv", bad.to_csv(index=False).encode(), "text/csv")})
    assert pr.status_code == 400
    assert "age" in pr.json()["detail"] and "auto_renew" in pr.json()["detail"]

    # cleanup works
    assert client.delete(f"/api/models/{m['id']}").status_code == 200
    assert client.get(f"/api/models/{m['id']}").status_code == 404


def test_task_detection():
    assert automl.detect_task(pd.Series(["Yes", "No", "Yes"])) == "binary"
    assert automl.detect_task(pd.Series([0, 1, 1, 0])) == "binary"
    assert automl.detect_task(pd.Series(["Gold", "Silver", "Bronze"] * 10)) == "multiclass"
    assert automl.detect_task(pd.Series([1.5, 2.25, 3.1, 4.8] * 10)) == "regression"


def test_regression_target_trains():
    df = pd.read_csv(SAMPLE / "insurance_churn_train.csv").head(400)
    bundle, report = automl.train(df, "annual_premium", tune_iter=0)
    assert report["task_type"] == "regression" and report["primary_metric"] == "r2"
    preds = automl.predict(bundle, df.drop(columns="annual_premium").head(5))
    assert "predicted_annual_premium" in preds.columns


def test_rejects_unsupported_file():
    r = client.post("/api/datasets", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
