"""Aplikasi Web Penelitian & API Prediksi Kuat Tekan Beton (FastAPI)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from utils import FEATURES, LABELS

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DEFAULT_ARTIFACTS = ROOT / "artifacts"


class ConcreteMixRow(BaseModel):
    cement: float = Field(..., description="Semen (kg/m3)")
    slag: float = Field(..., description="Blast Furnace Slag (kg/m3)")
    fly_ash: float = Field(..., description="Fly Ash (kg/m3)")
    water: float = Field(..., description="Air (kg/m3)")
    superplasticizer: float = Field(..., description="Superplasticizer (kg/m3)")
    coarse_agg: float = Field(..., description="Agregat Kasar (kg/m3)")
    fine_agg: float = Field(..., description="Agregat Halus (kg/m3)")
    age: int = Field(..., description="Umur curing beton (hari, 1-365)")

    @field_validator("*")
    @classmethod
    def check_finite(cls, v: float | int) -> float | int:
        if not math.isfinite(float(v)):
            raise ValueError("Nilai harus berupa angka hingga (bukan NaN/Inf).")
        return v

    @model_validator(mode="after")
    def check_physical_bounds(self) -> "ConcreteMixRow":
        if self.cement <= 0:
            raise ValueError("cement harus lebih besar dari 0 kg/m³.")
        if self.water <= 0:
            raise ValueError("water harus lebih besar dari 0 kg/m³.")
        for key in ("slag", "fly_ash", "superplasticizer", "coarse_agg", "fine_agg"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} tidak boleh bernilai negatif.")
        if self.age < 1 or self.age > 365:
            raise ValueError("age harus bilangan bulat antara 1 hingga 365 hari.")
        return self


class PredictBatchRequest(BaseModel):
    rows: list[ConcreteMixRow] = Field(..., min_length=1, max_length=500)


def _check_domain_warnings(row: dict[str, Any], ranges: dict[str, list[float]]) -> list[str]:
    warnings: list[str] = []
    for feat in FEATURES:
        bounds = ranges.get(feat)
        if bounds and len(bounds) >= 2:
            lo, hi = float(bounds[0]), float(bounds[1])
            val = float(row[feat])
            if val < lo or val > hi:
                label = LABELS.get(feat, feat)
                warnings.append(
                    f"{label} ({val:g}) di luar rentang data latih [{lo:g}, {hi:g}]."
                )
    binder = float(row["cement"]) + float(row["slag"]) + float(row["fly_ash"])
    wb = float(row["water"]) / max(binder, 1e-6)
    if wb < 0.22 or wb > 0.90:
        warnings.append(
            f"Rasio air/binder (W/B = {wb:.3f}) berada pada zona ekstrem di luar distribusi umum data latih."
        )
    return warnings


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    art_root = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACTS
    application = FastAPI(
        title="Laboratorium Prediksi Kuat Tekan Beton",
        description=(
            "Antarmuka penelitian & API evaluasi bebas kebocoran data: "
            "studi duplikat, seleksi model pohon boosting, dan analisis galat per kelompok umur."
        ),
        version="2.0.0",
    )

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _load_bundle() -> dict[str, Any]:
        bundle_path = art_root / "models" / "bundle.joblib"
        if not bundle_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Artefak model belum tersedia. Jalankan experiment.py terlebih dahulu.",
            )
        # Safe local model artifact generated exclusively by experiment.py in this repository.
        return joblib.load(bundle_path)

    def _load_report() -> dict[str, Any]:
        report_path = art_root / "results" / "report.json"
        if not report_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Laporan eksperimen belum tersedia. Jalankan experiment.py terlebih dahulu.",
            )
        return json.loads(report_path.read_text(encoding="utf-8"))

    @application.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @application.get("/style.css", include_in_schema=False)
    def serve_css() -> FileResponse:
        return FileResponse(STATIC_DIR / "style.css", media_type="text/css")

    @application.get("/app.js", include_in_schema=False)
    def serve_js() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")

    @application.get("/fonts/{font_name}", include_in_schema=False)
    def serve_font(font_name: str) -> FileResponse:
        safe_name = Path(font_name).name
        font_path = STATIC_DIR / "fonts" / safe_name
        if safe_name != font_name or not font_path.is_file():
            raise HTTPException(status_code=404, detail="Berkas font tidak ditemukan.")
        return FileResponse(font_path, media_type="font/ttf")

    @application.get("/api/health")
    def api_health() -> dict[str, Any]:
        bundle = _load_bundle()
        report = _load_report()
        return {
            "status": "ok",
            "model": bundle.get("winner", report.get("winner")),
            "run_id": report.get("run_id"),
            "training_only": bool(bundle.get("training_only", False)),
        }

    @application.get("/api/report")
    def api_report() -> dict[str, Any]:
        return _load_report()

    @application.post("/api/predict")
    def api_predict(payload: PredictBatchRequest) -> dict[str, Any]:
        bundle = _load_bundle()
        model = bundle["model"]
        q = float(bundle.get("q", 0.0))
        ranges = bundle.get("ranges", {})
        winner = str(bundle.get("winner", "Model"))

        records = [item.model_dump() for item in payload.rows]
        df_in = pd.DataFrame(records)[FEATURES]
        try:
            preds = model.predict(df_in)
        except AttributeError as exc:
            # CatBoost versi lama dapat tidak menyediakan __sklearn_tags__ yang
            # diwajibkan oleh sklearn versi baru pada Pipeline.predict().
            # Bypass hanya validasi tag Pipeline; transformasi fitur dan model
            # CatBoost terlatih yang sama tetap digunakan untuk inferensi.
            steps = getattr(model, "steps", [])
            if "__sklearn_tags__" not in str(exc) or [name for name, _ in steps] != ["features", "model"]:
                raise
            preds = steps[-1][1].predict(steps[0][1].transform(df_in))

        results: list[dict[str, Any]] = []
        for row_dict, pred_val in zip(records, preds):
            point = float(pred_val)
            lower = max(0.0, point - q)
            upper = point + q
            binder = float(row_dict["cement"]) + float(row_dict["slag"]) + float(row_dict["fly_ash"])
            wb_ratio = float(row_dict["water"]) / max(binder, 1e-6)
            wc_ratio = float(row_dict["water"]) / max(float(row_dict["cement"]), 1e-6)
            warnings = _check_domain_warnings(row_dict, ranges)
            results.append(
                {
                    "prediction": point,
                    "lower": lower,
                    "upper": upper,
                    "water_binder_ratio": wb_ratio,
                    "water_cement_ratio": wc_ratio,
                    "out_of_domain": len(warnings) > 0,
                    "warnings": warnings,
                }
            )

        return {
            "model": winner,
            "conformal_level": float(bundle.get("level", 0.9)),
            "conformal_q": q,
            "predictions": results,
        }

    @application.get("/api/download/{filename}")
    def api_download(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        results_dir = (art_root / "results").resolve()
        target = (results_dir / safe_name).resolve()
        if (
            safe_name != filename
            or target.parent != results_dir
            or target.suffix.lower() != ".csv"
            or not target.is_file()
        ):
            raise HTTPException(status_code=404, detail="Berkas CSV tidak ditemukan.")
        return FileResponse(
            target,
            media_type="text/csv",
            filename=safe_name,
        )

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
