from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import SalaryPredictor

app = FastAPI(title="Salary Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    job_title: str
    experience_years: float = Field(ge=0)
    education_level: str
    skills_count: float = Field(ge=0)
    industry: str
    company_size: str
    location: str
    remote_work: str
    certifications: float = Field(ge=0)


class PredictResponse(BaseModel):
    predicted_salary: float


predictor: SalaryPredictor | None = None


@app.on_event("startup")
def startup() -> None:
    global predictor
    predictor = SalaryPredictor(
        checkpoint_path=os.getenv(
            "CHECKPOINT_PATH", "artifacts/checkpoint.pth"
        ),
        preprocessor_path=os.getenv(
            "PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"
        ),
        y_scaler_path=os.getenv(
            "Y_SCALER_PATH", "artifacts/y_scaler.pkl"
        ),
        device=os.getenv("MODEL_DEVICE"),
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Salary inference API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        prediction = predictor.predict(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(predicted_salary=round(prediction, 2))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
