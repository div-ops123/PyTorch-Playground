from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
import torch.nn as nn

RAW_FEATURE_COLUMNS = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
]

class SalaryModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["experience_level"] = pd.cut(
        features["experience_years"],
        bins=[0, 2, 5, 10, 50],
        labels=["Junior", "Mid", "Senior", "Expert"],
    )
    features["skill_density"] = features["skills_count"] / (features["experience_years"] + 1)
    features["cert_per_year"] = features["certifications"] / (features["experience_years"] + 1)
    features["total_qualifications"] = features["skills_count"] + features["certifications"]
    features["exp_x_skills"] = features["experience_years"] * features["skills_count"]
    features["exp_x_cert"] = features["experience_years"] * features["certifications"]
    features["is_tech"] = (features["industry"] == "Tech").astype(int)
    features["is_masters_plus"] = (
        features["education_level"].isin(["Master's", "PhD"]).astype(int)
    )
    return features


class SalaryPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/checkpoint.pth",
        preprocessor_path: str | Path = "artifacts/preprocessor.pkl",
        y_scaler_path: str | Path = "artifacts/y_scaler.pkl",
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = self._resolve_artifact_path(checkpoint_path)
        self.preprocessor_path = self._resolve_artifact_path(preprocessor_path)
        self.y_scaler_path = self._resolve_artifact_path(y_scaler_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.y_scaler = joblib.load(self.y_scaler_path)
        self.model = self._load_model()

    @staticmethod
    def _resolve_artifact_path(path: str | Path) -> Path:
        raw_path = Path(path)
        candidates = [raw_path]

        if not raw_path.is_absolute():
            candidates.append(Path("notebook") / raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        checked_paths = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Artifact not found. Checked paths: {checked_paths}")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            return checkpoint["model_state"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("Unsupported checkpoint format. Expected a state dict or model_state key.")

    @staticmethod
    def _infer_input_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
        if "model.0.weight" in state_dict and state_dict["model.0.weight"].ndim == 2:
            return int(state_dict["model.0.weight"].shape[1])

        for key, value in state_dict.items():
            if key.endswith(".weight") and value.ndim == 2:
                return int(value.shape[1])
        raise ValueError("Could not infer model input dimension from checkpoint state dict.")

    def _load_model(self) -> SalaryModel:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        input_dim = self._infer_input_dim_from_state_dict(state_dict)
        model = SalaryModel(input_dim=input_dim).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict(self, payload: dict[str, Any]) -> float:
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in payload]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(f"Missing required fields: {missing_fields}")

        row = pd.DataFrame([{k: payload[k] for k in RAW_FEATURE_COLUMNS}])
        row = engineer_features(row)
        model_input = row.drop(columns=["experience_level"], errors="ignore")

        transformed = self.preprocessor.transform(model_input)
        tensor_input = torch.tensor(transformed, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            scaled_prediction = self.model(tensor_input).cpu().numpy().reshape(-1, 1)

        salary = self.y_scaler.inverse_transform(scaled_prediction)[0, 0]
        return float(salary)
