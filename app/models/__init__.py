"""
Datenmodelle für das XAI Dashboard
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request-Modell für Vorhersagen"""

    features: Dict[str, Union[float, int, str]]
    model_name: str = "default"

    class Config:
        arbitrary_types_allowed = True


class PredictionResponse(BaseModel):
    """Response-Modell für Vorhersagen"""

    prediction: Union[float, int, str, List[float]]
    probability: Optional[Dict[str, float]] = None
    model_name: str
    features_used: Dict[str, Union[float, int, str]]

    class Config:
        arbitrary_types_allowed = True


class ExplanationRequest(BaseModel):
    """Request-Modell für Erklärungen"""

    features: Dict[str, Union[float, int, str]]
    model_name: str = "default"
    explainer_type: str = "lime"
    n_features: int = 10

    class Config:
        arbitrary_types_allowed = True


class FeatureImportance(BaseModel):
    """Modell für Feature-Wichtigkeit"""

    feature_name: str
    importance: float
    rank: int


class ExplanationResponse(BaseModel):
    """Response-Modell für Erklärungen"""

    prediction: Union[float, int, str, List[float]]
    feature_importances: List[FeatureImportance]
    explainer_type: str
    explanation_data: Dict[str, Any]
    model_name: str

    class Config:
        arbitrary_types_allowed = True


class ModelInfo(BaseModel):
    """Informationen über ein ML-Modell"""

    name: str
    model_type: str  # "classification", "regression"
    feature_names: List[str]
    target_names: Optional[List[str]] = None
    description: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
