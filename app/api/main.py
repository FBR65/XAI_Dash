"""
FastAPI Backend für das XAI Dashboard
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
import numpy as np

from ..models import PredictionRequest, ExplanationRequest, ModelInfo
from ..models.model_manager import model_manager
from ..explainers.model_agnostic import explainer


# FastAPI App
app = FastAPI(
    title="XAI Dashboard API",
    description="API für Explainable AI Dashboard",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root Endpoint"""
    return {"message": "XAI Dashboard API", "status": "running"}


@app.get("/models", response_model=List[str])
async def list_models():
    """Gibt eine Liste aller verfügbaren Modelle zurück"""
    return model_manager.list_models()


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Gibt Informationen zu einem spezifischen Modell zurück"""
    model_info = model_manager.get_model_info(model_name)
    if model_info is None:
        raise HTTPException(
            status_code=404, detail=f"Modell '{model_name}' nicht gefunden"
        )
    return model_info


@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Führt eine Vorhersage durch"""
    try:
        result = model_manager.predict(request.model_name, request.features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fehler bei der Vorhersage: {str(e)}"
        )


@app.post("/explain")
async def explain_prediction(request: ExplanationRequest) -> Dict[str, Any]:
    """Erstellt eine Erklärung für eine Vorhersage"""
    try:
        # Hole Modell und Modell-Info
        model = model_manager.get_model(request.model_name)
        model_info = model_manager.get_model_info(request.model_name)

        if model is None or model_info is None:
            raise HTTPException(
                status_code=404, detail=f"Modell '{request.model_name}' nicht gefunden"
            )

        # Konvertiere Features zu Array
        feature_array = np.zeros(len(model_info.feature_names))
        for i, feature_name in enumerate(model_info.feature_names):
            if feature_name in request.features:
                feature_array[i] = float(request.features[feature_name])

        # Erstelle Hintergrunddaten (für bessere Erklärungen)
        # In einer echten Anwendung würden diese gespeichert werden
        X_background = np.random.rand(100, len(model_info.feature_names))

        # Erstelle Erklärung
        explanation_data = explainer.explain(
            method=request.explainer_type,
            instance=feature_array,
            model=model,
            feature_names=model_info.feature_names,
            X_background=X_background,
            num_features=request.n_features,
        )

        # Erstelle Vorhersage
        prediction_result = model_manager.predict(request.model_name, request.features)

        return {
            "prediction": prediction_result["prediction"],
            "feature_importances": [
                fi.dict() for fi in explanation_data["feature_importances"]
            ],
            "explainer_type": request.explainer_type,
            "explanation_data": explanation_data,
            "model_name": request.model_name,
            "probabilities": prediction_result.get("probabilities"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fehler bei der Erklärung: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health Check Endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.list_models()),
        "available_explainers": ["lime", "shap", "permutation"],
    }


if __name__ == "__main__":
    import uvicorn
    from ..config import Config

    uvicorn.run(
        "app.api.main:app", host=Config.HOST, port=Config.PORT, reload=Config.DEBUG
    )
