"""
Konfigurationsdatei für das XAI Dashboard
"""

import os


class Config:
    """Zentrale Konfiguration für das XAI Dashboard"""

    # Server Konfiguration
    HOST = "127.0.0.1"
    PORT = 8000
    DEBUG = True

    # Modell Konfiguration
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    # XAI Konfiguration
    SUPPORTED_EXPLAINERS = ["lime", "permutation", "feature_importance"]

    # UI Konfiguration
    GRADIO_THEME = "default"
    GRADIO_SHARE = False

    # Feature Limits
    MAX_FEATURES = 50
    MAX_SAMPLES = 1000

    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Gibt den vollständigen Pfad zu einem Modell zurück"""
        return os.path.join(cls.MODEL_DIR, f"{model_name}.pkl")

    @classmethod
    def ensure_directories(cls):
        """Stellt sicher, dass alle notwendigen Verzeichnisse existieren"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
