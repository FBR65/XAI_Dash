"""
ML Model Manager für das XAI Dashboard
"""

import os
import pickle
from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ..config import Config
from ..models import ModelInfo


class ModelManager:
    """Verwaltet ML-Modelle für das XAI Dashboard"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_infos: Dict[str, ModelInfo] = {}
        Config.ensure_directories()
        self._load_models()

    def _load_models(self):
        """Lädt alle verfügbaren Modelle"""
        # Lade gespeicherte Modelle
        if os.path.exists(Config.MODEL_DIR):
            for filename in os.listdir(Config.MODEL_DIR):
                if filename.endswith(".pkl"):
                    model_name = filename[:-4]  # Entferne .pkl
                    try:
                        self._load_model(model_name)
                    except Exception as e:
                        print(f"Fehler beim Laden des Modells {model_name}: {e}")

        # Erstelle Demo-Modelle falls keine vorhanden sind
        if not self.models:
            self._create_demo_models()

    def _load_model(self, model_name: str):
        """Lädt ein einzelnes Modell"""
        model_path = Config.get_model_path(model_name)
        info_path = model_path.replace(".pkl", "_info.pkl")

        # Lade Modell
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Lade Modell-Info
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                model_info = pickle.load(f)
        else:
            # Fallback: Erstelle minimale Info
            model_info = ModelInfo(
                name=model_name,
                model_type="unknown",
                feature_names=[
                    f"feature_{i}" for i in range(getattr(model, "n_features_in_", 10))
                ],
            )

        self.models[model_name] = model
        self.model_infos[model_name] = model_info

    def _create_demo_models(self):
        """Erstellt Demo-Modelle für Testzwecke"""
        print("Erstelle Demo-Modelle...")

        # 1. Iris Klassifikation
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        iris_model = RandomForestClassifier(n_estimators=100, random_state=42)
        iris_model.fit(X_train, y_train)

        iris_info = ModelInfo(
            name="iris_classifier",
            model_type="classification",
            feature_names=iris.feature_names,
            target_names=iris.target_names.tolist(),
            description="Iris Blüten Klassifikation (Setosa, Versicolor, Virginica)",
        )

        self.save_model("iris_classifier", iris_model, iris_info)

        # 2. Wine Klassifikation
        wine = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            wine.data, wine.target, test_size=0.2, random_state=42
        )

        wine_model = RandomForestClassifier(n_estimators=100, random_state=42)
        wine_model.fit(X_train, y_train)

        wine_info = ModelInfo(
            name="wine_classifier",
            model_type="classification",
            feature_names=wine.feature_names,
            target_names=wine.target_names.tolist(),
            description="Wein-Qualitäts Klassifikation",
        )

        self.save_model("wine_classifier", wine_model, wine_info)

        # 3. Breast Cancer Klassifikation
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, test_size=0.2, random_state=42
        )

        cancer_model = RandomForestClassifier(n_estimators=100, random_state=42)
        cancer_model.fit(X_train, y_train)

        cancer_info = ModelInfo(
            name="breast_cancer_classifier",
            model_type="classification",
            feature_names=cancer.feature_names,
            target_names=cancer.target_names.tolist(),
            description="Brustkrebs Diagnose (Bösartig vs. Gutartig)",
        )

        self.save_model("breast_cancer_classifier", cancer_model, cancer_info)

    def save_model(self, name: str, model: Any, model_info: ModelInfo):
        """Speichert ein Modell und lädt es in den Manager"""
        model_path = Config.get_model_path(name)
        info_path = model_path.replace(".pkl", "_info.pkl")

        # Speichere Modell
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Speichere Modell-Info
        with open(info_path, "wb") as f:
            pickle.dump(model_info, f)

        # Lade in Memory
        self.models[name] = model
        self.model_infos[name] = model_info

        print(f"Modell '{name}' gespeichert und geladen.")

    def get_model(self, name: str) -> Optional[Any]:
        """Gibt ein Modell zurück"""
        return self.models.get(name)

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Gibt Modell-Informationen zurück"""
        return self.model_infos.get(name)

    def list_models(self) -> List[str]:
        """Gibt eine Liste aller verfügbaren Modelle zurück"""
        return list(self.models.keys())

    def predict(
        self, model_name: str, features: Dict[str, Union[float, int, str]]
    ) -> Dict[str, Any]:
        """Führt eine Vorhersage durch"""
        model = self.get_model(model_name)
        model_info = self.get_model_info(model_name)

        if model is None or model_info is None:
            raise ValueError(f"Modell '{model_name}' nicht gefunden")

        # Konvertiere Features zu Array
        feature_array = self._features_to_array(features, model_info.feature_names)

        # Vorhersage
        prediction = model.predict(feature_array.reshape(1, -1))[0]

        # Wahrscheinlichkeiten (falls Klassifikation)
        probabilities = None
        if (
            hasattr(model, "predict_proba")
            and model_info.model_type == "classification"
        ):
            proba = model.predict_proba(feature_array.reshape(1, -1))[0]
            if model_info.target_names:
                probabilities = {
                    str(model_info.target_names[i]): float(proba[i])
                    for i in range(len(proba))
                }

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "model_name": model_name,
            "features_used": features,
        }

    def _features_to_array(
        self, features: Dict[str, Union[float, int, str]], feature_names: List[str]
    ) -> np.ndarray:
        """Konvertiert Feature-Dict zu numpy Array"""
        feature_array = np.zeros(len(feature_names))

        for i, feature_name in enumerate(feature_names):
            if feature_name in features:
                value = features[feature_name]
                # Konvertiere zu float
                if isinstance(value, (int, float)):
                    feature_array[i] = float(value)
                else:
                    # Für kategorische Features - einfache Implementierung
                    feature_array[i] = 0.0
            else:
                # Setze fehlende Features auf 0 oder Median
                feature_array[i] = 0.0

        return feature_array

    def register_model(self, model_name: str, model: Any, model_info: ModelInfo):
        """Registriert ein neues Modell zur Laufzeit"""
        self.models[model_name] = model
        self.model_infos[model_name] = model_info
        print(f"✅ Modell '{model_name}' erfolgreich registriert")

    def add_model(self, model_name: str, model: Any, model_info: ModelInfo):
        """Alias für register_model"""
        self.register_model(model_name, model, model_info)

    def remove_model(self, model_name: str) -> bool:
        """Entfernt ein Modell aus dem Manager"""
        try:
            if model_name in self.models:
                del self.models[model_name]
            if model_name in self.model_infos:
                del self.model_infos[model_name]

            # Entferne auch gespeicherte Dateien falls vorhanden
            model_path = Config.get_model_path(model_name)
            info_path = model_path.replace(".pkl", "_info.pkl")

            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(info_path):
                os.remove(info_path)

            print(f"✅ Modell '{model_name}' erfolgreich entfernt")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Entfernen des Modells: {e}")
            return False


# Global verfügbare Instanz
model_manager = ModelManager()
