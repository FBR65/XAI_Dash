"""
Model-agnostic Explainer für das XAI Dashboard
Unterstützt LIME, SHAP und Permutation Importance
"""

from typing import Dict, Any, List
import numpy as np
from lime import lime_tabular
import shap
from ..models import FeatureImportance


class ShapExplainer:
    """SHAP-basierte Erklärungen für verschiedene Modelltypen"""

    def __init__(self):
        self.explainer = None
        self.explainer_type = None

    def setup(
        self,
        model: Any,
        X_background: np.ndarray,
        model_type: str = "tree",
        feature_names: List[str] = None,
    ):
        """
        Initialisiert den SHAP Explainer

        Args:
            model: Das ML-Modell
            X_background: Hintergrunddaten für die Erklärung
            model_type: Typ des Modells ("tree", "linear", "kernel")
            feature_names: Namen der Features
        """
        try:
            if model_type == "tree" and hasattr(model, "feature_importances_"):
                # Für Tree-basierte Modelle (RandomForest, GradientBoosting, etc.)
                self.explainer = shap.TreeExplainer(model)
                self.explainer_type = "tree"
            elif model_type == "linear":
                # Für lineare Modelle
                self.explainer = shap.LinearExplainer(model, X_background)
                self.explainer_type = "linear"
            else:
                # Fallback: Kernel-Explainer (model-agnostic, aber langsamer)
                self.explainer = shap.KernelExplainer(
                    model.predict, X_background[: min(50, len(X_background))]
                )
                self.explainer_type = "kernel"

        except Exception as e:
            print(f"Warnung: SHAP Explainer Setup fehlgeschlagen: {e}")
            # Fallback zu Kernel-Explainer
            self.explainer = shap.KernelExplainer(
                model.predict, X_background[: min(10, len(X_background))]
            )
            self.explainer_type = "kernel"

    def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: List[str],
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Erklärt eine einzelne Vorhersage mit SHAP

        Args:
            instance: Die zu erklärende Instanz
            feature_names: Namen der Features
            num_features: Anzahl der wichtigsten Features

        Returns:
            Dict mit SHAP-Erklärungsdaten
        """
        if self.explainer is None:
            raise ValueError(
                "SHAP Explainer nicht initialisiert. Rufen Sie setup() auf."
            )

        # Berechne SHAP-Werte
        try:
            shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        except Exception as e:
            # Fallback für problematische Modelle
            print(f"SHAP Berechnung fehlgeschlagen: {e}")
            # Erstelle dummy Werte
            values = np.zeros(len(feature_names))
        else:
            # Behandle verschiedene SHAP-Ausgabeformate
            try:
                if isinstance(shap_values, list):
                    # Multi-class Classification
                    if len(shap_values) > 1:
                        # Nehme erste Klasse für Einfachheit
                        values = np.array(shap_values[0]).flatten()
                    else:
                        values = np.array(shap_values[0]).flatten()
                else:
                    # Binary classification oder Regression
                    values = np.array(shap_values).flatten()

                # Stelle sicher, dass wir die richtige Anzahl Werte haben
                if len(values) != len(feature_names):
                    if len(values) > len(feature_names):
                        values = values[: len(feature_names)]
                    else:
                        # Padding mit Nullen
                        padded_values = np.zeros(len(feature_names))
                        padded_values[: len(values)] = values
                        values = padded_values

            except Exception as e:
                print(f"SHAP Werte-Extraktion fehlgeschlagen: {e}")
                values = np.zeros(len(feature_names))

        # Behandle verschiedene SHAP-Ausgabeformate
        if isinstance(shap_values, list):
            # Multi-class Classification: nehme erste Klasse oder wichtigste
            if len(shap_values) > 1:
                # Für Multi-Class: nehme die Klasse mit den höchsten absoluten SHAP-Werten
                try:
                    max_class_idx = np.argmax(
                        [np.sum(np.abs(sv[0])) for sv in shap_values]
                    )
                    values = shap_values[max_class_idx][0]
                except (IndexError, TypeError):
                    # Fallback: erste Klasse
                    values = shap_values[0]
                    if hasattr(values, "shape") and len(values.shape) > 1:
                        values = values[0]
            else:
                values = shap_values[0]
                if hasattr(values, "shape") and len(values.shape) > 1:
                    values = values[0]
        else:
            # Binary classification oder Regression
            if hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
                values = shap_values[0]
            else:
                values = shap_values

        # Erstelle Feature-Wichtigkeiten
        feature_importances = []
        for i, (feature_name, shap_value) in enumerate(zip(feature_names, values)):
            feature_importances.append(
                FeatureImportance(
                    feature_name=feature_name,
                    importance=float(shap_value),
                    rank=0,  # Wird später gesetzt
                )
            )

        # Sortiere nach absoluter Wichtigkeit
        feature_importances.sort(key=lambda x: abs(x.importance), reverse=True)

        # Setze Ranks
        for i, fi in enumerate(feature_importances):
            fi.rank = i + 1

        # Gebe nur die wichtigsten Features zurück
        top_features = feature_importances[:num_features]

        return {
            "feature_importances": top_features,
            "shap_values": values.tolist(),
            "base_value": getattr(self.explainer, "expected_value", 0),
            "explainer_type": self.explainer_type,
            "method": "shap",
        }


class LimeExplainer:
    """LIME-basierte Erklärungen für Tabular-Daten"""

    def __init__(self):
        self.explainer = None

    def setup(
        self,
        X_train: np.ndarray,
        feature_names: List[str],
        mode: str = "classification",
    ):
        """Initialisiert den LIME Explainer mit Trainingsdaten"""
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names, mode=mode, discretize_continuous=True
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: callable,
        feature_names: List[str],
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Erklärt eine einzelne Vorhersage

        Args:
            instance: Die zu erklärende Instanz
            predict_fn: Funktion die Vorhersagen macht
            feature_names: Namen der Features
            num_features: Anzahl der wichtigsten Features

        Returns:
            Dict mit Erklärungsdaten
        """
        if self.explainer is None:
            # Fallback: Erstelle temporären Explainer
            self.setup(np.array([instance]), feature_names, mode="classification")

        # Erhalte LIME Erklärung
        explanation = self.explainer.explain_instance(
            instance, predict_fn, num_features=min(num_features, len(feature_names))
        )

        # Extrahiere Feature-Wichtigkeiten
        feature_importances = []
        explanation_list = explanation.as_list()

        for i, (feature_desc, importance) in enumerate(explanation_list):
            # Extrahiere Feature-Namen aus der Beschreibung
            feature_name = self._extract_feature_name(feature_desc, feature_names)

            feature_importances.append(
                FeatureImportance(
                    feature_name=feature_name, importance=float(importance), rank=i + 1
                )
            )

        return {
            "feature_importances": feature_importances,
            "explanation_html": explanation.as_html(),
            "local_exp": explanation.local_exp,
            "intercept": explanation.intercept,
            "score": explanation.score,
            "method": "lime",
        }

    def _extract_feature_name(self, feature_desc: str, feature_names: List[str]) -> str:
        """Extrahiert den Feature-Namen aus der LIME-Beschreibung"""
        # LIME gibt manchmal Beschreibungen wie "feature_name <= 5.2" zurück
        # Wir versuchen den ursprünglichen Feature-Namen zu finden
        for name in feature_names:
            if name in feature_desc:
                return name

        # Fallback: Verwende die erste Wort der Beschreibung
        return feature_desc.split()[0] if feature_desc else "unknown"


class PermutationExplainer:
    """Permutation-basierte Feature-Wichtigkeit"""

    def explain_instance(
        self,
        instance: np.ndarray,
        model: Any,
        X_background: np.ndarray,
        feature_names: List[str],
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Erklärt eine Instanz durch Permutation Feature Importance

        Args:
            instance: Die zu erklärende Instanz
            model: Das ML-Modell
            X_background: Hintergrunddaten für Permutation
            feature_names: Namen der Features
            num_features: Anzahl der wichtigsten Features

        Returns:
            Dict mit Erklärungsdaten
        """
        # Baseline Score
        baseline_pred = model.predict(instance.reshape(1, -1))[0]

        # Berechne Permutation Importance für jedes Feature
        feature_importances = []

        for i, feature_name in enumerate(feature_names):
            # Erstelle permutierte Version
            permuted_instance = instance.copy()

            # Permutiere das Feature mit Werten aus den Hintergrunddaten
            if len(X_background) > 0:
                random_idx = np.random.randint(0, len(X_background))
                permuted_instance[i] = X_background[random_idx, i]
            else:
                # Fallback: Setze auf 0
                permuted_instance[i] = 0

            # Vorhersage mit permutiertem Feature
            permuted_pred = model.predict(permuted_instance.reshape(1, -1))[0]

            # Importance als Differenz zur Baseline
            importance = abs(float(baseline_pred) - float(permuted_pred))

            feature_importances.append(
                FeatureImportance(
                    feature_name=feature_name,
                    importance=importance,
                    rank=0,  # Wird später gesetzt
                )
            )

        # Sortiere nach Wichtigkeit
        feature_importances.sort(key=lambda x: x.importance, reverse=True)

        # Setze Ranks
        for i, fi in enumerate(feature_importances):
            fi.rank = i + 1

        # Gebe nur die wichtigsten Features zurück
        top_features = feature_importances[:num_features]

        return {
            "feature_importances": top_features,
            "baseline_prediction": baseline_pred,
            "method": "permutation_importance",
        }


class ModelAgnosticExplainer:
    """Hauptklasse für model-agnostische Erklärungen"""

    def __init__(self):
        self.lime_explainer = LimeExplainer()
        self.shap_explainer = ShapExplainer()
        self.permutation_explainer = PermutationExplainer()

    def explain(
        self,
        method: str,
        instance: np.ndarray,
        model: Any,
        feature_names: List[str],
        X_background: np.ndarray = None,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Erstellt Erklärungen basierend auf der gewählten Methode

        Args:
            method: Erklärungsmethode ("lime", "shap", "permutation")
            instance: Zu erklärende Instanz
            model: ML-Modell
            feature_names: Feature-Namen
            X_background: Hintergrunddaten
            num_features: Anzahl der wichtigsten Features

        Returns:
            Erklärungsdaten
        """
        if method == "lime":
            if X_background is not None:
                self.lime_explainer.setup(X_background, feature_names)

            return self.lime_explainer.explain_instance(
                instance,
                model.predict_proba
                if hasattr(model, "predict_proba")
                else model.predict,
                feature_names,
                num_features,
            )

        elif method == "shap":
            if X_background is None:
                # Erstelle dummy Hintergrunddaten
                X_background = np.zeros((10, len(feature_names)))

            # Bestimme Modelltyp automatisch
            model_type = "tree" if hasattr(model, "feature_importances_") else "kernel"

            self.shap_explainer.setup(model, X_background, model_type, feature_names)

            return self.shap_explainer.explain_instance(
                instance, feature_names, num_features
            )

        elif method == "permutation":
            if X_background is None:
                # Erstelle dummy Hintergrunddaten
                X_background = np.zeros((10, len(feature_names)))

            return self.permutation_explainer.explain_instance(
                instance, model, X_background, feature_names, num_features
            )

        else:
            raise ValueError(f"Unbekannte Erklärungsmethode: {method}")


# Global verfügbare Instanz
explainer = ModelAgnosticExplainer()
