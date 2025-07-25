"""
Erweiterte Dashboard-Version mit vollständigem Model-Management
Unterstützt Pickle-Upload, Dateisystem-Integration und Multi-Tab Interface
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import pickle
import os
import json
from typing import List, Any, Tuple, Dict, Optional
from pathlib import Path
import tempfile
import shutil

from ..models.model_manager import model_manager
from ..explainers.model_agnostic import explainer
from ..models import ModelInfo


class AdvancedXAIDashboard:
    """Erweiterte Dashboard-Klasse mit vollständigem Model-Management"""

    def __init__(self):
        self.current_model = None
        self.current_model_info = None
        self.models_directory = Path("uploaded_models")
        self.models_directory.mkdir(exist_ok=True)

        # Model metadata storage
        self.metadata_file = self.models_directory / "models_metadata.json"
        self.load_metadata()

    def load_metadata(self):
        """Lädt Model-Metadaten aus der JSON-Datei"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.models_metadata = json.load(f)
        else:
            self.models_metadata = {}

    def save_metadata(self):
        """Speichert Model-Metadaten in JSON-Datei"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.models_metadata, f, indent=2, ensure_ascii=False)

    def upload_model_file(
        self,
        file_path: str,
        model_name: str,
        model_type: str,
        description: str,
        feature_names: str,
        target_names: str,
    ) -> str:
        """Lädt eine Pickle/Joblib-Datei hoch und registriert das Modell"""
        try:
            if not file_path:
                return "Keine Datei ausgewählt"

            if not model_name:
                return "Modellname ist erforderlich"

            # Datei lesen und Modell laden
            try:
                if file_path.endswith(".pkl"):
                    with open(file_path, "rb") as f:
                        model = pickle.load(f)
                elif file_path.endswith(".joblib"):
                    model = joblib.load(file_path)
                else:
                    return "Unterstützte Formate: .pkl, .joblib"
            except Exception as e:
                return f"Fehler beim Laden der Datei: {str(e)}"

            # Feature-Namen verarbeiten
            if feature_names.strip():
                feature_list = [name.strip() for name in feature_names.split(",")]
            else:
                # Versuche Feature-Namen automatisch zu erkennen
                if hasattr(model, "feature_names_in_"):
                    feature_list = list(model.feature_names_in_)
                elif hasattr(model, "n_features_in_"):
                    feature_list = [
                        f"Feature_{i + 1}" for i in range(model.n_features_in_)
                    ]
                else:
                    return "Feature-Namen konnten nicht automatisch erkannt werden. Bitte manuell eingeben."

            # Target-Namen verarbeiten
            if target_names.strip():
                target_list = [name.strip() for name in target_names.split(",")]
            else:
                # Versuche Target-Namen automatisch zu erkennen
                if hasattr(model, "classes_"):
                    target_list = [str(cls) for cls in model.classes_]
                else:
                    target_list = None

            # Datei in unser Verzeichnis kopieren
            file_extension = Path(file_path).suffix
            new_file_path = self.models_directory / f"{model_name}{file_extension}"
            shutil.copy2(file_path, new_file_path)

            # ModelInfo erstellen
            model_info = ModelInfo(
                name=model_name,
                model_type=model_type,
                description=description,
                feature_names=feature_list,
                target_names=target_list,
            )

            # Modell registrieren
            model_manager.register_model(model_name, model, model_info)

            # Metadaten speichern
            self.models_metadata[model_name] = {
                "file_path": str(new_file_path),
                "model_type": model_type,
                "description": description,
                "feature_names": feature_list,
                "target_names": target_list,
                "upload_timestamp": pd.Timestamp.now().isoformat(),
            }
            self.save_metadata()

            return f"Modell '{model_name}' erfolgreich hochgeladen und registriert!\nGespeichert in: {new_file_path}"

        except Exception as e:
            return f"Fehler beim Upload: {str(e)}"

    def load_model_from_filesystem(self, file_path: str) -> str:
        """Lädt ein Modell direkt aus dem Dateisystem"""
        try:
            if not os.path.exists(file_path):
                return "Datei nicht gefunden"

            # Automatische Modell-Erkennung
            filename = Path(file_path).stem

            if file_path.endswith(".pkl"):
                with open(file_path, "rb") as f:
                    model = pickle.load(f)
            elif file_path.endswith(".joblib"):
                model = joblib.load(file_path)
            else:
                return "Unterstützte Formate: .pkl, .joblib"

            # Automatische Feature-Erkennung
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, "n_features_in_"):
                feature_names = [
                    f"Feature_{i + 1}" for i in range(model.n_features_in_)
                ]
            else:
                feature_names = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]

            # Automatische Target-Erkennung
            if hasattr(model, "classes_"):
                target_names = [str(cls) for cls in model.classes_]
                model_type = "classification"
            else:
                target_names = None
                model_type = "regression"

            # ModelInfo erstellen
            model_info = ModelInfo(
                name=filename,
                model_type=model_type,
                description=f"Automatisch geladen aus {file_path}",
                feature_names=feature_names,
                target_names=target_names,
            )

            # Modell registrieren
            model_manager.register_model(filename, model, model_info)

            return f"Modell '{filename}' automatisch geladen!\nTyp: {model_type}\nFeatures: {len(feature_names)}"

        except Exception as e:
            return f"Fehler beim Laden: {str(e)}"

    def delete_model(self, model_name: str) -> str:
        """Löscht ein Modell"""
        try:
            if not model_name:
                return "Kein Modell ausgewählt"

            # Aus Model Manager entfernen
            if model_name in model_manager.models:
                del model_manager.models[model_name]
                del model_manager.model_infos[model_name]

            # Datei löschen falls vorhanden
            if model_name in self.models_metadata:
                file_path = Path(self.models_metadata[model_name]["file_path"])
                if file_path.exists():
                    file_path.unlink()
                del self.models_metadata[model_name]
                self.save_metadata()

            return f"Modell '{model_name}' erfolgreich gelöscht"

        except Exception as e:
            return f"Fehler beim Löschen: {str(e)}"

    def get_model_details(self, model_name: str) -> str:
        """Gibt detaillierte Modell-Informationen zurück"""
        if not model_name:
            return "Kein Modell ausgewählt"

        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            return "Modell nicht gefunden"

        details = f"""
        ## Modell-Details: {model_info.name}
        
        **Typ:** {model_info.model_type}
        **Beschreibung:** {model_info.description or "Keine Beschreibung"}
        **Features:** {len(model_info.feature_names)}
        **Klassen/Targets:** {len(model_info.target_names) if model_info.target_names else "N/A"}
        
        ### Feature-Namen:
        """

        for i, feature in enumerate(model_info.feature_names, 1):
            details += f"\n{i}. {feature}"

        if model_info.target_names:
            details += "\n\n### Target-Klassen:"
            for i, target in enumerate(model_info.target_names, 1):
                details += f"\n{i}. {target}"

        # Zusätzliche Metadaten falls verfügbar
        if model_name in self.models_metadata:
            meta = self.models_metadata[model_name]
            details += "\n\n### Datei-Info:"
            details += f"\n**Pfad:** {meta.get('file_path', 'N/A')}"
            details += f"\n**Upload:** {meta.get('upload_timestamp', 'N/A')}"

        return details

    def predict_with_model(self, model_name: str, *feature_values) -> str:
        """Führt Vorhersage mit dem ausgewählten Modell durch"""
        try:
            if not model_name:
                return "Bitte Modell auswählen"

            model_info = model_manager.get_model_info(model_name)
            if not model_info:
                return "Modell nicht verfügbar"

            # Feature-Dictionary erstellen
            features = {}
            for i, feature_name in enumerate(model_info.feature_names):
                if i < len(feature_values):
                    features[feature_name] = float(feature_values[i])
                else:
                    features[feature_name] = 0.0

            # Vorhersage durchführen
            result = model_manager.predict(model_name, features)

            prediction_text = f"""
            ## Vorhersage-Ergebnis
            
            **Modell:** {result["model_name"]}
            **Vorhersage:** {result["prediction"]}
            """

            # Wahrscheinlichkeiten hinzufügen falls verfügbar
            if result.get("probabilities"):
                prediction_text += "\n### Wahrscheinlichkeiten:"
                for class_name, prob in result["probabilities"].items():
                    prediction_text += (
                        f"\n- **{class_name}:** {prob:.3f} ({prob * 100:.1f}%)"
                    )

            return prediction_text

        except Exception as e:
            return f"Fehler bei der Vorhersage: {str(e)}"

    def explain_with_model(
        self, model_name: str, explainer_type: str, n_features: int, *feature_values
    ) -> Tuple[str, Any]:
        """Erstellt XAI-Erklärung für das Modell"""
        try:
            if not model_name:
                return "Bitte Modell auswählen", None

            model_info = model_manager.get_model_info(model_name)
            model = model_manager.get_model(model_name)

            if not model_info or not model:
                return "Modell nicht verfügbar", None

            # Feature-Array erstellen
            feature_array = np.zeros(len(model_info.feature_names))
            for i, feature_name in enumerate(model_info.feature_names):
                if i < len(feature_values):
                    feature_array[i] = float(feature_values[i])

            # Hintergrunddaten generieren
            X_background = np.random.rand(50, len(model_info.feature_names))

            # Erklärung erstellen
            explanation_data = explainer.explain(
                method=explainer_type,
                instance=feature_array,
                model=model,
                feature_names=model_info.feature_names,
                X_background=X_background,
                num_features=n_features,
            )

            # Erklärungstext formatieren
            explanation_text = f"""
            ## {explainer_type.upper()} Erklärung
            
            **Modell:** {model_name}
            **Methode:** {explainer_type}
            
            ### Feature-Wichtigkeiten:
            """

            for fi in explanation_data["feature_importances"]:
                explanation_text += (
                    f"\n{fi.rank}. **{fi.feature_name}:** {fi.importance:.4f}"
                )

            # Plot erstellen
            fig = self.create_explanation_plot(explanation_data["feature_importances"])

            return explanation_text, fig

        except Exception as e:
            return f"Fehler bei der Erklärung: {str(e)}", None

    def create_explanation_plot(self, feature_importances: List) -> go.Figure:
        """Erstellt Plotly-Diagramm für Feature-Wichtigkeiten"""
        if not feature_importances:
            return go.Figure()

        sorted_features = sorted(
            feature_importances, key=lambda x: abs(x.importance), reverse=True
        )

        feature_names = [fi.feature_name for fi in sorted_features]
        importances = [fi.importance for fi in sorted_features]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation="h",
                    marker=dict(
                        color=[abs(imp) for imp in importances],
                        colorscale="Viridis",
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Feature-Wichtigkeiten",
            xaxis_title="Wichtigkeit",
            yaxis_title="Features",
            height=400,
            template="plotly_white",
        )

        return fig

    def create_advanced_interface(self) -> gr.Blocks:
        """Erstellt das erweiterte Multi-Tab Interface"""

        with gr.Blocks(
            title="Advanced XAI Dashboard", theme=gr.themes.Monochrome()
        ) as interface:
            gr.Markdown("# Advanced XAI Dashboard")
            gr.Markdown(
                "**Vollständiges Model-Management mit Pickle-Upload und XAI-Erklärungen**"
            )

            with gr.Tabs():
                # =================== TAB 1: MODEL UPLOAD ===================
                with gr.TabItem("Model Upload"):
                    gr.Markdown("## Pickle/Joblib Modell hochladen")

                    with gr.Row():
                        with gr.Column(scale=1):
                            upload_file = gr.File(
                                label="Pickle/Joblib Datei auswählen",
                                file_types=[".pkl", ".joblib"],
                            )
                            model_name_input = gr.Textbox(
                                label="Modellname",
                                placeholder="z.B. mein_customer_model",
                            )
                            model_type_input = gr.Dropdown(
                                choices=["classification", "regression"],
                                label="Modelltyp",
                                value="classification",
                            )
                            description_input = gr.Textbox(
                                label="Beschreibung",
                                placeholder="z.B. Kundensegmentierung Model",
                                lines=2,
                            )
                            feature_names_input = gr.Textbox(
                                label="Feature-Namen (kommagetrennt)",
                                placeholder="z.B. age,income,score,experience",
                                lines=2,
                            )
                            target_names_input = gr.Textbox(
                                label="Target-Namen (kommagetrennt, optional)",
                                placeholder="z.B. Premium,Standard,Basic",
                                lines=1,
                            )

                            upload_btn = gr.Button(
                                "Modell hochladen", variant="primary"
                            )

                        with gr.Column(scale=1):
                            upload_status = gr.Markdown("Bereit für Upload...")

                            gr.Markdown("### Upload-Hilfe:")
                            gr.Markdown("""
                            **Unterstützte Formate:**
                            - `.pkl` (Pickle)
                            - `.joblib` (Joblib)
                            
                            **Feature-Namen:**
                            - Kommagetrennt eingeben
                            - Leer lassen für automatische Erkennung
                            
                            **Target-Namen:**
                            - Nur bei Klassifikation
                            - Automatische Erkennung verfügbar
                            """)

                    # Upload Event
                    upload_btn.click(
                        fn=self.upload_model_file,
                        inputs=[
                            upload_file,
                            model_name_input,
                            model_type_input,
                            description_input,
                            feature_names_input,
                            target_names_input,
                        ],
                        outputs=[upload_status],
                    )

                # =================== TAB 2: MODEL MANAGER ===================
                with gr.TabItem("Model Manager"):
                    gr.Markdown("## Modelle verwalten und Details anzeigen")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Verfügbare Modelle")
                            model_list = gr.Dropdown(
                                choices=model_manager.list_models(),
                                label="Modell auswählen",
                                interactive=True,
                            )

                            with gr.Row():
                                refresh_btn = gr.Button("Aktualisieren")
                                delete_btn = gr.Button("Löschen", variant="stop")

                            delete_status = gr.Markdown("")

                            gr.Markdown("### Dateisystem-Loader")
                            filesystem_path = gr.Textbox(
                                label="Dateipfad zur Pickle/Joblib-Datei",
                                placeholder="C:/path/to/model.pkl",
                            )
                            load_filesystem_btn = gr.Button("Aus Dateisystem laden")
                            filesystem_status = gr.Markdown("")

                        with gr.Column(scale=2):
                            model_details = gr.Markdown(
                                "Wählen Sie ein Modell aus für Details..."
                            )

                    # Event Handlers
                    model_list.change(
                        fn=self.get_model_details,
                        inputs=[model_list],
                        outputs=[model_details],
                    )

                    refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=model_manager.list_models()),
                        outputs=[model_list],
                    )

                    delete_btn.click(
                        fn=self.delete_model,
                        inputs=[model_list],
                        outputs=[delete_status],
                    )

                    load_filesystem_btn.click(
                        fn=self.load_model_from_filesystem,
                        inputs=[filesystem_path],
                        outputs=[filesystem_status],
                    )

                # =================== TAB 3: PREDICTION ===================
                with gr.TabItem("Prediction"):
                    gr.Markdown("## Vorhersagen mit ausgewähltem Modell")

                    with gr.Row():
                        with gr.Column(scale=1):
                            pred_model_dropdown = gr.Dropdown(
                                choices=model_manager.list_models(),
                                label="Modell für Vorhersage",
                                value=model_manager.list_models()[0]
                                if model_manager.list_models()
                                else None,
                            )

                            gr.Markdown("### Feature-Eingaben")
                            # Dynamische Feature-Eingaben (vereinfacht mit 10 Features)
                            feature_inputs = []
                            for i in range(10):
                                feature_input = gr.Number(
                                    label=f"Feature {i + 1}",
                                    value=0.0,
                                    precision=3,
                                    visible=False,
                                )
                                feature_inputs.append(feature_input)

                            predict_btn = gr.Button(
                                "Vorhersage durchführen", variant="primary"
                            )

                        with gr.Column(scale=1):
                            prediction_result = gr.Markdown(
                                "Klicken Sie auf 'Vorhersage durchführen'"
                            )

                    # Feature-Inputs basierend auf Modell anzeigen
                    def update_feature_inputs(model_name):
                        if not model_name:
                            return [gr.Number(visible=False) for _ in range(10)]

                        model_info = model_manager.get_model_info(model_name)
                        if not model_info:
                            return [gr.Number(visible=False) for _ in range(10)]

                        updated_inputs = []
                        for i in range(10):
                            if i < len(model_info.feature_names):
                                updated_inputs.append(
                                    gr.Number(
                                        label=model_info.feature_names[i],
                                        value=0.0,
                                        precision=3,
                                        visible=True,
                                    )
                                )
                            else:
                                updated_inputs.append(gr.Number(visible=False))

                        return updated_inputs

                    pred_model_dropdown.change(
                        fn=update_feature_inputs,
                        inputs=[pred_model_dropdown],
                        outputs=feature_inputs,
                    )

                    predict_btn.click(
                        fn=self.predict_with_model,
                        inputs=[pred_model_dropdown] + feature_inputs,
                        outputs=[prediction_result],
                    )

                # =================== TAB 4: EXPLANATION ===================
                with gr.TabItem("Explanation"):
                    gr.Markdown("## XAI-Erklärungen generieren")

                    with gr.Row():
                        with gr.Column(scale=1):
                            exp_model_dropdown = gr.Dropdown(
                                choices=model_manager.list_models(),
                                label="Modell für Erklärung",
                                value=model_manager.list_models()[0]
                                if model_manager.list_models()
                                else None,
                            )

                            explainer_type = gr.Dropdown(
                                choices=["lime", "shap", "permutation"],
                                label="Erklärungsmethode",
                                value="lime",
                            )

                            n_features = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Anzahl wichtigster Features",
                            )

                            gr.Markdown("### Feature-Eingaben für Erklärung")
                            # Weitere Feature-Inputs für Erklärung
                            exp_feature_inputs = []
                            for i in range(10):
                                exp_feature_input = gr.Number(
                                    label=f"Feature {i + 1}",
                                    value=0.0,
                                    precision=3,
                                    visible=False,
                                )
                                exp_feature_inputs.append(exp_feature_input)

                            explain_btn = gr.Button(
                                "Erklärung generieren", variant="secondary"
                            )

                        with gr.Column(scale=2):
                            explanation_text = gr.Markdown(
                                "Klicken Sie auf 'Erklärung generieren'"
                            )
                            explanation_plot = gr.Plot(label="Feature-Wichtigkeiten")

                    # Feature-Updates für Erklärung
                    exp_model_dropdown.change(
                        fn=update_feature_inputs,
                        inputs=[exp_model_dropdown],
                        outputs=exp_feature_inputs,
                    )

                    explain_btn.click(
                        fn=self.explain_with_model,
                        inputs=[exp_model_dropdown, explainer_type, n_features]
                        + exp_feature_inputs,
                        outputs=[explanation_text, explanation_plot],
                    )

                # =================== TAB 5: MODEL INFO ===================
                with gr.TabItem("Model Info"):
                    gr.Markdown("## Detaillierte Modell-Informationen")

                    info_model_dropdown = gr.Dropdown(
                        choices=model_manager.list_models(),
                        label="Modell auswählen für Details",
                        value=model_manager.list_models()[0]
                        if model_manager.list_models()
                        else None,
                    )

                    model_info_display = gr.Markdown("Wählen Sie ein Modell aus...")

                    info_model_dropdown.change(
                        fn=self.get_model_details,
                        inputs=[info_model_dropdown],
                        outputs=[model_info_display],
                    )

            # Global refresh function für alle Dropdowns
            def refresh_all_dropdowns():
                choices = model_manager.list_models()
                value = choices[0] if choices else None
                return (
                    gr.Dropdown(choices=choices, value=value),
                    gr.Dropdown(choices=choices, value=value),
                    gr.Dropdown(choices=choices, value=value),
                    gr.Dropdown(choices=choices, value=value),
                )

            # Update alle Dropdowns wenn Upload erfolgt
            def upload_and_refresh(*args):
                # Führe Upload durch
                result = self.upload_model_file(*args)

                # Aktualisiere alle Dropdowns
                choices = model_manager.list_models()
                value = choices[0] if choices else None

                return (
                    result,  # upload_status
                    gr.Dropdown(choices=choices, value=value),  # model_list
                    gr.Dropdown(choices=choices, value=value),  # pred_model_dropdown
                    gr.Dropdown(choices=choices, value=value),  # exp_model_dropdown
                    gr.Dropdown(choices=choices, value=value),  # info_model_dropdown
                )

            upload_btn.click(
                fn=upload_and_refresh,
                inputs=[
                    upload_file,
                    model_name_input,
                    model_type_input,
                    description_input,
                    feature_names_input,
                    target_names_input,
                ],
                outputs=[
                    upload_status,
                    model_list,
                    pred_model_dropdown,
                    exp_model_dropdown,
                    info_model_dropdown,
                ],
            )

            # Auch refresh Button in Manager Tab aktualisiert alle
            def refresh_and_update():
                choices = model_manager.list_models()
                value = choices[0] if choices else None
                return (
                    gr.Dropdown(choices=choices, value=value),  # model_list
                    gr.Dropdown(choices=choices, value=value),  # pred_model_dropdown
                    gr.Dropdown(choices=choices, value=value),  # exp_model_dropdown
                    gr.Dropdown(choices=choices, value=value),  # info_model_dropdown
                )

            refresh_btn.click(
                fn=refresh_and_update,
                outputs=[
                    model_list,
                    pred_model_dropdown,
                    exp_model_dropdown,
                    info_model_dropdown,
                ],
            )

        return interface

    def launch(self, **kwargs):
        """Startet das erweiterte Dashboard"""
        interface = self.create_advanced_interface()
        interface.launch(**kwargs)


# Global verfügbare erweiterte Instanz
advanced_dashboard = AdvancedXAIDashboard()


def create_advanced_dashboard():
    """Factory-Funktion für das erweiterte Dashboard"""
    return advanced_dashboard.create_advanced_interface()
