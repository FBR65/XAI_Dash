#!/usr/bin/env python3
"""
XAI Dashboard Starter
Startet das Advanced XAI Dashboard mit allen Features
"""

import sys
import os

# Füge den Projektpfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from app.ui.advanced_dashboard import AdvancedXAIDashboard


def main():
    """Hauptfunktion zum Starten des XAI Dashboards"""
    print("Starte Advanced XAI Dashboard...")
    print("Features: Model Upload, XAI Explanations, Predictions")
    print("Dashboard öffnet sich automatisch im Browser")
    print("-" * 50)

    # Dashboard erstellen und starten
    dashboard = AdvancedXAIDashboard()
    interface = dashboard.create_advanced_interface()

    # Server starten
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
