#!/usr/bin/env python3
"""
Point d'entrée simplifié pour le dashboard multi-pages des météorites.
Ce script permet de lancer l'application depuis la racine du projet.
"""
import os
import sys

# Ajout du répertoire src au chemin Python
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

if __name__ == "__main__":
    # Importer et exécuter l'application
    # Dash chargera automatiquement toutes les pages du dossier src/pages
    from app import app

    # Configuration du port et du mode debug
    import utils.config as config

    print(f"Lancement du tableau de bord multi-pages d'analyse de météorites sur http://127.0.0.1:{config.PORT}/")
    # Note: En mode debug, Dash recharge l'application au démarrage, ce qui peut causer des messages dupliqués
    app.run_server(debug=config.DEBUG_MODE, port=config.PORT)