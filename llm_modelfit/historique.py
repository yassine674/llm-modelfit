# garde une trace de chaque recherche faite avec l'outil
import json
from pathlib import Path
from datetime import datetime

CHEMIN_HISTORIQUE = Path(__file__).parent / "data" / "historique.json"


def charger_historique():
    if not CHEMIN_HISTORIQUE.exists():
        return []
    with open(CHEMIN_HISTORIQUE) as f:
        return json.load(f)


def ajouter_historique(modele, precision, contexte, plateforme, diagnostic):
    historique = charger_historique()
    historique.append({
        "date": datetime.now().isoformat(timespec="seconds"),
        "modele": modele,
        "precision": precision,
        "contexte": contexte,
        "plateforme": plateforme,
        "detail_go": diagnostic["detail_go"],
        "compatible": diagnostic["compat"]["compatible"],
        "taux_go": diagnostic["compat"]["taux_go"],
        "suggestion_precision": diagnostic["suggestion_precision"],
        "suggestion_multi_gpu": diagnostic["suggestion_multi_gpu"],
        "plateformes_alternatives": diagnostic["plateformes_alternatives"],
    })
    with open(CHEMIN_HISTORIQUE, "w") as f:
        json.dump(historique, f, indent=2)
    return historique
