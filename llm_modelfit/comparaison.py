# compare un modèle à des plateformes et propose des alternatives si ça ne rentre pas
import json
from pathlib import Path
from .engine import memoire_totale, octets_vers_go, nb_gpu_necessaire

CHEMIN_PLATEFORMES = Path(__file__).parent / "data" / "platforms.json"
ORDRE_PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]


def charger_plateformes():
    with open(CHEMIN_PLATEFORMES) as f:
        return json.load(f)


def verifier_compat(total_go, plateforme):
    dispo = plateforme["memory_gb"]
    return {
        "plateforme": plateforme["name"],
        "dispo_go": dispo,
        "requis_go": round(total_go, 3),
        "compatible": total_go <= dispo,
        "taux_go": round(total_go / dispo * 100, 1),
    }


def comparer_toutes_plateformes(total_go):
    resultats = [verifier_compat(total_go, p) for p in charger_plateformes()]
    return sorted(resultats, key=lambda r: not r["compatible"])


def suggerer_precision(specs, contexte, plateforme, precision_actuelle):
    depart = ORDRE_PRECISIONS.index(precision_actuelle)
    for precision in ORDRE_PRECISIONS[depart + 1:]:
        total_go = octets_vers_go(memoire_totale(specs, precision, contexte)["total"])
        if verifier_compat(total_go, plateforme)["compatible"]:
            return precision
    return None


def suggerer_multi_gpu(specs, contexte, plateforme):
    if plateforme.get("unified_memory", False):
        return None
    total_go = octets_vers_go(memoire_totale(specs, "INT4", contexte)["total"])
    return nb_gpu_necessaire(total_go, plateforme["memory_gb"])


def diagnostiquer(specs, precision, contexte, plateforme):
    # Un seul appel qui fait tout le raisonnement :
    # 1. ça rentre -> rien de plus
    # 2. ça ne rentre pas -> essaie une précision plus légère
    # 3. toujours rien -> GPU discret : combien il en faudrait / carte embarquée : quelles autres plateformes conviendraient

    r = memoire_totale(specs, precision, contexte)
    detail_go = {k: octets_vers_go(v) for k, v in r.items()}
    compat = verifier_compat(detail_go["total"], plateforme)

    resultat = {
        "detail_go": detail_go,
        "compat": compat,
        "suggestion_precision": None,
        "suggestion_multi_gpu": None,
        "plateformes_alternatives": [],
    }

    # cas 1 : ça rentre déjà, rien à suggérer
    if compat["compatible"]:
        return resultat

    # cas 2 : une précision plus légère du même modèle suffit peut-être
    resultat["suggestion_precision"] = suggerer_precision(specs, contexte, plateforme, precision)
    if resultat["suggestion_precision"] is not None:
        return resultat

    # cas 3 : même la précision la plus légère ne suffit pas
    if not plateforme.get("unified_memory", False):
        # GPU discret -> on peut en mettre plusieurs
        resultat["suggestion_multi_gpu"] = suggerer_multi_gpu(specs, contexte, plateforme)
    else:
        # carte embarquée -> pas de multi-GPU possible, on propose d'autres plateformes
        total_min_go = octets_vers_go(memoire_totale(specs, "INT4", contexte)["total"])
        resultat["plateformes_alternatives"] = [
            a["plateforme"] for a in comparer_toutes_plateformes(total_min_go)
            if a["compatible"] and a["plateforme"] != plateforme["name"]
        ]

    return resultat
