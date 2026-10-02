# compare un modèle à des plateformes et propose des alternatives si ça ne rentre pas
from .db import connexion
from .engine import memoire_totale, octets_vers_go, nb_gpu_necessaire, autonomie_heures, note_thermique

ORDRE_PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]


def charger_plateformes():
    conn = connexion()
    lignes = conn.execute("SELECT * FROM platforms").fetchall()
    conn.close()
    plateformes = [dict(l) for l in lignes]
    for p in plateformes:
        p["has_tensor_cores"] = bool(p["has_tensor_cores"])
        p["unified_memory"] = bool(p["unified_memory"])
    return plateformes


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


def meilleure_option(specs, contexte, plateforme):
    """
    Pour une plateforme donnée : la meilleure précision qui tient sur une seule
    unité. Si aucune ne tient, et que c'est un GPU discret (pas de mémoire
    unifiée), le nombre de GPU nécessaires à la précision la plus légère.
    Renvoie precision=None si rien ne convient jamais (cartes embarquées).
    """
    for precision in ORDRE_PRECISIONS:
        total_go = octets_vers_go(memoire_totale(specs, precision, contexte)["total"])
        if verifier_compat(total_go, plateforme)["compatible"]:
            return {
                "plateforme": plateforme["name"],
                "precision": precision,
                "nb_gpu": 1,
                "total_go": total_go,
                "power_watts": plateforme["power_watts"],
                "note_thermique": note_thermique(plateforme["power_watts"]),
            }

    if plateforme.get("unified_memory", False):
        return {
            "plateforme": plateforme["name"],
            "precision": None,
            "nb_gpu": None,
            "total_go": None,
            "power_watts": plateforme["power_watts"],
            "note_thermique": note_thermique(plateforme["power_watts"]),
        }

    total_go_int4 = octets_vers_go(memoire_totale(specs, "INT4", contexte)["total"])
    n = nb_gpu_necessaire(total_go_int4, plateforme["memory_gb"])
    return {
        "plateforme": plateforme["name"],
        "precision": "INT4",
        "nb_gpu": n,
        "total_go": total_go_int4,
        "power_watts": plateforme["power_watts"] * n,
        "note_thermique": note_thermique(plateforme["power_watts"]),
    }


def toutes_les_options(specs, contexte):
    """Meilleure option pour chaque plateforme, triée (le moins de GPU, la
    meilleure précision, la puissance la plus faible en premier)."""
    options = [meilleure_option(specs, contexte, p) for p in charger_plateformes()]
    options = [o for o in options if o["precision"] is not None]
    options.sort(key=lambda o: (o["nb_gpu"], ORDRE_PRECISIONS.index(o["precision"]), o["power_watts"]))
    return options


def diagnostiquer(specs, precision, contexte, plateforme, capacite_batterie_wh=None):
    r = memoire_totale(specs, precision, contexte)
    detail_go = {k: octets_vers_go(v) for k, v in r.items()}
    compat = verifier_compat(detail_go["total"], plateforme)
    puissance = plateforme["power_watts"]

    resultat = {
        "detail_go": detail_go,
        "compat": compat,
        "power_watts": puissance,
        "note_thermique": note_thermique(puissance),
        "autonomie_heures": autonomie_heures(capacite_batterie_wh, puissance) if capacite_batterie_wh else None,
        "meilleures_options": [],
    }

    if not compat["compatible"]:
        resultat["meilleures_options"] = toutes_les_options(specs, contexte)

    return resultat
