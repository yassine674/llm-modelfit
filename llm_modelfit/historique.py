# garde une trace de chaque recherche faite avec l'outil
from datetime import datetime
from .db import connexion


def charger_historique():
    conn = connexion()
    lignes = conn.execute("SELECT * FROM historique ORDER BY id").fetchall()

    historique = []
    for l in lignes:
        e = dict(l)
        e["compatible"] = bool(e["compatible"])
        e["detail_go"] = {
            "poids": e.pop("poids_go"),
            "cache_kv": e.pop("cache_kv_go"),
            "buffers": e.pop("buffers_go"),
            "total": e.pop("total_go"),
        }
        options = conn.execute(
            "SELECT plateforme, \"precision\", nb_gpu, total_go, power_watts, note_thermique "
            "FROM historique_options WHERE historique_id = ?",
            (e["id"],),
        ).fetchall()
        e["meilleures_options"] = [dict(o) for o in options]
        historique.append(e)

    conn.close()
    return historique


def ajouter_historique(modele, precision, contexte, plateforme, diagnostic):
    conn = connexion()
    detail = diagnostic["detail_go"]
    curseur = conn.execute(
        "INSERT INTO historique "
        '(date, modele, "precision", contexte, plateforme, poids_go, cache_kv_go, buffers_go, total_go, '
        "compatible, taux_go, power_watts, note_thermique, autonomie_heures) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            modele, precision, contexte, plateforme,
            detail["poids"], detail["cache_kv"], detail["buffers"], detail["total"],
            int(diagnostic["compat"]["compatible"]),
            diagnostic["compat"]["taux_go"],
            diagnostic["power_watts"],
            diagnostic["note_thermique"],
            diagnostic["autonomie_heures"],
        ),
    )
    historique_id = curseur.lastrowid

    for o in diagnostic["meilleures_options"]:
        conn.execute(
            "INSERT INTO historique_options "
            '(historique_id, plateforme, "precision", nb_gpu, total_go, power_watts, note_thermique) '
            "VALUES (?,?,?,?,?,?,?)",
            (
                historique_id, o["plateforme"], o["precision"], o["nb_gpu"],
                o.get("total_go"), o["power_watts"], o["note_thermique"],
            ),
        )

    conn.commit()
    conn.close()
    return charger_historique()


def vider_historique():
    conn = connexion()
    conn.execute("DELETE FROM historique_options")
    conn.execute("DELETE FROM historique")
    conn.commit()
    conn.close()
