# get_specs() : essaie l'API Hugging Face, retombe sur la base locale si ça échoue
from .db import connexion
from .fetch.huggingface import get_model_config, normalize_config, get_nb_param


def _depuis_local(model_id):
    conn = connexion()
    ligne = conn.execute(
        "SELECT * FROM models WHERE hf_id = ? OR name = ?", (model_id, model_id)
    ).fetchone()
    conn.close()
    if ligne is None:
        return None
    m = dict(ligne)
    specs = normalize_config(m)
    specs["nb_param"] = m["params"]
    return specs


def lister_modeles():
    conn = connexion()
    lignes = conn.execute("SELECT name, hf_id FROM models").fetchall()
    conn.close()
    return [dict(l) for l in lignes]


def get_specs(model_id):
    try:
        specs = normalize_config(get_model_config(model_id))
        nb_param = get_nb_param(model_id)
        if nb_param is None:
            raise RuntimeError("nb de paramètres indisponible via l'API")
        specs["nb_param"] = nb_param
        return specs
    except RuntimeError:
        secours = _depuis_local(model_id)
        if secours is not None:
            return secours
        raise
