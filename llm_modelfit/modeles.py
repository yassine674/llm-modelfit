# get_specs() : essaie l'API Hugging Face, retombe sur la base locale si ça échoue
import json
from datetime import datetime

from .db import connexion
from .fetch.huggingface import get_model_config, normalize_config, get_nb_param


def _depuis_local(model_id):
    # d'abord les modèles de référence, puis les configs déjà vues en ligne
    conn = connexion()
    ligne = conn.execute(
        "SELECT config_json, params AS nb_param FROM models WHERE hf_id = ? OR name = ?", (model_id, model_id)
    ).fetchone()
    if ligne is None:
        ligne = conn.execute(
            "SELECT config_json, nb_param FROM cache_configs WHERE hf_id = ?", (model_id,)
        ).fetchone()
    conn.close()
    if ligne is None:
        return None
    specs = normalize_config(json.loads(ligne["config_json"]))
    specs["nb_param"] = ligne["nb_param"]
    return specs


def _memoriser(model_id, config, nb_param):
    conn = connexion()
    conn.execute(
        "INSERT OR REPLACE INTO cache_configs (hf_id, config_json, nb_param, date) VALUES (?,?,?,?)",
        (model_id, json.dumps(config), nb_param, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def lister_modeles():
    conn = connexion()
    lignes = conn.execute("SELECT name, hf_id FROM models").fetchall()
    conn.close()
    return [dict(l) for l in lignes]


def get_specs(model_id):
    try:
        config = get_model_config(model_id)
        specs = normalize_config(config)
        nb_param = get_nb_param(model_id)
        if nb_param is None:
            raise RuntimeError("nb de paramètres indisponible via l'API")
        specs["nb_param"] = nb_param
        _memoriser(model_id, config, nb_param)
        return specs
    except Exception as e:
        # RuntimeError (config inexploitable) mais aussi erreurs réseau : hors ligne,
        # on veut la base locale, pas une exception venue de requests ou huggingface_hub
        secours = _depuis_local(model_id)
        if secours is not None:
            secours["avertissement"] = str(e)
            return secours

        # pas de config.json exploitable (modèle de diffusion, multi-composants...) :
        # si on peut au moins compter les paramètres, on estime le poids seul
        try:
            nb_param = get_nb_param(model_id)
        except Exception:
            nb_param = None
        if nb_param is not None:
            return {"poids_seulement": True, "nb_param": nb_param, "avertissement": str(e)}

        raise
