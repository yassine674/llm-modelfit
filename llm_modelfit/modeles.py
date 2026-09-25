import json
from pathlib import Path
from .fetch.huggingface import get_model_config, normalize_config, get_nb_param

CHEMIN_MODELS = Path(__file__).parent / "data" / "models.json"


def _depuis_local(model_id):
    with open(CHEMIN_MODELS) as f:
        modeles = json.load(f)
    for m in modeles:
        if model_id in (m.get("hf_id"), m.get("name")):
            specs = normalize_config(m)
            specs["nb_param"] = m["params"]
            return specs
    return None


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
