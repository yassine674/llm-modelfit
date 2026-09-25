# récupère les caractéristiques d'un modèle Hugging Face (config + nb de paramètres)
import json
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def get_model_config(model_id):
    try:
        chemin = hf_hub_download(repo_id=model_id, filename="config.json", token=HF_TOKEN)
    except HfHubHTTPError as e:
        code = e.response.status_code
        if code == 403:
            raise RuntimeError(f"Licence non acceptée pour '{model_id}' : https://huggingface.co/{model_id}") from e
        if code == 404:
            raise RuntimeError(f"Modèle '{model_id}' introuvable.") from e
        raise

    with open(chemin) as f:
        return json.load(f)


def get_nb_param(model_id):
    info = HfApi(token=HF_TOKEN).model_info(model_id)
    if info.safetensors:
        return info.safetensors.total
    return None


def normalize_config(cfg):
    nb_couches = cfg["num_hidden_layers"]
    nb_tetes = cfg["num_attention_heads"]
    taille_cachee = cfg["hidden_size"]
    # pas de GQA -> autant de têtes KV que de têtes normales
    nb_tetes_kv = cfg.get("num_key_value_heads", nb_tetes)
    # Gemma donne head_dim direct, les autres non (on le calcule)
    dim_tete = cfg.get("head_dim", taille_cachee // nb_tetes)

    return {
        "num_layers": nb_couches,
        "num_kv_heads": nb_tetes_kv,
        "head_dim": dim_tete,
        "vocab_size": cfg["vocab_size"],
        "context_max": cfg.get("max_position_embeddings"),
    }
