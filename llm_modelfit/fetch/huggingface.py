# récupère les caractéristiques d'un modèle Hugging Face (config + nb de paramètres)
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# connexion réutilisée (keep-alive) : évite de refaire une poignée de main
# TLS à chaque appel, ce qui coûtait le plus de temps sur la recherche live
_session = requests.Session()

_cache_avatars = {}


def avatar_auteur(nom):
    if nom in _cache_avatars:
        return _cache_avatars[nom]

    avatar = None
    for type_compte in ("organizations", "users"):
        try:
            r = _session.get(f"https://huggingface.co/api/{type_compte}/{nom}/avatar", timeout=2)
            if r.ok:
                avatar = r.json().get("avatarUrl")
                break
        except requests.RequestException:
            pass

    _cache_avatars[nom] = avatar
    return avatar


def get_model_config(model_id):
    try:
        chemin = hf_hub_download(repo_id=model_id, filename="config.json", token=HF_TOKEN)
    except HfHubHTTPError as e:
        code = e.response.status_code
        if code == 403:
            raise RuntimeError(f"Licence non acceptée pour '{model_id}' : https://huggingface.co/{model_id}") from e
        if code == 404:
            try:
                HfApi(token=HF_TOKEN).model_info(model_id)
            except Exception:
                raise RuntimeError(f"Modèle '{model_id}' introuvable.") from e
            raise RuntimeError(
                f"'{model_id}' n'a pas de config.json à la racine "
                "(probablement un modèle multi-composants : diffusion, image, audio...)."
            ) from e
        raise

    with open(chemin) as f:
        return json.load(f)


def get_nb_param(model_id):
    info = HfApi(token=HF_TOKEN).model_info(model_id)
    if info.safetensors:
        return info.safetensors.total
    if info.gguf:
        return info.gguf["total"]
    return None


def rechercher_modeles(recherche, limite=8, tache=None):
    if not recherche or len(recherche) < 2:
        return []
    resultats = list(HfApi(token=HF_TOKEN).list_models(
        search=recherche, limit=limite, sort="downloads", pipeline_tag=tache
    ))
    # les avatars (un appel HTTP par auteur distinct) sont le vrai goulot
    # d'étranglement : on les récupère en parallèle plutôt qu'un par un
    auteurs = [m.id.split("/")[0] for m in resultats]
    with ThreadPoolExecutor(max_workers=max(len(auteurs), 1)) as executor:
        avatars = list(executor.map(avatar_auteur, auteurs))
    return [
        {"id": m.id, "downloads": m.downloads or 0, "avatar": avatar}
        for m, avatar in zip(resultats, avatars)
    ]


def rechercher_organisations(recherche, limite=5):
    # utilise la même recherche instantanée que la barre de recherche du site
    # Hugging Face (organisations réelles, pas une liste figée à quelques noms)
    if not recherche or len(recherche) < 1:
        return []
    try:
        r = _session.get("https://huggingface.co/api/quicksearch", params={"q": recherche}, timeout=2)
        if not r.ok:
            return []
    except requests.RequestException:
        return []

    return [
        {"nom": o["name"], "nom_complet": o.get("fullname"), "avatar": o.get("avatarUrl")}
        for o in r.json().get("orgs", [])[:limite]
    ]


#  anciens noms de champs utilisés par GPT-2 et ses dérivés, avant que la
# convention num_hidden_layers / num_attention_heads / hidden_size se généralise
ALIAS_ANCIENS = {"num_hidden_layers": "n_layer", "num_attention_heads": "n_head", "hidden_size": "n_embd"}


def _couches_glissantes(cfg, nb_couches):
    # attention à fenêtre glissante : (nombre de couches concernées, taille de la fenêtre)
    fenetre = cfg.get("sliding_window")
    if not fenetre or cfg.get("use_sliding_window") is False:
        return 0, None  # ex. Qwen2 : fenêtre renseignée mais désactivée
    types = cfg.get("layer_types")
    if types:
        return sum(t == "sliding_attention" for t in types), fenetre
    motif = cfg.get("sliding_window_pattern") or cfg.get("_sliding_window_pattern")
    if motif:
        # Gemma 3 : une couche globale toutes les `motif` couches, les autres glissantes
        return nb_couches - nb_couches // motif, fenetre
    if cfg.get("model_type") == "gemma2":
        # Gemma 2 : alternance, les couches d'indice pair sont glissantes
        return (nb_couches + 1) // 2, fenetre
    return nb_couches, fenetre  # Mistral 7B v0.1 : toutes les couches


def normalize_config(cfg):
    requis = ["num_hidden_layers", "num_attention_heads", "hidden_size", "vocab_size"]
    manquants = [c for c in requis if c not in cfg]

    # modèles multimodaux (texte + image/audio) : la config du décodeur texte
    # est alors rangée dans une sous-section "text_config"
    if manquants and "text_config" in cfg:
        cfg = cfg["text_config"]
        manquants = [c for c in requis if c not in cfg]

    if manquants:
        cfg = dict(cfg)
        for cible, source in ALIAS_ANCIENS.items():
            if cible not in cfg and source in cfg:
                cfg[cible] = cfg[source]
        manquants = [c for c in requis if c not in cfg]

    if manquants:
        raise RuntimeError(
            "Ce modèle n'a pas la structure attendue d'un LLM texte "
            f"(champs manquants dans sa config : {', '.join(manquants)}). "
            "C'est probablement un modèle d'un autre type (image, audio...)."
        )

    nb_couches = cfg["num_hidden_layers"]
    nb_tetes = cfg["num_attention_heads"]
    taille_cachee = cfg["hidden_size"]
    # pas de GQA -> autant de têtes KV que de têtes normales
    nb_tetes_kv = cfg.get("num_key_value_heads", nb_tetes)
    # Gemma donne head_dim direct, les autres non (on le calcule)
    dim_tete = cfg.get("head_dim", taille_cachee // nb_tetes)
    nb_glissantes, fenetre = _couches_glissantes(cfg, nb_couches)

    return {
        "num_layers": nb_couches,
        "num_kv_heads": nb_tetes_kv,
        "head_dim": dim_tete,
        "num_sliding_layers": nb_glissantes,
        "sliding_window": fenetre,
        "vocab_size": cfg["vocab_size"],
        "context_max": cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("n_ctx"),
    }
