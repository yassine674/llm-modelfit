# calcule la mémoire nécessaire pour faire tourner un modèle : poids + cache KV + buffers
import math

OCTETS_PAR_PRECISION = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
    "INT4": 0.5,
}


def memoire_poids(nb_param, precision):
    return nb_param * OCTETS_PAR_PRECISION[precision]


def memoire_cache_kv(nb_couches, nb_tetes_kv, dim_tete, contexte, precision_kv="FP16",
                     nb_couches_glissantes=0, fenetre=None):
    # attention à fenêtre glissante (Mistral 7B v0.1, Gemma 2/3) : ces couches ne
    # gardent que les `fenetre` derniers tokens, quelle que soit la longueur du contexte
    # precision_kv : un nom ("FP16"...) ou directement un nombre d'octets par valeur
    # (ex. 1.0625 pour le format q8_0 d'Ollama, qui n'a pas de nom ici)
    octets = precision_kv if isinstance(precision_kv, (int, float)) else OCTETS_PAR_PRECISION[precision_kv]
    tokens_glissants = min(contexte, fenetre) if fenetre else contexte
    tokens_stockes = (nb_couches - nb_couches_glissantes) * contexte + nb_couches_glissantes * tokens_glissants
    return 2 * nb_tetes_kv * dim_tete * tokens_stockes * octets


def memoire_buffers(poids, cache, pct=0.1):
    # pas de formule exacte, approximé en % de (poids + cache)
    return (poids + cache) * pct


def octets_vers_go(octets):
    return octets / (1024 ** 3)


def memoire_totale(specs, precision, contexte, precision_kv="FP16", pct_buffers=0.1):
    poids = memoire_poids(specs["nb_param"], precision)
    # modèle sans décodeur texte (diffusion, etc.) : pas de génération token par
    # token, donc pas de cache KV ni de notion de longueur de contexte
    if specs.get("poids_seulement"):
        cache = 0
    else:
        cache = memoire_cache_kv(
            specs["num_layers"], specs["num_kv_heads"], specs["head_dim"], contexte, precision_kv,
            specs.get("num_sliding_layers", 0), specs.get("sliding_window"),
        )
    buffers = memoire_buffers(poids, cache, pct_buffers)

    return {
        "poids": poids,
        "cache_kv": cache,
        "buffers": buffers,
        "total": poids + cache + buffers,
    }


# part de la mémoire laissée libre : OS, contexte CUDA, fragmentation de l'allocateur
MARGE_MEMOIRE = 0.15


def nb_gpu_necessaire(total_go, mem_par_gpu_go, marge=MARGE_MEMOIRE):
    n = math.ceil(total_go / mem_par_gpu_go)
    while total_go > n * mem_par_gpu_go * (1 - marge):
        n += 1
    return n


def autonomie_heures(capacite_wh, puissance_watts):
    if not puissance_watts:
        return None
    return capacite_wh / puissance_watts


TENSION_BATTERIE_V = 5.0  # alimentation typique des cartes embarquées (USB-C / barrel 5V)


def mah_vers_wh(capacite_mah, tension_v=TENSION_BATTERIE_V):
    return capacite_mah * tension_v / 1000


def note_thermique(puissance_watts):
    if puissance_watts < 15:
        return "passif"
    if puissance_watts < 50:
        return "ventilation recommandée"
    return "refroidissement actif nécessaire"
