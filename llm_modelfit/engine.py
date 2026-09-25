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


def memoire_cache_kv(nb_couches, nb_tetes_kv, dim_tete, contexte, precision_kv="FP16"):
    octets = OCTETS_PAR_PRECISION[precision_kv]
    return 2 * nb_couches * nb_tetes_kv * dim_tete * contexte * octets


def memoire_buffers(poids, cache, pct=0.1):
    # pas de formule exacte, approximé en % de (poids + cache)
    return (poids + cache) * pct


def octets_vers_go(octets):
    return octets / (1024 ** 3)


def memoire_totale(specs, precision, contexte, precision_kv="FP16", pct_buffers=0.1):
    poids = memoire_poids(specs["nb_param"], precision)
    cache = memoire_cache_kv(specs["num_layers"], specs["num_kv_heads"], specs["head_dim"], contexte, precision_kv)
    buffers = memoire_buffers(poids, cache, pct_buffers)

    return {
        "poids": poids,
        "cache_kv": cache,
        "buffers": buffers,
        "total": poids + cache + buffers,
    }


def nb_gpu_necessaire(total_go, mem_par_gpu_go, marge=0.15):
    n = math.ceil(total_go / mem_par_gpu_go)
    while total_go > n * mem_par_gpu_go * (1 - marge):
        n += 1
    return n


def autonomie_heures(capacite_wh, puissance_watts):
    if not puissance_watts:
        return None
    return capacite_wh / puissance_watts


def note_thermique(puissance_watts):
    if puissance_watts < 15:
        return "passif"
    if puissance_watts < 50:
        return "ventilation recommandée"
    return "refroidissement actif nécessaire"
