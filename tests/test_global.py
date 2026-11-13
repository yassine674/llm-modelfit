import os
import tempfile

# base jetable : les tests ne doivent pas écrire dans l'historique réel
os.environ["LLM_MODELFIT_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

from llm_modelfit.modeles import get_specs
from llm_modelfit.engine import (
    memoire_totale, octets_vers_go, nb_gpu_necessaire,
    autonomie_heures, note_thermique,
)
from llm_modelfit.comparaison import (
    charger_plateformes, verifier_compat, comparer_toutes_plateformes,
    meilleure_option, toutes_les_options, diagnostiquer,
)
from llm_modelfit.historique import ajouter_historique, charger_historique
from llm_modelfit.fetch.huggingface import normalize_config

erreurs = []


def verifier(nom, condition):
    if condition:
        print(f"OK   - {nom}")
    else:
        print(f"FAIL - {nom}")
        erreurs.append(nom)


# --- récupération des modèles ---
specs_api = get_specs("mistralai/Mistral-7B-v0.1")
verifier("get_specs via API (Mistral)", specs_api["num_layers"] == 32 and specs_api["num_kv_heads"] == 8)

specs_local = get_specs("Llama-3-8B")
verifier("get_specs via secours local (Llama-3-8B)", specs_local["nb_param"] == 8_000_000_000)

# --- moteur de calcul ---
r = memoire_totale(specs_local, "INT4", 8192)
total_go = octets_vers_go(r["total"])
verifier("memoire_totale proche de l'exemple du sujet (5.2 Go)", abs(total_go - 5.2) < 0.1)

# --- fenêtre glissante (sliding window) ---
base = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "hidden_size": 4096, "vocab_size": 32000}
s_mistral = normalize_config({**base, "sliding_window": 4096})
verifier("Mistral v0.1 : 32 couches glissantes, fenêtre 4096",
         s_mistral["num_sliding_layers"] == 32 and s_mistral["sliding_window"] == 4096)
s_qwen = normalize_config({**base, "sliding_window": 131072, "use_sliding_window": False})
verifier("fenêtre désactivée (Qwen2) ignorée", s_qwen["num_sliding_layers"] == 0)
s_gemma3 = normalize_config({**base, "num_hidden_layers": 26, "sliding_window": 512, "sliding_window_pattern": 6})
verifier("Gemma 3 : une couche globale sur 6", s_gemma3["num_sliding_layers"] == 22)
s_types = normalize_config({**base, "num_hidden_layers": 4, "sliding_window": 128,
                            "layer_types": ["sliding_attention", "full_attention"] * 2})
verifier("layer_types explicite pris en compte", s_types["num_sliding_layers"] == 2)
kv_8k = memoire_totale({**s_mistral, "nb_param": 1}, "FP16", 8192)["cache_kv"]
kv_4k = memoire_totale({**s_mistral, "nb_param": 1}, "FP16", 4096)["cache_kv"]
verifier("cache KV plafonné à la fenêtre au-delà de 4096 tokens", kv_8k == kv_4k)
kv_sans = memoire_totale({**s_qwen, "nb_param": 1}, "FP16", 8192)["cache_kv"]
verifier("sans fenêtre, le cache KV suit tout le contexte", kv_sans == 2 * kv_4k)

verifier("nb_gpu_necessaire(200, 80) == 3", nb_gpu_necessaire(200, 80) == 3)
verifier("autonomie_heures(100, 15) proche de 6.67h", abs(autonomie_heures(100, 15) - 6.67) < 0.01)
verifier("autonomie_heures sans puissance -> None", autonomie_heures(100, 0) is None)
verifier("note_thermique(10) == passif", note_thermique(10) == "passif")
verifier("note_thermique(400) == refroidissement actif nécessaire", note_thermique(400) == "refroidissement actif nécessaire")

# --- comparaison / plateformes ---
plateformes = charger_plateformes()
verifier("17 plateformes dans la base", len(plateformes) == 17)

compat_plus_gros = verifier_compat(total_go, plateformes[0])
verifier("Llama INT4 8K compatible avec la plateforme la plus spacieuse", compat_plus_gros["compatible"] is True)

resultats_tries = comparer_toutes_plateformes(total_go)
verifier("comparer_toutes_plateformes trie les compatibles en premier", resultats_tries[0]["compatible"] is True)

rpi5 = next(p for p in plateformes if p["name"] == "Raspberry Pi 5 8GB")
r_fp16 = octets_vers_go(memoire_totale(specs_api, "FP16", 8192)["total"])
verifier("Mistral FP16 8K ne rentre PAS sur Raspberry Pi 5", verifier_compat(r_fp16, rpi5)["compatible"] is False)
verifier("marge : 90 % de la mémoire occupée n'est PAS compatible", verifier_compat(0.90 * rpi5["memory_gb"], rpi5)["compatible"] is False)
verifier("marge : 80 % de la mémoire occupée est compatible", verifier_compat(0.80 * rpi5["memory_gb"], rpi5)["compatible"] is True)

option_rpi5 = meilleure_option(specs_api, 8192, rpi5)
verifier("meilleure_option propose INT4 pour Mistral sur Raspberry Pi 5", option_rpi5["precision"] == "INT4" and option_rpi5["nb_gpu"] == 1)

plateforme_perso = {"name": "Carte perso", "memory_gb": 6, "power_watts": 50}
option_perso = meilleure_option(specs_local, 2048, plateforme_perso)
verifier("meilleure_option fonctionne sans la clé unified_memory", option_perso["nb_gpu"] is not None)

# --- diagnostic unifié ---
diag_ok = diagnostiquer(specs_api, "INT4", 8192, plateformes[0])
verifier("diagnostiquer : cas compatible, pas d'options proposées", diag_ok["compat"]["compatible"] and diag_ok["meilleures_options"] == [])
verifier("diagnostiquer : renvoie puissance et note thermique", diag_ok["power_watts"] > 0 and diag_ok["note_thermique"] is not None)

diag_batterie = diagnostiquer(specs_api, "INT4", 8192, rpi5, capacite_batterie_wh=100)
verifier("diagnostiquer : calcule l'autonomie si capacité batterie fournie", diag_batterie["autonomie_heures"] is not None)

diag_precision = diagnostiquer(specs_api, "FP16", 8192, rpi5)
options = diag_precision["meilleures_options"]
verifier("diagnostiquer : propose des options quand FP16 ne rentre pas", len(options) > 0)
verifier("diagnostiquer : la meilleure option sur Raspberry Pi 5 est INT4 à 1 GPU", any(o["plateforme"] == "Raspberry Pi 5 8GB" and o["precision"] == "INT4" and o["nb_gpu"] == 1 for o in options))

specs_geant = {"nb_param": 500_000_000_000, "num_layers": 96, "num_kv_heads": 96, "head_dim": 128}
diag_multi = diagnostiquer(specs_geant, "INT4", 2048, plateformes[0])
verifier("diagnostiquer : propose du multi-GPU quand rien ne rentre sur 1 seul GPU discret", any(o["nb_gpu"] and o["nb_gpu"] > 1 for o in diag_multi["meilleures_options"]))

toutes = toutes_les_options(specs_api, 8192)
verifier("toutes_les_options couvre toutes les plateformes compatibles", len(toutes) > 0)
verifier("toutes_les_options est triée (moins de GPU en premier)", toutes[0]["nb_gpu"] <= toutes[-1]["nb_gpu"])

# --- historique ---
avant = len(charger_historique())
ajouter_historique("Mistral-7B-v0.1", "INT4", 8192, "NVIDIA A100 80GB", diag_ok)
apres = len(charger_historique())
verifier("ajouter_historique ajoute bien une ligne", apres == avant + 1)

# --- base locale : migrations, secours hors ligne, configs incomplètes ---
import json
import sqlite3
from llm_modelfit import db
from llm_modelfit.modeles import _depuis_local
from llm_modelfit.engine import memoire_cache_kv

# ancienne base (v0.1.0) : colonne bandwidth_gbps, pas de config_json, numéro de version 0
ancienne = os.path.join(tempfile.mkdtemp(), "ancienne.db")
c = sqlite3.connect(ancienne)
c.executescript(db.SCHEMA.replace("bandwidth_gb_s REAL", "bandwidth_gbps REAL").replace(
    ",\n    config_json TEXT  -- config brute, relue par normalize_config comme une config Hugging Face", ""))
c.execute("INSERT INTO platforms (name, memory_gb, bandwidth_gbps, power_watts) VALUES ('Test', 8, 17, 5)")
c.execute("INSERT INTO models (hf_id, name, params, num_hidden_layers, vocab_size) VALUES ('google/gemma-7b', 'Gemma-7B', 1, 28, 256000)")
c.commit(); c.close()
os.environ["LLM_MODELFIT_DB"], base_tests = ancienne, os.environ["LLM_MODELFIT_DB"]
c = db.connexion()
cols_p = {l[1] for l in c.execute("PRAGMA table_info(platforms)")}
verifier("migration : bandwidth_gbps renommée en bandwidth_gb_s, valeur gardée",
         "bandwidth_gb_s" in cols_p and "bandwidth_gbps" not in cols_p
         and c.execute("SELECT bandwidth_gb_s FROM platforms WHERE name='Test'").fetchone()[0] == 17)
verifier("migration : modèles de référence rafraîchis (config_json remplie)",
         c.execute("SELECT config_json FROM models WHERE name='Gemma-7B'").fetchone()[0] is not None)
verifier("migration : numéro de version à jour", c.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS))
c.close()
db.connexion().close()  # deuxième ouverture : les migrations ne se rejouent pas sans erreur
os.environ["LLM_MODELFIT_DB"] = base_tests

mistral_local = _depuis_local("Mistral-7B-v0.1")
verifier("secours local : Mistral v0.1 garde sa fenêtre glissante",
         mistral_local["num_sliding_layers"] == 32 and mistral_local["sliding_window"] == 4096)
verifier("secours local : Gemma-7B a head_dim 256 (et pas 3072/16 = 192)", _depuis_local("Gemma-7B")["head_dim"] == 256)
verifier("secours local : une config vue en ligne est réutilisable hors ligne",
         _depuis_local("mistralai/Mistral-7B-v0.1") is not None)

gemma3_4b = {"model_type": "gemma3", "text_config": {"hidden_size": 2560, "model_type": "gemma3_text",
             "num_hidden_layers": 34, "sliding_window": 1024}}
s4 = normalize_config(gemma3_4b)
verifier("Gemma 3 4B : valeurs par défaut de transformers complétées",
         s4["num_kv_heads"] == 4 and s4["head_dim"] == 256 and s4["vocab_size"] == 262208)
verifier("Gemma 3 4B : la fenêtre de la config prime sur le défaut, motif 1 globale / 6",
         s4["sliding_window"] == 1024 and s4["num_sliding_layers"] == 34 - 34 // 6)
gemma3_27b = {"text_config": {"head_dim": 128, "hidden_size": 5376, "model_type": "gemma3_text",
              "num_attention_heads": 32, "num_hidden_layers": 62, "num_key_value_heads": 16}}
verifier("Gemma 3 27B : les valeurs présentes dans la config priment", normalize_config(gemma3_27b)["head_dim"] == 128)
verifier("head_dim null dans la config : recalculé",
         normalize_config({"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256,
                           "vocab_size": 10, "head_dim": None})["head_dim"] == 64)
verifier("cache KV : précision donnée en octets (q8_0 d'Ollama = 1.0625)",
         memoire_cache_kv(1, 1, 1, 16, 1.0625) == 2 * 16 * 1.0625)

# --- résumé ---
print()
if erreurs:
    print(f"{len(erreurs)} problème(s) trouvé(s) :", erreurs)
else:
    print("Tout est correct, aucun problème détecté.")
