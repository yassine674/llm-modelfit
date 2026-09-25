from llm_modelfit.modeles import get_specs
from llm_modelfit.engine import (
    memoire_totale, octets_vers_go, nb_gpu_necessaire,
    autonomie_heures, note_thermique,
)
from llm_modelfit.comparaison import (
    charger_plateformes, verifier_compat, comparer_toutes_plateformes,
    suggerer_precision, suggerer_multi_gpu, diagnostiquer,
)
from llm_modelfit.historique import ajouter_historique, charger_historique

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

verifier("nb_gpu_necessaire(200, 80) == 3", nb_gpu_necessaire(200, 80) == 3)
verifier("autonomie_heures(100, 15) proche de 6.67h", abs(autonomie_heures(100, 15) - 6.67) < 0.01)
verifier("autonomie_heures sans puissance -> None", autonomie_heures(100, 0) is None)
verifier("note_thermique(10) == passif", note_thermique(10) == "passif")
verifier("note_thermique(400) == refroidissement actif nécessaire", note_thermique(400) == "refroidissement actif nécessaire")

# --- comparaison / plateformes ---
plateformes = charger_plateformes()
verifier("17 plateformes dans la base", len(plateformes) == 17)

compat_a100 = verifier_compat(total_go, plateformes[0])
verifier("Llama INT4 8K compatible avec A100", compat_a100["compatible"] is True)

resultats_tries = comparer_toutes_plateformes(total_go)
verifier("comparer_toutes_plateformes trie les compatibles en premier", resultats_tries[0]["compatible"] is True)

rpi5 = next(p for p in plateformes if p["name"] == "Raspberry Pi 5 8GB")
r_fp16 = octets_vers_go(memoire_totale(specs_api, "FP16", 8192)["total"])
verifier("Mistral FP16 8K ne rentre PAS sur Raspberry Pi 5", verifier_compat(r_fp16, rpi5)["compatible"] is False)

suggestion = suggerer_precision(specs_api, 8192, rpi5, "FP16")
verifier("suggerer_precision propose INT4 pour Mistral sur Raspberry Pi 5", suggestion == "INT4")

plateforme_perso = {"name": "Carte perso", "memory_gb": 6}
nb_gpu = suggerer_multi_gpu(specs_local, 2048, plateforme_perso)
verifier("suggerer_multi_gpu fonctionne sans la clé unified_memory", nb_gpu is not None)

# --- diagnostic unifié ---
diag_ok = diagnostiquer(specs_api, "INT4", 8192, plateformes[0])
verifier("diagnostiquer : cas compatible, pas de suggestion", diag_ok["compat"]["compatible"] and diag_ok["suggestion_precision"] is None)

diag_precision = diagnostiquer(specs_api, "FP16", 8192, rpi5)
verifier("diagnostiquer : suggère INT4 quand FP16 ne rentre pas", diag_precision["suggestion_precision"] == "INT4")

specs_geant = {"nb_param": 500_000_000_000, "num_layers": 96, "num_kv_heads": 96, "head_dim": 128}
diag_multi = diagnostiquer(specs_geant, "INT4", 2048, plateformes[0])
verifier("diagnostiquer : suggère du multi-GPU quand rien ne rentre sur GPU discret", diag_multi["suggestion_multi_gpu"] is not None)

rpi4 = next(p for p in plateformes if p["name"] == "Raspberry Pi 4 8GB")
specs_gemma = get_specs("google/gemma-7b")
diag_alt = diagnostiquer(specs_gemma, "INT4", 8192, rpi4)
verifier("diagnostiquer : propose d'autres plateformes sur carte embarquée", len(diag_alt["plateformes_alternatives"]) > 0)

# --- historique ---
avant = len(charger_historique())
ajouter_historique("Mistral-7B-v0.1", "INT4", 8192, "NVIDIA A100 80GB", diag_ok)
apres = len(charger_historique())
verifier("ajouter_historique ajoute bien une ligne", apres == avant + 1)

# --- résumé ---
print()
if erreurs:
    print(f"{len(erreurs)} problème(s) trouvé(s) :", erreurs)
else:
    print("Tout est correct, aucun problème détecté.")
