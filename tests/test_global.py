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

# --- résumé ---
print()
if erreurs:
    print(f"{len(erreurs)} problème(s) trouvé(s) :", erreurs)
else:
    print("Tout est correct, aucun problème détecté.")
