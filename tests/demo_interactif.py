# Petit test en direct, dans la console : pose les questions qu'une interface
# graphique poserait, et utilise les vraies fonctions du projet pour répondre.
from llm_modelfit.modeles import get_specs
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer
from llm_modelfit.historique import ajouter_historique

PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]


def demander_precision():
    while True:
        p = input(f"Précision {PRECISIONS} : ").strip().upper()
        if p in PRECISIONS:
            return p
        print("Précision invalide, réessaie.")


def demander_contexte():
    while True:
        try:
            return int(input("Longueur de contexte (en tokens) : "))
        except ValueError:
            print("Entre un nombre entier.")


def choisir_plateforme(plateformes):
    print("\nPlateformes disponibles :")
    for i, p in enumerate(plateformes, 1):
        print(f"  {i}. {p['name']} ({p['memory_gb']} Go)")
    print("  0. Plateforme personnalisée")

    choix = input("Ton choix : ").strip()
    if choix == "0":
        nom = input("Nom de la plateforme : ")
        mem = float(input("Mémoire disponible (Go) : "))
        watts = float(input("Puissance électrique (Watts) : "))
        embarque = input("Carte embarquée, sur batterie ? (o/n) : ").strip().lower() == "o"
        return {"name": nom, "memory_gb": mem, "power_watts": watts, "unified_memory": embarque}
    return plateformes[int(choix) - 1]


def demander_batterie(plateforme):
    if not plateforme.get("unified_memory", False):
        return None
    reponse = input("Capacité de la batterie en Wh (laisser vide si non concerné) : ").strip()
    return float(reponse) if reponse else None


def main():
    print("=== Test en direct : estimation mémoire d'un LLM ===\n")

    modele_id = input("Modèle (identifiant Hugging Face ou nom local) : ").strip()
    print("\nRécupération des caractéristiques...")
    specs = get_specs(modele_id)
    print("Données récupérées :")
    for cle, valeur in specs.items():
        print(f"  {cle} : {valeur}")

    precision = demander_precision()
    contexte = demander_contexte()
    plateforme = choisir_plateforme(charger_plateformes())
    batterie = demander_batterie(plateforme)

    diagnostic = diagnostiquer(specs, precision, contexte, plateforme, capacite_batterie_wh=batterie)

    print(f"\n=== Résultat pour {modele_id} ===")
    print(f"Plateforme testée : {plateforme['name']}")
    detail = diagnostic["detail_go"]
    print(f"  Poids      : {detail['poids']:.2f} Go")
    print(f"  Cache KV   : {detail['cache_kv']:.2f} Go")
    print(f"  Buffers    : {detail['buffers']:.2f} Go")
    print(f"  Total      : {detail['total']:.2f} Go")

    print(f"Puissance de la plateforme : {diagnostic['power_watts']} W ({diagnostic['note_thermique']})")
    if diagnostic["autonomie_heures"]:
        print(f"Autonomie estimée sur cette batterie : {diagnostic['autonomie_heures']:.1f} h")

    compat = diagnostic["compat"]
    if compat["compatible"]:
        print(f"Compatible : OUI (occupe {compat['taux_go']}% de la mémoire)")
    else:
        print(f"Compatible : NON (demande {compat['taux_go']}% de la mémoire disponible)")
        print("\nMeilleures combinaisons trouvées (toutes plateformes confondues) :")
        for o in diagnostic["meilleures_options"][:5]:
            gpu = f"{o['nb_gpu']}x " if o["nb_gpu"] > 1 else ""
            print(f"  - {gpu}{o['plateforme']} en {o['precision']} ({o['power_watts']} W, {o['note_thermique']})")

    ajouter_historique(modele_id, precision, contexte, plateforme["name"], diagnostic)
    print("\nRecherche enregistrée dans l'historique.")


if __name__ == "__main__":
    main()
