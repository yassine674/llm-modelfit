# Petit test en direct, dans la console : pose les questions qu'une interface
# graphique poserait, et utilise les vraies fonctions du projet pour répondre.
from llm_modelfit.modeles import get_specs
from llm_modelfit.engine import octets_vers_go
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
        return {"name": nom, "memory_gb": mem}
    return plateformes[int(choix) - 1]


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

    diagnostic = diagnostiquer(specs, precision, contexte, plateforme)

    print(f"\n=== Résultat pour {modele_id} ===")
    print(f"Plateforme conseillée à tester : {plateforme['name']}")
    detail = diagnostic["detail_go"]
    print(f"  Poids      : {detail['poids']:.2f} Go")
    print(f"  Cache KV   : {detail['cache_kv']:.2f} Go")
    print(f"  Buffers    : {detail['buffers']:.2f} Go")
    print(f"  Total      : {detail['total']:.2f} Go")

    compat = diagnostic["compat"]
    if compat["compatible"]:
        print(f"Compatible : OUI (occupe {compat['taux_go']}% de la mémoire)")
    else:
        print(f"Compatible : NON (demande {compat['taux_go']}% de la mémoire disponible)")
        if diagnostic["suggestion_precision"]:
            print(f"  Suggestion : passer en {diagnostic['suggestion_precision']}")
        if diagnostic["suggestion_multi_gpu"]:
            print(f"  Suggestion : utiliser {diagnostic['suggestion_multi_gpu']} GPU en parallèle")
        if diagnostic["plateformes_alternatives"]:
            print("  Autres plateformes compatibles :")
            for nom in diagnostic["plateformes_alternatives"]:
                print(f"    - {nom}")

    ajouter_historique(modele_id, precision, contexte, plateforme["name"], diagnostic)
    print("\nRecherche enregistrée dans l'historique.")


if __name__ == "__main__":
    main()
