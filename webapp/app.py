# Interface web : formulaire (index) -> résultats (dashboard) -> historique
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request, redirect, url_for

from llm_modelfit.modeles import get_specs, lister_modeles
from llm_modelfit.engine import octets_vers_go, memoire_totale, mah_vers_wh, OCTETS_PAR_PRECISION
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer, comparer_toutes_plateformes
from llm_modelfit.historique import ajouter_historique, charger_historique
from llm_modelfit.fetch.huggingface import rechercher_modeles, rechercher_organisations, avatar_auteur

app = Flask(__name__)

PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]
CONTEXTES = [2048, 4096, 8192, 16384, 32768]


def _barres(n):
    # icône "barres de signal" : n barres sur 4 remplies
    barres = [("3", "14", "7"), ("8.5", "10", "11"), ("14", "6", "15"), ("19.5", "2", "19")]
    return "".join(
        f'<rect x="{x}" y="{y}" width="3" height="{h}" rx="1" fill="currentColor" opacity="{1 if i < n else 0.25}"></rect>'
        for i, (x, y, h) in enumerate(barres)
    )


# niveaux affichés dans le sélecteur de précision de la barre de recherche
NIVEAUX_PRECISION = [
    {"nom": "Faible", "precision": "INT4", "icone": _barres(1),
     "blurb": "Le moins de mémoire, qualité réduite. Pour le matériel limité."},
    {"nom": "Standard", "precision": "INT8", "icone": _barres(2),
     "blurb": "Bon compromis mémoire / qualité, courant pour le déploiement."},
    {"nom": "Élevé", "precision": "FP16", "icone": _barres(3),
     "blurb": "Précision native de la plupart des modèles, la plus utilisée."},
    {"nom": "Maximale", "precision": "FP32", "icone": _barres(4),
     "blurb": "Précision complète, le plus de mémoire. Rarement nécessaire."},
]

# raccourcis affichés sous forme de boutons au-dessus de la recherche de modèle
FAMILLES_MODELES = [
    {"nom": "Llama", "recherche": "llama", "org": "meta-llama"},
    {"nom": "Mistral", "recherche": "mistral", "org": "mistralai"},
    {"nom": "Gemma", "recherche": "gemma", "org": "google"},
    {"nom": "Qwen", "recherche": "qwen", "org": "Qwen"},
    {"nom": "Phi", "recherche": "phi", "org": "microsoft"},
]

# les 5 catégories les plus connues (style page Models de Hugging Face). Aucune
# n'est activée par défaut : une recherche sans tâche cochée cherche partout.
# Seule "Text to Text" correspond à une architecture que notre formule sait calculer
# (décodeur simple) ; les autres restent proposées pour la recherche, mais le calcul
# de mémoire échouera dessus (modèles image/audio, voir comparaison.py).
TACHES_MODELES = [
    {"nom": "Text to Text", "valeur": "text-generation", "icone": "edit", "couleur": "#f43f5e"},
    {"nom": "Text-to-Image", "valeur": "text-to-image", "icone": "image", "couleur": "#3b82f6"},
    {"nom": "Image-to-Text", "valeur": "image-to-text", "icone": "image", "couleur": "#3b82f6"},
    {"nom": "Image-to-Image", "valeur": "image-to-image", "icone": "image", "couleur": "#3b82f6"},
    {"nom": "Text-to-Speech", "valeur": "text-to-speech", "icone": "speech", "couleur": "#10b981"},
]


def modeles_connus():
    return lister_modeles()


def familles_avec_logos():
    familles = []
    for f in FAMILLES_MODELES:
        familles.append({**f, "logo": avatar_auteur(f["org"])})
    return familles


def plateformes_avec_images():
    plateformes = charger_plateformes()
    for p in plateformes:
        p["image"] = image_plateforme(p["name"])
    return plateformes


def image_plateforme(nom):
    nom = nom.lower()
    if "jetson" in nom:
        return "jetson.webp"
    if "raspberry" in nom:
        return "cpu.webp"
    if "h100" in nom:
        return "h100.webp"
    if " t4" in nom or nom.startswith("t4"):
        return "t4.webp"
    if "mi250" in nom or "mi300" in nom:
        return "mi300.webp"
    if "rtx" in nom:
        return "rtx-series.webp"
    return "gpu.webp"


@app.route("/api/recherche-modeles")
def api_recherche_modeles():
    q = request.args.get("q", "")
    tache = request.args.get("tache") or None
    with ThreadPoolExecutor(max_workers=2) as executor:
        futur_modeles = executor.submit(rechercher_modeles, q, tache=tache)
        futur_entreprises = executor.submit(rechercher_organisations, q)
        modeles = futur_modeles.result()
        entreprises = futur_entreprises.result()
    return jsonify({"modeles": modeles, "entreprises": entreprises})


@app.route("/")
def index():
    return render_template(
        "index.html",
        precisions=PRECISIONS,
        contextes=CONTEXTES,
        plateformes=plateformes_avec_images(),
        familles=familles_avec_logos(),
        taches=TACHES_MODELES,
        niveaux_precision=NIVEAUX_PRECISION,
    )


@app.route("/dashboard")
def dashboard():
    modele_id = request.args.get("modele")
    precision = request.args.get("precision", "FP16")
    contexte = int(request.args.get("contexte", 8192))
    plateforme_nom = request.args.get("plateforme")
    batterie_mah = request.args.get("batterie_mah")
    batterie = mah_vers_wh(float(batterie_mah)) if batterie_mah else None

    plateformes = plateformes_avec_images()
    nom_custom = request.args.get("plateforme_custom_nom")
    if nom_custom:
        plateforme = {
            "name": nom_custom,
            "memory_gb": float(request.args.get("plateforme_custom_mem")),
            "power_watts": float(request.args.get("plateforme_custom_watts")),
            "unified_memory": request.args.get("plateforme_custom_embarque") == "oui",
            "image": image_plateforme(nom_custom),
        }
    else:
        plateforme = next(p for p in plateformes if p["name"] == plateforme_nom)

    try:
        specs = get_specs(modele_id)
    except RuntimeError as e:
        return render_template(
            "index.html",
            precisions=PRECISIONS,
            contextes=CONTEXTES,
            plateformes=plateformes,
            familles=familles_avec_logos(),
            taches=TACHES_MODELES,
            niveaux_precision=NIVEAUX_PRECISION,
            erreur=str(e),
        ), 400

    avertissement = specs.pop("avertissement", None)
    diagnostic = diagnostiquer(specs, precision, contexte, plateforme, capacite_batterie_wh=batterie)
    ajouter_historique(modele_id, precision, contexte, plateforme["name"], diagnostic)

    courbe = [
        {"contexte": c, "total_go": octets_vers_go(memoire_totale(specs, precision, c)["total"])}
        for c in CONTEXTES
    ]

    comparaisons = comparer_toutes_plateformes(diagnostic["detail_go"]["total"])
    for c in comparaisons:
        c["image"] = image_plateforme(c["plateforme"])
    historique_recent = list(reversed(charger_historique()))[:3]

    total_go = diagnostic["detail_go"]["total"]
    segments = [
        {"nom": "Poids", "valeur": diagnostic["detail_go"]["poids"], "couleur": "#3b82f6"},
        {"nom": "Cache KV", "valeur": diagnostic["detail_go"]["cache_kv"], "couleur": "#22d3ee"},
        {"nom": "Buffers", "valeur": diagnostic["detail_go"]["buffers"], "couleur": "#94a3b8"},
    ]
    for s in segments:
        s["pct"] = round(s["valeur"] / total_go * 100, 1) if total_go else 0

    return render_template(
        "dashboard.html",
        modele_id=modele_id,
        precision=precision,
        contexte=contexte,
        plateforme=plateforme,
        specs=specs,
        avertissement=avertissement,
        octets_precision=OCTETS_PAR_PRECISION[precision],
        diagnostic=diagnostic,
        courbe=courbe,
        contextes=CONTEXTES,
        comparaisons=comparaisons,
        segments=segments,
        familles=familles_avec_logos(),
        taches=TACHES_MODELES,
        niveaux_precision=NIVEAUX_PRECISION,
        precisions=PRECISIONS,
        plateformes=plateformes,
        historique_recent=historique_recent,
    )


@app.route("/historique")
def historique():
    plateformes_par_nom = {p["name"]: p for p in charger_plateformes()}
    entrees = list(reversed(charger_historique()))
    for e in entrees:
        p = plateformes_par_nom.get(e["plateforme"])
        e["dispo_go"] = p["memory_gb"] if p else None
    return render_template(
        "historique.html",
        entrees=entrees,
        precisions=PRECISIONS,
        plateformes_utilisees=sorted({e["plateforme"] for e in entrees}),
    )


if __name__ == "__main__":
    # threaded=True : une recherche en cours (appels réseau HF) ne doit pas
    # mettre en file d'attente la requête suivante tapée dans la barre
    app.run(debug=True, threaded=True)
