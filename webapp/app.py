# Interface web : formulaire (index) -> résultats (dashboard) -> historique
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, redirect, url_for

from llm_modelfit.modeles import get_specs, lister_modeles
from llm_modelfit.engine import octets_vers_go, memoire_totale, OCTETS_PAR_PRECISION
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer, comparer_toutes_plateformes
from llm_modelfit.historique import ajouter_historique, charger_historique

app = Flask(__name__)

PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]
CONTEXTES = [2048, 4096, 8192, 16384, 32768]


def modeles_connus():
    return lister_modeles()


@app.route("/")
def index():
    return render_template(
        "index.html",
        modeles=modeles_connus(),
        precisions=PRECISIONS,
        contextes=CONTEXTES,
        plateformes=charger_plateformes(),
    )


@app.route("/dashboard")
def dashboard():
    modele_id = request.args.get("modele_custom") or request.args.get("modele")
    precision = request.args.get("precision", "FP16")
    contexte = int(request.args.get("contexte", 8192))
    plateforme_nom = request.args.get("plateforme")
    batterie = request.args.get("batterie_wh")
    batterie = float(batterie) if batterie else None

    plateformes = charger_plateformes()
    nom_custom = request.args.get("plateforme_custom_nom")
    if nom_custom:
        plateforme = {
            "name": nom_custom,
            "memory_gb": float(request.args.get("plateforme_custom_mem")),
            "power_watts": float(request.args.get("plateforme_custom_watts")),
            "unified_memory": request.args.get("plateforme_custom_embarque") == "oui",
        }
    else:
        plateforme = next(p for p in plateformes if p["name"] == plateforme_nom)

    specs = get_specs(modele_id)
    diagnostic = diagnostiquer(specs, precision, contexte, plateforme, capacite_batterie_wh=batterie)
    ajouter_historique(modele_id, precision, contexte, plateforme["name"], diagnostic)

    courbe = [
        {"contexte": c, "total_go": octets_vers_go(memoire_totale(specs, precision, c)["total"])}
        for c in CONTEXTES
    ]

    comparaisons = comparer_toutes_plateformes(diagnostic["detail_go"]["total"])
    historique_recent = list(reversed(charger_historique()))[:3]

    return render_template(
        "dashboard.html",
        modele_id=modele_id,
        precision=precision,
        contexte=contexte,
        plateforme=plateforme,
        specs=specs,
        octets_precision=OCTETS_PAR_PRECISION[precision],
        diagnostic=diagnostic,
        courbe=courbe,
        contextes=CONTEXTES,
        comparaisons=comparaisons,
        modeles=modeles_connus(),
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
    return render_template("historique.html", entrees=entrees)


if __name__ == "__main__":
    app.run(debug=True)
