# Interface web : formulaire (index) -> résultats (dashboard) -> historique
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, redirect, url_for

from llm_modelfit.modeles import get_specs
from llm_modelfit.engine import octets_vers_go, memoire_totale
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer, comparer_toutes_plateformes
from llm_modelfit.historique import ajouter_historique, charger_historique

app = Flask(__name__)

PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]
CONTEXTES = [2048, 4096, 8192, 16384, 32768]
CHEMIN_MODELES = Path(__file__).parent.parent / "llm_modelfit" / "data" / "models.json"


def modeles_connus():
    return json.load(open(CHEMIN_MODELES))


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

    plateformes = charger_plateformes()
    plateforme = next(p for p in plateformes if p["name"] == plateforme_nom)

    specs = get_specs(modele_id)
    diagnostic = diagnostiquer(specs, precision, contexte, plateforme)
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
