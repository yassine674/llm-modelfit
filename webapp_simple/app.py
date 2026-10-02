# Version simple de l'interface, sans décoration : sert à montrer que le calcul
# fonctionne via une interface graphique, avant de la rendre jolie plus tard.
# Ne dépend que du package llm_modelfit installé (pip install), pas d'un
# dossier voisin : ce fichier + templates/ peuvent être envoyés seuls.
from flask import Flask, render_template, request

from llm_modelfit.modeles import get_specs, lister_modeles
from llm_modelfit.engine import octets_vers_go
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer, comparer_toutes_plateformes
from llm_modelfit.historique import ajouter_historique, charger_historique

app = Flask(__name__)

PRECISIONS = ["FP32", "FP16", "INT8", "INT4"]
CONTEXTES = [2048, 4096, 8192, 16384, 32768]


def modeles_connus():
    return lister_modeles()


@app.route("/", methods=["GET", "POST"])
def index():
    resultat = None
    comparaisons = None
    erreur = None

    if request.method == "POST":
        modele_id = request.form.get("modele_custom") or request.form.get("modele")
        precision = request.form.get("precision")
        contexte = int(request.form.get("contexte"))
        plateforme_nom = request.form.get("plateforme")
        batterie = request.form.get("batterie_wh")
        batterie = float(batterie) if batterie else None

        plateformes = charger_plateformes()
        plateforme = next(p for p in plateformes if p["name"] == plateforme_nom)

        try:
            specs = get_specs(modele_id)
            resultat = diagnostiquer(specs, precision, contexte, plateforme, capacite_batterie_wh=batterie)
            ajouter_historique(modele_id, precision, contexte, plateforme["name"], resultat)
            comparaisons = comparer_toutes_plateformes(resultat["detail_go"]["total"])
        except Exception as e:
            erreur = str(e)

        return render_template(
            "index.html",
            modeles=modeles_connus(), precisions=PRECISIONS, contextes=CONTEXTES,
            plateformes=charger_plateformes(),
            resultat=resultat, comparaisons=comparaisons, erreur=erreur,
            modele_id=modele_id, precision=precision, contexte=contexte, plateforme_nom=plateforme_nom,
        )

    return render_template(
        "index.html",
        modeles=modeles_connus(), precisions=PRECISIONS, contextes=CONTEXTES,
        plateformes=charger_plateformes(),
        resultat=None, comparaisons=None, erreur=None,
        modele_id=None, precision=None, contexte=8192, plateforme_nom=None,
    )


@app.route("/historique")
def historique():
    return render_template("historique.html", entrees=list(reversed(charger_historique())))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
