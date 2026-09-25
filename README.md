# llm-modelfit

Estime la mémoire nécessaire pour faire tourner un modèle de langage (LLM) sur une plateforme donnée (GPU serveur ou carte embarquée), et propose des alternatives quand ça ne rentre pas.

Développé dans le cadre d'un stage sur l'estimation des besoins mémoire des LLM sur plateformes GPU et Edge.

## Fonctionnalités

- Récupère les caractéristiques d'un modèle directement depuis Hugging Face (n'importe quel modèle public, avec secours local pour les modèles à licence restreinte).
- Calcule la mémoire nécessaire : poids, cache KV, buffers, selon la précision choisie (FP32/FP16/INT8/INT4).
- Compare le résultat à une base de 17 plateformes (GPU datacenter, cartes embarquées) ou à une plateforme personnalisée.
- Si le modèle ne rentre pas : propose une précision plus légère, calcule le nombre de GPU nécessaires, ou liste les plateformes compatibles.
- Estime l'autonomie sur batterie et donne un repère de refroidissement nécessaire.
- Garde un historique des recherches.

## Installation

```bash
git clone https://github.com/<ton-compte>/llm-modelfit.git
cd llm-modelfit
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Ou directement depuis GitHub, sans cloner :
```bash
pip install git+https://github.com/<ton-compte>/llm-modelfit.git
```

### Configurer le token Hugging Face

Certains modèles (Llama, Gemma...) nécessitent un compte Hugging Face et un token d'accès. Crée un fichier `.env` à la racine du projet :
```
HF_TOKEN=ton_token_ici
```
Pour les modèles à licence restreinte ("gated"), il faut aussi accepter la licence sur la page du modèle (ex: https://huggingface.co/meta-llama/Meta-Llama-3-8B) avant de pouvoir le récupérer.

## Utilisation

```python
from llm_modelfit.modeles import get_specs
from llm_modelfit.comparaison import charger_plateformes, diagnostiquer

specs = get_specs("mistralai/Mistral-7B-v0.1")
plateformes = {p["name"]: p for p in charger_plateformes()}

resultat = diagnostiquer(specs, "INT4", contexte=8192, plateforme=plateformes["NVIDIA A100 80GB"])
print(resultat)
```

`diagnostiquer` renvoie un dictionnaire avec le détail de la mémoire (poids/cache KV/buffers), la compatibilité avec la plateforme choisie, et — si ça ne rentre pas — une suggestion (précision plus légère, nombre de GPU nécessaires, ou autres plateformes compatibles).

## Structure du projet

```
llm_modelfit/
├── fetch/huggingface.py   # récupération des modèles depuis Hugging Face
├── modeles.py             # get_specs() : API + secours local
├── engine.py              # formules de calcul mémoire
├── comparaison.py         # compatibilité, suggestions, diagnostic
├── historique.py          # historique des recherches
└── data/
    ├── models.json        # secours pour les modèles gated/refusés
    └── platforms.json     # base de plateformes (GPU + embarqué)
```

## Limites connues

- Les valeurs des cartes embarquées (Jetson, Raspberry Pi) et des modèles gated dans `models.json` viennent de fiches techniques publiques, pas d'une API — à vérifier si une précision fine est nécessaire.
- Le repère "chaleur" n'est pas une simulation thermique, juste une indication à partir de la puissance électrique.
- La base de modèles/plateformes est volontairement ciblée (les plus utilisés), pas exhaustive.

## Tests

```bash
python3 tests/test_global.py
```
