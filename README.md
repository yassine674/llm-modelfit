# llm-modelfit

Estimation de la mémoire nécessaire pour faire tourner un modèle de langage (LLM), comparaison avec de vraies plateformes (GPU serveur ou carte embarquée), et suggestions automatiques quand ça ne rentre pas.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-en%20développement-orange)

## Fonctionnalités

- **N'importe quel modèle** — récupère les caractéristiques directement depuis Hugging Face, avec secours local pour les modèles à licence restreinte.
- **Calcul détaillé** — poids, cache KV, buffers, selon la précision (FP32 / FP16 / INT8 / INT4).
- **17 plateformes** — des GPU de datacenter (A100, H100, MI300X...) aux cartes embarquées (Jetson, Raspberry Pi).
- **Suggestions automatiques** — si ça ne rentre pas : une précision plus légère, le nombre de GPU nécessaires, ou une autre plateforme.
- **Batterie et chaleur** — autonomie estimée et repère de refroidissement pour les déploiements embarqués.
- **Historique** — garde une trace de toutes les recherches effectuées.

## Installation

```bash
git clone https://github.com/yassine674/llm-modelfit.git
cd llm-modelfit
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Ou directement depuis GitHub, sans cloner :

```bash
pip install git+https://github.com/yassine674/llm-modelfit.git
```

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

---

Projet développé dans le cadre d'un stage sur l'estimation des besoins mémoire des LLM sur plateformes GPU et Edge.
