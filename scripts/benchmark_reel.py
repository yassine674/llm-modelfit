# Compare l'estimation de l'outil à la mémoire réellement utilisée, en
# chargeant un vrai modèle et en le faisant générer à différentes longueurs
# de contexte. Fonctionne sur GPU CUDA (A100, Jetson) ou CPU (Raspberry Pi).
#
# Installation (depuis la racine du dépôt outil/) :
#   pip install -e .
#   pip install -r scripts/requirements-benchmark.txt
#   Sur Jetson : torch doit venir du wheel NVIDIA pour Jetson, pas de PyPI
#   (voir https://developer.nvidia.com/embedded/downloads#?search=PyTorch).
#   Retirer alors "torch" de requirements-benchmark.txt avant l'installation.
#
# Usage :
#   python scripts/benchmark_reel.py mistralai/Mistral-7B-v0.1 --precision FP16 --contextes 512 2048 8192
#   python scripts/benchmark_reel.py mistralai/Mistral-7B-v0.1 --precision FP32 --out resultats.csv

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    BNB_DISPONIBLE = True
except ImportError:
    BNB_DISPONIBLE = False

from llm_modelfit.engine import memoire_totale, octets_vers_go
from llm_modelfit.modeles import get_specs

DTYPE_PAR_PRECISION = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}


def detecter_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    # Raspberry Pi / Jetson (sans CUDA) : le modèle exact de la carte est
    # disponible via le device tree Linux, plus fiable que platform.processor()
    # qui peut renvoyer une valeur inexploitable selon la distribution.
    modele_dt = Path("/proc/device-tree/model")
    if modele_dt.exists():
        nom = modele_dt.read_bytes().decode(errors="ignore").strip("\x00").strip()
        if nom:
            return "cpu", nom
    return "cpu", platform.machine()


def charger_modele(modele_id, precision, device_type):
    if precision in DTYPE_PAR_PRECISION:
        dtype = DTYPE_PAR_PRECISION[precision]
        if device_type == "cpu" and dtype != torch.float32:
            raise ValueError(f"{precision} n'est pas fiable sur CPU (opérations non supportées) : utilisez FP32.")
        return AutoModelForCausalLM.from_pretrained(modele_id, torch_dtype=dtype)

    if precision in ("INT8", "INT4"):
        if device_type == "cpu":
            raise ValueError(f"{precision} nécessite un GPU CUDA (bitsandbytes) : indisponible sur CPU.")
        if not BNB_DISPONIBLE:
            raise ValueError("bitsandbytes n'est pas installé : INT8/INT4 indisponibles sur cette machine.")
        config = BitsAndBytesConfig(load_in_8bit=True) if precision == "INT8" else BitsAndBytesConfig(load_in_4bit=True)
        return AutoModelForCausalLM.from_pretrained(modele_id, quantization_config=config)

    raise ValueError(f"précision inconnue : {precision}")


def memoire_reelle_go(model, nb_tokens, device_type, vocab_size):
    # prompt de exactement nb_tokens tokens : peu importe leur sens,
    # seule la longueur compte pour remplir le cache KV comme en situation réelle
    prompt = torch.zeros((1, nb_tokens), dtype=torch.long, device=device_type)
    prompt.clamp_(max=vocab_size - 1)

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=1, do_sample=False)
        # memory_allocated (pas memory_reserved) : la mémoire réellement occupée par
        # les tenseurs, comparable à un calcul théorique. nvidia-smi affichera un
        # chiffre plus haut (contexte CUDA ~300-500 Mo + réserve de l'allocateur).
        return octets_vers_go(torch.cuda.max_memory_allocated())

    # CPU : pas d'équivalent à max_memory_allocated. On utilise le pic de mémoire
    # résidente du processus (ru_maxrss, en Ko sous Linux) après génération :
    # approximatif (inclut l'interpréteur Python), mais seule mesure disponible
    # sans dépendance supplémentaire sur une carte comme le Raspberry Pi.
    import resource
    with torch.no_grad():
        model.generate(prompt, max_new_tokens=1, do_sample=False)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modele", help="identifiant Hugging Face du modèle")
    parser.add_argument("--precision", default="FP32", choices=["FP32", "FP16", "BF16", "INT8", "INT4"])
    parser.add_argument("--contextes", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--out", help="fichier CSV où ajouter les résultats (créé s'il n'existe pas)")
    args = parser.parse_args()

    device_type, device_name = detecter_device()
    print(f"Machine détectée : {device_name} ({device_type})")
    print(f"Chargement de {args.modele} en {args.precision}...")

    specs = get_specs(args.modele)
    tokenizer = AutoTokenizer.from_pretrained(args.modele)
    model = charger_modele(args.modele, args.precision, device_type)
    if device_type == "cuda":
        model = model.to("cuda")
    model.eval()

    lignes = []
    print(f"\n{'Contexte':>10} {'Estime (Go)':>14} {'Mesure (Go)':>14} {'Ecart':>10}")
    for contexte in args.contextes:
        estime = octets_vers_go(memoire_totale(specs, args.precision, contexte)["total"])
        mesure = memoire_reelle_go(model, contexte, device_type, tokenizer.vocab_size)
        ecart_pct = (mesure - estime) / mesure * 100
        print(f"{contexte:>10} {estime:>14.2f} {mesure:>14.2f} {ecart_pct:>9.1f}%")
        lignes.append((device_name, args.modele, args.precision, contexte, round(estime, 2), round(mesure, 2), round(ecart_pct, 1)))

    if args.out:
        import csv
        nouveau = not Path(args.out).exists()
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if nouveau:
                w.writerow(["machine", "modele", "precision", "contexte", "estime_go", "mesure_go", "ecart_pct"])
            w.writerows(lignes)
        print(f"\nRésultats ajoutés à {args.out}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
