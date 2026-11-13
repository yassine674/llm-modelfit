# Compare l'estimation de l'outil à la réalité, en chargeant un vrai modèle
# (transformers) et en le faisant générer à différentes longueurs de contexte.
# Fonctionne sur GPU CUDA (A100, Jetson) ou CPU (Raspberry Pi, Linux uniquement).
#
# Pour chaque contexte, le script mesure : la mémoire des poids, le pic de mémoire
# total pendant la génération, la latence du premier token (TTFT) et le débit de
# décodage (tokens/s), et compare les deux mémoires à l'estimation de l'outil.
#
# Attention : INT8/INT4 désignent ici la quantification bitsandbytes (INT4 = NF4),
# pas les formats GGUF de llama.cpp/Ollama. Les écarts mesurés avec ce script ne
# s'appliquent pas tels quels à un autre moteur d'inférence.
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
#   python scripts/benchmark_reel.py mistralai/Mistral-7B-v0.1 --precision FP32 --nouveaux-tokens 64 --out resultats.csv

import argparse
import ctypes
import csv
import gc
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    BNB_DISPONIBLE = True
except ImportError:
    BNB_DISPONIBLE = False

from llm_modelfit.engine import memoire_totale, octets_vers_go
from llm_modelfit.modeles import get_specs

DTYPE_PAR_PRECISION = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}

# transformers >= 4.56 attend "dtype" ; les versions plus anciennes (fréquentes
# sur Jetson, liées à JetPack) ne connaissent que "torch_dtype"
_version_tf = tuple(int(x) for x in transformers.__version__.split(".")[:2])
CLE_DTYPE = "dtype" if _version_tf >= (4, 56) else "torch_dtype"

ENTETE_CSV = [
    "machine", "modele", "precision", "precision_kv", "contexte", "nouveaux_tokens",
    "estime_poids_go", "mesure_poids_go", "estime_total_go", "mesure_total_go",
    "ratio_mesure_estime", "ecart_pct", "ttft_s", "debit_tokens_s",
]


def detecter_device():
    # Jetson / Raspberry Pi : le device tree Linux donne le nom complet de la carte
    # (sur Jetson, CUDA ne renvoie que "Orin"), plus fiable que platform.processor()
    nom_carte = None
    modele_dt = Path("/proc/device-tree/model")
    if modele_dt.exists():
        nom_carte = modele_dt.read_bytes().decode(errors="ignore").strip("\x00").strip() or None
    if torch.cuda.is_available():
        return "cuda", nom_carte or torch.cuda.get_device_name(0)
    return "cpu", nom_carte or platform.machine()


def precision_cache_kv(precision):
    # le cache KV est stocké dans le type de calcul du modèle, pas dans le format des
    # poids : avec bitsandbytes, les activations (donc K et V) restent en FP16
    return precision if precision in DTYPE_PAR_PRECISION else "FP16"


def charger_modele(modele_id, precision, device_type):
    if precision in DTYPE_PAR_PRECISION:
        dtype = DTYPE_PAR_PRECISION[precision]
        if device_type == "cpu" and dtype != torch.float32:
            raise ValueError(f"{precision} n'est pas fiable sur CPU (opérations non supportées) : utilisez FP32.")
        model = AutoModelForCausalLM.from_pretrained(modele_id, **{CLE_DTYPE: dtype})
        return model.to(device_type).eval()

    if precision in ("INT8", "INT4"):
        if device_type == "cpu":
            raise ValueError(f"{precision} nécessite un GPU CUDA (bitsandbytes) : indisponible sur CPU.")
        if not BNB_DISPONIBLE:
            raise ValueError("bitsandbytes n'est pas installé : INT8/INT4 indisponibles sur cette machine.")
        if precision == "INT8":
            config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        # bitsandbytes place lui-même le modèle sur le GPU (device_map) :
        # un .to("cuda") après coup lève une erreur avec un modèle quantifié
        model = AutoModelForCausalLM.from_pretrained(
            modele_id, quantization_config=config, device_map={"": 0}, **{CLE_DTYPE: torch.float16},
        )
        return model.eval()

    raise ValueError(f"précision inconnue : {precision}")


# --- mesure mémoire sur CPU (Linux) -------------------------------------------
# /proc/self/status donne la mémoire résidente actuelle (VmRSS) et son pic (VmHWM).
# Écrire "5" dans /proc/self/clear_refs remet le pic au niveau actuel : chaque
# contexte a ainsi son propre pic, au lieu du maximum de toute la vie du processus.

def _champ_status_go(champ):
    for ligne in Path("/proc/self/status").read_text().splitlines():
        if ligne.startswith(champ + ":"):
            return int(ligne.split()[1]) / (1024 ** 2)  # valeur en Ko
    raise ValueError(f"{champ} absent de /proc/self/status")


def verifier_mesure_cpu():
    if not Path("/proc/self/clear_refs").exists():
        raise ValueError("la mesure mémoire sur CPU nécessite Linux (/proc/self/clear_refs).")


class Memoire:
    # lit la mémoire occupée par le modèle, de la même façon sur CPU et sur GPU

    def __init__(self, device_type):
        self.device_type = device_type
        # sur CPU, le processus occupe déjà de la mémoire avant le modèle (Python,
        # torch, transformers) : on la retire de chaque mesure
        self.base_go = 0 if device_type == "cuda" else _champ_status_go("VmRSS")

    def actuelle_go(self):
        if self.device_type == "cuda":
            # memory_allocated (pas memory_reserved) : la mémoire réellement occupée par
            # les tenseurs, comparable à un calcul théorique. nvidia-smi affichera un
            # chiffre plus haut (contexte CUDA ~300-500 Mo + réserve de l'allocateur).
            return octets_vers_go(torch.cuda.memory_allocated())
        return _champ_status_go("VmRSS") - self.base_go

    def reinitialiser_pic(self):
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            return
        # l'allocateur C (glibc) garde la mémoire libérée par le contexte précédent au
        # lieu de la rendre au système : malloc_trim la rend, sinon le pic d'un petit
        # contexte mesuré après un grand serait gonflé
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass  # libc autre que glibc (ex. musl) : mesure un peu moins précise
        Path("/proc/self/clear_refs").write_text("5")

    def pic_go(self):
        if self.device_type == "cuda":
            return octets_vers_go(torch.cuda.max_memory_allocated())
        return _champ_status_go("VmHWM") - self.base_go


# --- génération ----------------------------------------------------------------

def generer(model, prompt, nb_tokens, pad_id):
    # min_new_tokens force exactement nb_tokens tokens, même si le modèle émet un
    # token de fin : le décodage mesuré a toujours la même longueur
    debut = time.perf_counter()
    with torch.no_grad():
        model.generate(
            prompt, attention_mask=torch.ones_like(prompt),
            max_new_tokens=nb_tokens, min_new_tokens=nb_tokens,
            do_sample=False, pad_token_id=pad_id,
        )
    if prompt.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - debut


def mesurer(model, memoire, contexte, nouveaux_tokens, pad_id):
    # prompt de exactement `contexte` tokens : peu importe leur sens,
    # seule la longueur compte pour remplir le cache KV comme en situation réelle
    prompt = torch.zeros((1, contexte), dtype=torch.long, device=model.device)

    # 1 seul token : prefill + premier token, c'est la latence ressentie (TTFT)
    ttft = generer(model, prompt, 1, pad_id)

    # génération complète : le pic couvre le prefill et tout le décodage
    memoire.reinitialiser_pic()
    duree = generer(model, prompt, nouveaux_tokens, pad_id)
    pic = memoire.pic_go()

    # le premier token sort du prefill : le débit de décodage porte sur les suivants
    debit = (nouveaux_tokens - 1) / (duree - ttft) if duree > ttft else None
    return pic, ttft, debit


def verifier_csv(chemin):
    # vérifié avant le benchmark, pas après : sur une Jetson, une mesure complète
    # peut prendre longtemps et ne doit pas être perdue à cause du fichier de sortie
    chemin = Path(chemin)
    if not chemin.exists():
        return
    with open(chemin, newline="") as f:
        entete = next(csv.reader(f), None)
    if entete != ENTETE_CSV:
        raise ValueError(
            f"{chemin} a été créé par une autre version du script (colonnes différentes) : "
            "choisissez un autre fichier avec --out pour ne pas mélanger les conventions."
        )


def ecrire_csv(chemin, lignes):
    chemin = Path(chemin)
    nouveau = not chemin.exists()
    with open(chemin, "a", newline="") as f:
        w = csv.writer(f)
        if nouveau:
            w.writerow(ENTETE_CSV)
        w.writerows(lignes)


def main():
    parser = argparse.ArgumentParser(description="Compare l'estimation mémoire de l'outil à la mesure réelle.")
    parser.add_argument("modele", help="identifiant Hugging Face du modèle")
    parser.add_argument("--precision", default="FP32", choices=["FP32", "FP16", "BF16", "INT8", "INT4"])
    parser.add_argument("--contextes", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--nouveaux-tokens", type=int, default=32,
                        help="tokens générés à chaque contexte pour mesurer le débit (au moins 2)")
    parser.add_argument("--out", help="fichier CSV où ajouter les résultats (créé s'il n'existe pas)")
    args = parser.parse_args()
    if args.nouveaux_tokens < 2:
        raise ValueError("--nouveaux-tokens doit valoir au moins 2 (1 pour le TTFT, au moins 1 pour le débit).")
    if args.out:
        verifier_csv(args.out)

    device_type, device_name = detecter_device()
    if device_type == "cpu":
        verifier_mesure_cpu()
    print(f"Machine détectée : {device_name} ({device_type})")
    print(f"Chargement de {args.modele} en {args.precision}...")

    specs = get_specs(args.modele)
    precision_kv = precision_cache_kv(args.precision)
    tokenizer = AutoTokenizer.from_pretrained(args.modele)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    memoire = Memoire(device_type)
    model = charger_modele(args.modele, args.precision, device_type)

    # génération d'échauffement : initialise les noyaux CUDA (sinon le premier TTFT est
    # faussé) et, sur CPU, force la lecture de tous les poids depuis le disque (chargés
    # à la demande) pour que la mesure des poids soit complète
    generer(model, torch.zeros((1, 8), dtype=torch.long, device=model.device), 2, pad_id)
    mesure_poids = memoire.actuelle_go()
    estime_poids = octets_vers_go(memoire_totale(specs, args.precision, 0, precision_kv)["poids"])
    print(f"Poids : estimé {estime_poids:.2f} Go, mesuré {mesure_poids:.2f} Go (cache KV en {precision_kv})")

    lignes = []
    print(f"\n{'Contexte':>9} {'Estimé (Go)':>12} {'Mesuré (Go)':>12} {'Écart':>8} {'TTFT (s)':>9} {'Débit (tok/s)':>14}")
    for contexte in args.contextes:
        # le cache KV contient le prompt et les tokens générés
        estime = octets_vers_go(
            memoire_totale(specs, args.precision, contexte + args.nouveaux_tokens, precision_kv)["total"]
        )
        mesure, ttft, debit = mesurer(model, memoire, contexte, args.nouveaux_tokens, pad_id)
        ratio = mesure / estime
        ecart_pct = (ratio - 1) * 100
        debit_txt = f"{debit:.1f}" if debit else "-"
        print(f"{contexte:>9} {estime:>12.2f} {mesure:>12.2f} {ecart_pct:>7.1f}% {ttft:>9.3f} {debit_txt:>14}")
        lignes.append((
            device_name, args.modele, args.precision, precision_kv, contexte, args.nouveaux_tokens,
            round(estime_poids, 3), round(mesure_poids, 3), round(estime, 3), round(mesure, 3),
            round(ratio, 4), round(ecart_pct, 1), round(ttft, 4), round(debit, 2) if debit else "",
        ))

    print("\nÉcart = (mesuré - estimé) / estimé : positif quand l'outil sous-estime.")
    if args.out:
        ecrire_csv(args.out, lignes)
        print(f"Résultats ajoutés à {args.out}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
