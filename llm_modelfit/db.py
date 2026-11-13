# connexion à la base SQLite partagée par tout le projet
# la base vit dans le dossier de données utilisateur (pas dans le paquet installé,
# qu'une réinstallation écraserait) ; LLM_MODELFIT_DB permet d'imposer un autre chemin
# si la table platforms est vide, elle est peuplée à partir de data/platforms.json et data/models.json
# les bases déjà créées sont mises à niveau par _migrer() (numéro de version dans PRAGMA user_version)
import json
import os
import sqlite3
from pathlib import Path

from platformdirs import user_data_dir

CHEMIN_DATA = Path(__file__).parent / "data"


def chemin_db():
    chemin = os.environ.get("LLM_MODELFIT_DB")
    if chemin:
        return Path(chemin)
    return Path(user_data_dir("llm_modelfit", appauthor=False)) / "llm_modelfit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS platforms (
    name TEXT PRIMARY KEY,
    memory_gb REAL NOT NULL,
    memory_type TEXT,
    bandwidth_gb_s REAL,  -- bande passante mémoire en gigaoctets par seconde
    power_watts REAL NOT NULL,
    has_tensor_cores INTEGER NOT NULL DEFAULT 0,
    unified_memory INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS models (
    hf_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT,
    params INTEGER NOT NULL,
    hidden_size INTEGER,
    num_attention_heads INTEGER,
    num_key_value_heads INTEGER,
    num_hidden_layers INTEGER NOT NULL,
    vocab_size INTEGER NOT NULL,
    max_position_embeddings INTEGER,
    quantizations_available TEXT,
    config_json TEXT  -- config brute, relue par normalize_config comme une config Hugging Face
);

-- configs déjà récupérées sur Hugging Face : permet de réutiliser un modèle hors ligne
CREATE TABLE IF NOT EXISTS cache_configs (
    hf_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    nb_param INTEGER NOT NULL,
    date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS historique (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    modele TEXT NOT NULL,
    "precision" TEXT NOT NULL,
    contexte INTEGER NOT NULL,
    plateforme TEXT NOT NULL,
    poids_go REAL NOT NULL,
    cache_kv_go REAL NOT NULL,
    buffers_go REAL NOT NULL,
    total_go REAL NOT NULL,
    compatible INTEGER NOT NULL,
    taux_go REAL NOT NULL,
    power_watts REAL,
    note_thermique TEXT,
    autonomie_heures REAL
);

CREATE TABLE IF NOT EXISTS historique_options (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    historique_id INTEGER NOT NULL REFERENCES historique(id),
    plateforme TEXT NOT NULL,
    "precision" TEXT NOT NULL,
    nb_gpu INTEGER NOT NULL,
    total_go REAL,
    power_watts REAL,
    note_thermique TEXT
);
"""


def _inserer_modeles(conn):
    # INSERT OR REPLACE : relancé par une migration, met aussi à jour les modèles déjà présents
    with open(CHEMIN_DATA / "models.json") as f:
        for m in json.load(f):
            conn.execute(
                "INSERT OR REPLACE INTO models "
                "(hf_id, name, family, params, hidden_size, num_attention_heads, num_key_value_heads, "
                "num_hidden_layers, vocab_size, max_position_embeddings, quantizations_available, config_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m["hf_id"], m["name"], m.get("family"), m["params"], m.get("hidden_size"),
                    m.get("num_attention_heads"), m.get("num_key_value_heads"), m["num_hidden_layers"],
                    m["vocab_size"], m.get("max_position_embeddings"),
                    json.dumps(m.get("quantizations_available", [])),
                    json.dumps(m),
                ),
            )


def _peupler(conn):
    with open(CHEMIN_DATA / "platforms.json") as f:
        for p in json.load(f):
            conn.execute(
                "INSERT INTO platforms "
                "(name, memory_gb, memory_type, bandwidth_gb_s, power_watts, has_tensor_cores, unified_memory) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    p["name"], p["memory_gb"], p.get("memory_type"), p.get("bandwidth_gb_s"),
                    p["power_watts"], int(p.get("has_tensor_cores", False)), int(p.get("unified_memory", False)),
                ),
            )
    _inserer_modeles(conn)
    conn.commit()


def _colonnes(conn, table):
    return {ligne[1] for ligne in conn.execute(f"PRAGMA table_info({table})")}


def _migration_1(conn):
    # bandwidth_gbps contenait déjà des Go/s : le nom faisait croire à des gigabits
    if "bandwidth_gbps" in _colonnes(conn, "platforms"):
        conn.execute("ALTER TABLE platforms RENAME COLUMN bandwidth_gbps TO bandwidth_gb_s")


def _migration_2(conn):
    # config brute des modèles (fenêtre glissante, head_dim...) et rafraîchissement des
    # modèles de référence, sinon les corrections de models.json n'atteignent jamais une base existante
    if "config_json" not in _colonnes(conn, "models"):
        conn.execute("ALTER TABLE models ADD COLUMN config_json TEXT")
    _inserer_modeles(conn)


MIGRATIONS = [_migration_1, _migration_2]


def _migrer(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for numero, migration in enumerate(MIGRATIONS[version:], start=version + 1):
        migration(conn)
        conn.execute(f"PRAGMA user_version = {numero}")
    conn.commit()


def connexion():
    chemin = chemin_db()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(chemin)
    conn.row_factory = sqlite3.Row
    # CREATE TABLE IF NOT EXISTS : sans effet sur les tables existantes,
    # mais crée celles ajoutées au schéma depuis la création de la base
    conn.executescript(SCHEMA)
    if conn.execute("SELECT COUNT(*) FROM platforms").fetchone()[0] == 0:
        _peupler(conn)
    _migrer(conn)
    return conn
