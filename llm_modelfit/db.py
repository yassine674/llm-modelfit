# connexion à la base SQLite partagée par tout le projet
# au premier lancement (fichier .db absent), la base est créée et peuplée
# à partir des fixtures data/platforms.json et data/models.json
import json
import sqlite3
from pathlib import Path

CHEMIN_DATA = Path(__file__).parent / "data"
CHEMIN_DB = CHEMIN_DATA / "llm_modelfit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS platforms (
    name TEXT PRIMARY KEY,
    memory_gb REAL NOT NULL,
    memory_type TEXT,
    bandwidth_gbps REAL,
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
    quantizations_available TEXT
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


def _initialiser(conn):
    conn.executescript(SCHEMA)

    with open(CHEMIN_DATA / "platforms.json") as f:
        for p in json.load(f):
            conn.execute(
                "INSERT INTO platforms "
                "(name, memory_gb, memory_type, bandwidth_gbps, power_watts, has_tensor_cores, unified_memory) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    p["name"], p["memory_gb"], p.get("memory_type"), p.get("bandwidth_gbps"),
                    p["power_watts"], int(p.get("has_tensor_cores", False)), int(p.get("unified_memory", False)),
                ),
            )

    with open(CHEMIN_DATA / "models.json") as f:
        for m in json.load(f):
            conn.execute(
                "INSERT INTO models "
                "(hf_id, name, family, params, hidden_size, num_attention_heads, num_key_value_heads, "
                "num_hidden_layers, vocab_size, max_position_embeddings, quantizations_available) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m["hf_id"], m["name"], m.get("family"), m["params"], m.get("hidden_size"),
                    m.get("num_attention_heads"), m.get("num_key_value_heads"), m["num_hidden_layers"],
                    m["vocab_size"], m.get("max_position_embeddings"),
                    json.dumps(m.get("quantizations_available", [])),
                ),
            )

    conn.commit()


def connexion():
    premiere_fois = not CHEMIN_DB.exists()
    conn = sqlite3.connect(CHEMIN_DB)
    conn.row_factory = sqlite3.Row
    if premiere_fois:
        _initialiser(conn)
    return conn
