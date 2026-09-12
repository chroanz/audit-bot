"""
Camada de persistencia do Audit-Bot.

RNF03 (Auditabilidade e Imutabilidade): os registros gravados aqui sao
"append-only". Isso e garantido em duas camadas:
  1) A aplicacao so expoe funcoes de INSERT (nunca UPDATE/DELETE) para as
     tabelas de log.
  2) Gatilhos (TRIGGERS) no proprio SQLite bloqueiam fisicamente qualquer
     tentativa de UPDATE ou DELETE nessas tabelas, mesmo que alguem
     acesse o banco diretamente -- dando respaldo juridico contra
     adulteracao (o requisito fala em seguranca juridica).

Toda decisao de acompanhamento (ex: DPO revisou um alerta) e registrada
como um NOVO registro na tabela alert_reviews, nunca como alteracao do
alerta original.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from app.config import DB_PATH

_IMMUTABLE_TABLES = ("transactions", "alerts", "alert_reviews")


def _install_immutability_triggers(conn: sqlite3.Connection) -> None:
    for table in _IMMUTABLE_TABLES:
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS trg_{table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'audit log e append-only: UPDATE nao permitido em {table}');
            END;
            """
        )
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS trg_{table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'audit log e append-only: DELETE nao permitido em {table}');
            END;
            """
        )


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                received_at TEXT NOT NULL,
                source_system TEXT,
                features_json TEXT NOT NULL,
                protected_attributes_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                score REAL,
                explanation_json TEXT NOT NULL,
                explanation_text TEXT NOT NULL,
                explanation_method TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                protected_attribute TEXT NOT NULL,
                group_value TEXT NOT NULL,
                reference_value TEXT NOT NULL,
                disparate_impact_ratio REAL NOT NULL,
                threshold REAL NOT NULL,
                group_sample_size INTEGER NOT NULL,
                message TEXT NOT NULL,
                sample_transaction_ids_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                note TEXT NOT NULL,
                FOREIGN KEY (alert_id) REFERENCES alerts (id)
            );
            """
        )
        _install_immutability_triggers(conn)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_transaction(
    *,
    transaction_id: str,
    source_system: Optional[str],
    features: dict[str, Any],
    protected_attributes: dict[str, Any],
    prediction: int,
    score: Optional[float],
    explanation: dict[str, Any],
    explanation_text: str,
    explanation_method: str,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO transactions (
                transaction_id, received_at, source_system, features_json,
                protected_attributes_json, prediction, score,
                explanation_json, explanation_text, explanation_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                _now(),
                source_system,
                json.dumps(features, ensure_ascii=False),
                json.dumps(protected_attributes, ensure_ascii=False),
                prediction,
                score,
                json.dumps(explanation, ensure_ascii=False),
                explanation_text,
                explanation_method,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_transaction(transaction_id: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        )
        return cur.fetchone()


def list_transactions(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def get_group_stats(protected_attribute: str) -> list[sqlite3.Row]:
    """Taxa de decisao favoravel (prediction=1) por valor do grupo protegido."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT
                json_extract(protected_attributes_json, ?) AS group_value,
                COUNT(*) AS n,
                AVG(CAST(prediction AS REAL)) AS favorable_rate
            FROM transactions
            WHERE json_extract(protected_attributes_json, ?) IS NOT NULL
            GROUP BY group_value
            """,
            (f"$.{protected_attribute}", f"$.{protected_attribute}"),
        )
        return cur.fetchall()


def list_distinct_protected_attributes() -> list[str]:
    with get_connection() as conn:
        cur = conn.execute("SELECT protected_attributes_json FROM transactions")
        keys: set[str] = set()
        for row in cur.fetchall():
            keys.update(json.loads(row["protected_attributes_json"]).keys())
        return sorted(keys)


def insert_alert(
    *,
    alert_type: str,
    protected_attribute: str,
    group_value: str,
    reference_value: str,
    disparate_impact_ratio: float,
    threshold: float,
    group_sample_size: int,
    message: str,
    sample_transaction_ids: list[str],
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (
                created_at, alert_type, protected_attribute, group_value,
                reference_value, disparate_impact_ratio, threshold,
                group_sample_size, message, sample_transaction_ids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                alert_type,
                protected_attribute,
                group_value,
                reference_value,
                disparate_impact_ratio,
                threshold,
                group_sample_size,
                message,
                json.dumps(sample_transaction_ids, ensure_ascii=False),
            ),
        )
        conn.commit()
        return cur.lastrowid


def has_recent_open_alert(protected_attribute: str, group_value: str) -> bool:
    """Evita spam de alertas repetidos para o mesmo grupo/atributo.

    MVP simplificado: considera "aberto" um alerta sem nenhum review
    associado ainda.
    """
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT a.id FROM alerts a
            LEFT JOIN alert_reviews r ON r.alert_id = a.id
            WHERE a.protected_attribute = ? AND a.group_value = ? AND r.id IS NULL
            LIMIT 1
            """,
            (protected_attribute, group_value),
        )
        return cur.fetchone() is not None


def list_alerts(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()


def get_alert(alert_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        return cur.fetchone()


def insert_alert_review(*, alert_id: int, reviewer: str, note: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO alert_reviews (alert_id, created_at, reviewer, note)
            VALUES (?, ?, ?, ?)
            """,
            (alert_id, _now(), reviewer, note),
        )
        conn.commit()
        return cur.lastrowid


def list_alert_reviews(alert_id: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM alert_reviews WHERE alert_id = ? ORDER BY id ASC",
            (alert_id,),
        )
        return cur.fetchall()


def count_transactions() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM transactions")
        return cur.fetchone()["n"]
