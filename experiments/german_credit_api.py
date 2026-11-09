from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID
import json
import sqlite3
import hashlib

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models/modelo_seed42.joblib"
DB_PATH = BASE / "results/auditoria_german_credit.db"

MODEL_ID = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

modelo = joblib.load(MODEL_PATH)
colunas = list(modelo.feature_names_in_)

app = FastAPI(
    title="Audit-Bot — German Credit Experimental",
    version="0.1.0-experimental",
)


class Decisao(BaseModel):
    transaction_id: str
    source_system: str
    features: dict[str, str | int | float]
    protected_attributes: dict[str, str]
    prediction: int = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)


def conectar():
    return sqlite3.connect(DB_PATH)


def iniciar_banco():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transacoes (
                transaction_id TEXT PRIMARY KEY,
                received_at TEXT NOT NULL,
                model_id TEXT NOT NULL,
                grupo TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                score REAL NOT NULL,
                features_json TEXT NOT NULL
            )
        """)

        # Impede alterações nas transações
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS transacoes_no_update
            BEFORE UPDATE ON transacoes
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'Transações são append-only'
                );
            END;
        """)

        # Impede exclusões de transações
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS transacoes_no_delete
            BEFORE DELETE ON transacoes
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'Transações não podem ser excluídas'
                );
            END;
        """)


iniciar_banco()


@app.get("/health")
def health():
    with conectar() as conn:
        quantidade = conn.execute("SELECT COUNT(*) FROM transacoes").fetchone()[0]

    return {
        "status": "ok",
        "model_loaded": True,
        "model_sha256": MODEL_ID,
        "transactions_audited": quantidade,
    }


@app.post("/webhook/decisions")
def receber_decisao(payload: Decisao):
    try:
        identificador = UUID(payload.transaction_id)

        if identificador.version != 4:
            raise ValueError("UUID não é versão 4")
    except ValueError:
        raise HTTPException(422, "UUID v4 inválido")

    if payload.source_system != "german-credit-seed42":
        raise HTTPException(422, "Sistema de origem inesperado")

    if set(payload.features) != set(colunas):
        raise HTTPException(422, "Atributos incompatíveis com o modelo")

    grupo = payload.protected_attributes.get("faixa_etaria")

    if set(payload.protected_attributes) != {"faixa_etaria"} or grupo not in {
        "Menos de 25",
        "25 ou mais",
    }:
        raise HTTPException(422, "Grupo etário inválido")

    # Reconstruir as entradas na ordem do treinamento
    entrada = pd.DataFrame([payload.features], columns=colunas)

    try:
        probabilidade = float(modelo.predict_proba(entrada)[0, 1])
    except (ValueError, TypeError) as erro:
        raise HTTPException(422, f"Entrada inválida: {erro}")

    if not np.isfinite(probabilidade):
        raise HTTPException(422, "Escore não finito")

    classe = int(probabilidade >= 0.5)

    # Confirmar que a cópia recebida corresponde
    # à decisão produzida pelo modelo versionado
    if classe != payload.prediction or abs(probabilidade - payload.score) > 1e-9:
        raise HTTPException(422, "Predição ou escore inconsistente")

    try:
        with conectar() as conn:
            conn.execute(
                """
                INSERT INTO transacoes
                (transaction_id, received_at, model_id,
                 grupo, prediction, score, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.transaction_id,
                    datetime.now(timezone.utc).isoformat(),
                    MODEL_ID,
                    grupo,
                    classe,
                    probabilidade,
                    json.dumps(payload.features, ensure_ascii=False),
                ),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Transação já registrada")

    return {
        "transaction_id": payload.transaction_id,
        "prediction": classe,
        "score": probabilidade,
        "model_sha256": MODEL_ID,
        "stored": True,
    }


@app.get("/governance/summary")
def resumo():
    with conectar() as conn:
        linhas = conn.execute("""
            SELECT grupo,
                   COUNT(*) AS total,
                   SUM(prediction) AS favoraveis,
                   AVG(prediction) AS taxa_selecao
            FROM transacoes
            GROUP BY grupo
        """).fetchall()

    return {
        "total_transactions": sum(r[1] for r in linhas),
        "groups": [
            {
                "grupo": r[0],
                "total": r[1],
                "favoraveis": r[2],
                "taxa_selecao": r[3],
            }
            for r in linhas
        ],
    }


@app.get("/governance/fairness")
def consultar_equidade():
    with conectar() as conn:
        linhas = conn.execute("""
            SELECT grupo,
                   COUNT(*) AS total,
                   SUM(prediction) AS favoraveis
            FROM transacoes
            GROUP BY grupo
        """).fetchall()

    grupos = {
        grupo: {
            "total": total,
            "favoraveis": favoraveis,
            "taxa_selecao": favoraveis / total,
            "amostra_reduzida": total < 30,
        }
        for grupo, total, favoraveis in linhas
    }

    jovem = grupos.get("Menos de 25")
    referencia = grupos.get("25 ou mais")

    if jovem is None or referencia is None:
        return {
            "status": "insufficient_groups",
            "total_transactions": sum(g["total"] for g in grupos.values()),
            "groups": grupos,
            "DPD": None,
            "DI": None,
            "alerta_DPD": False,
        }

    sr_jovem = jovem["taxa_selecao"]
    sr_referencia = referencia["taxa_selecao"]

    dpd = abs(sr_jovem - sr_referencia)

    di = sr_jovem / sr_referencia if sr_referencia > 0 else None

    return {
        "status": "calculated",
        "total_transactions": sum(g["total"] for g in grupos.values()),
        "groups": grupos,
        "DPD": dpd,
        "DI": di,
        "limiar_DPD": 0.10,
        "alerta_DPD": bool(dpd > 0.10),
        "grupo_referencia": "25 ou mais",
    }


# ============================================
# RF03 — ALERTAS DE GOVERNANÇA
# ============================================

LIMIAR_DPD = 0.10
LOTE_EXPERIMENTAL = "german-credit-seed42"


def iniciar_tabela_alertas():
    with conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                lote TEXT NOT NULL,
                model_id TEXT NOT NULL,
                tipo TEXT NOT NULL,
                grupo_afetado TEXT NOT NULL,
                grupo_referencia TEXT NOT NULL,
                dpd REAL NOT NULL,
                di REAL,
                limiar REAL NOT NULL,
                total_transacoes INTEGER NOT NULL,
                mensagem TEXT NOT NULL,
                UNIQUE(lote, model_id, tipo)
            )
        """)

        # Impedir alterações e exclusões pela interface SQL
        # convencional, preservando registros anteriores.
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS
            alertas_no_update
            BEFORE UPDATE ON alertas
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'Alertas são append-only'
                );
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS
            alertas_no_delete
            BEFORE DELETE ON alertas
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'Alertas não podem ser excluídos'
                );
            END;
        """)


iniciar_tabela_alertas()


@app.post("/governance/evaluate-alerts")
def avaliar_alertas():
    with conectar() as conn:
        linhas = conn.execute(
            """
            SELECT grupo,
                   COUNT(*) AS total,
                   SUM(prediction) AS favoraveis
            FROM transacoes
            WHERE model_id = ?
            GROUP BY grupo
        """,
            (MODEL_ID,),
        ).fetchall()

        grupos = {
            grupo: {
                "total": total,
                "favoraveis": favoraveis,
                "taxa": favoraveis / total,
            }
            for grupo, total, favoraveis in linhas
        }

        # Verificar o lote fixo da semente 42.
        # Evita registrar um alerta a partir de
        # um conjunto parcial ou contaminado.
        esperado = {
            "Menos de 25": (47, 31),
            "25 ou mais": (253, 195),
        }

        for nome, (total, favoraveis) in esperado.items():
            g = grupos.get(nome)
            if g is None or g["total"] != total or g["favoraveis"] != favoraveis:
                raise HTTPException(
                    409, "Lote experimental incompleto ou inconsistente"
                )

        if sum(g["total"] for g in grupos.values()) != 300:
            raise HTTPException(409, "Quantidade inesperada de transações")

        if set(grupos) != set(esperado):
            raise HTTPException(409, "Grupos inesperados no lote")

        jovem = grupos["Menos de 25"]["taxa"]
        referencia = grupos["25 ou mais"]["taxa"]

        dpd = abs(jovem - referencia)
        di = jovem / referencia if referencia > 0 else None

        if dpd <= LIMIAR_DPD:
            return {
                "alerta": False,
                "DPD": dpd,
                "DI": di,
                "motivo": "DPD dentro do limite operacional",
            }

        mensagem = (
            f"Disparidade de seleção identificada: "
            f"grupo menor de 25 anos = {jovem:.2%}; "
            f"grupo de 25 anos ou mais = {referencia:.2%}. "
            f"DPD = {dpd:.4f}, acima do limite "
            f"operacional de {LIMIAR_DPD:.2f}. "
            f"Recomenda-se investigação humana. "
            f"Este alerta não comprova discriminação."
        )

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO alertas (
                created_at, lote, model_id, tipo,
                grupo_afetado, grupo_referencia,
                dpd, di, limiar,
                total_transacoes, mensagem
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now(timezone.utc).isoformat(),
                LOTE_EXPERIMENTAL,
                MODEL_ID,
                "DPD",
                "Menos de 25",
                "25 ou mais",
                dpd,
                di,
                LIMIAR_DPD,
                300,
                mensagem,
            ),
        )

        criado = cursor.rowcount == 1

        registro = conn.execute(
            """
            SELECT id FROM alertas
            WHERE lote = ?
              AND model_id = ?
              AND tipo = 'DPD'
        """,
            (LOTE_EXPERIMENTAL, MODEL_ID),
        ).fetchone()

    return {
        "alerta": True,
        "alert_id": registro[0],
        "novo_registro": criado,
        "DPD": dpd,
        "DI": di,
        "limiar": LIMIAR_DPD,
        "mensagem": mensagem,
    }


@app.get("/governance/experimental-alerts")
def listar_alertas_experimentais():
    with conectar() as conn:
        cursor = conn.execute("""
            SELECT id, created_at, lote, model_id,
                   tipo, grupo_afetado, grupo_referencia,
                   dpd, di, limiar, total_transacoes,
                   mensagem
            FROM alertas
            ORDER BY id DESC
        """)

        colunas = [descricao[0] for descricao in cursor.description]

        registros = [dict(zip(colunas, linha)) for linha in cursor.fetchall()]

    return {"total_alertas": len(registros), "alertas": registros}


# ============================================
# RF04 — EXPLICAÇÕES INDIVIDUAIS SHAP
# ============================================

from experiments.shap_service import GermanCreditExplainer

explicador_shap = GermanCreditExplainer(modelo)


def iniciar_tabela_explicacoes():
    with conectar() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS explicacoes (
                transaction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_id TEXT NOT NULL,
                explanation_json TEXT NOT NULL,
                FOREIGN KEY (transaction_id)
                    REFERENCES transacoes(transaction_id)
            )
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS explicacoes_no_update
            BEFORE UPDATE ON explicacoes
            BEGIN
                SELECT RAISE(
                    ABORT, 'Explicações são append-only'
                );
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS explicacoes_no_delete
            BEFORE DELETE ON explicacoes
            BEGIN
                SELECT RAISE(
                    ABORT, 'Explicações não podem ser excluídas'
                );
            END;
        """)


iniciar_tabela_explicacoes()


@app.post("/transactions/{transaction_id}/explain")
def gerar_explicacao(transaction_id: str):
    with conectar() as conn:
        transacao = conn.execute(
            """
            SELECT features_json, model_id
            FROM transacoes
            WHERE transaction_id = ?
        """,
            (transaction_id,),
        ).fetchone()

        if transacao is None:
            raise HTTPException(404, "Transação não encontrada")

        if transacao[1] != MODEL_ID:
            raise HTTPException(409, "Versão do modelo incompatível")

        existente = conn.execute(
            """
            SELECT explanation_json
            FROM explicacoes
            WHERE transaction_id = ?
        """,
            (transaction_id,),
        ).fetchone()

        if existente is not None:
            return {
                "transaction_id": transaction_id,
                "novo_registro": False,
                "model_id": MODEL_ID,
                "explicacao": json.loads(existente[0]),
            }

        features = json.loads(transacao[0])
        resultado = explicador_shap.explicar(features)

        # JSON padrão não deve armazenar NaN/Infinity.
        conteudo = json.dumps(
            resultado,
            ensure_ascii=False,
            allow_nan=False,
        )

        try:
            conn.execute(
                """
                INSERT INTO explicacoes (
                    transaction_id, created_at,
                    model_id, explanation_json
                ) VALUES (?, ?, ?, ?)
            """,
                (
                    transaction_id,
                    datetime.now(timezone.utc).isoformat(),
                    MODEL_ID,
                    conteudo,
                ),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Explicação registrada simultaneamente")

    return {
        "transaction_id": transaction_id,
        "novo_registro": True,
        "model_id": MODEL_ID,
        "explicacao": resultado,
    }


@app.get("/transactions/{transaction_id}/explanation")
def consultar_explicacao(transaction_id: str):
    with conectar() as conn:
        registro = conn.execute(
            """
            SELECT created_at, model_id, explanation_json
            FROM explicacoes
            WHERE transaction_id = ?
        """,
            (transaction_id,),
        ).fetchone()

    if registro is None:
        raise HTTPException(404, "Explicação ainda não registrada")

    return {
        "transaction_id": transaction_id,
        "created_at": registro[0],
        "model_id": registro[1],
        "explicacao": json.loads(registro[2]),
    }


# ============================================
# RF05 — RELATÓRIO PDF POR UUID
# ============================================

from fastapi import Response
from experiments.pdf_service import gerar_relatorio


@app.get("/transactions/{transaction_id}/report")
def exportar_relatorio(transaction_id: str):
    with conectar() as conn:
        conn.row_factory = sqlite3.Row

        transacao = conn.execute(
            """
            SELECT * FROM transacoes
            WHERE transaction_id = ?
        """,
            (transaction_id,),
        ).fetchone()

        if transacao is None:
            raise HTTPException(404, "Transação não encontrada")

        registro_shap = conn.execute(
            """
            SELECT explanation_json, model_id
            FROM explicacoes
            WHERE transaction_id = ?
        """,
            (transaction_id,),
        ).fetchone()

        if registro_shap is None:
            raise HTTPException(409, "Gere a explicação SHAP antes do relatório")

        if transacao["model_id"] != MODEL_ID or registro_shap["model_id"] != MODEL_ID:
            raise HTTPException(409, "Versão do modelo incompatível")

        alerta = conn.execute(
            """
            SELECT * FROM alertas
            WHERE lote = ?
              AND model_id = ?
              AND tipo = 'DPD'
            ORDER BY id DESC LIMIT 1
        """,
            (LOTE_EXPERIMENTAL, MODEL_ID),
        ).fetchone()

    equidade = consultar_equidade()

    if equidade["status"] != "calculated" or equidade["total_transactions"] != 300:
        raise HTTPException(409, "Lote experimental incompleto")

    pdf = gerar_relatorio(
        dict(transacao),
        json.loads(registro_shap["explanation_json"]),
        equidade,
        dict(alerta) if alerta else None,
    )

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="auditoria_{transaction_id}.pdf"'
        },
    )
